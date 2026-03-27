"""Tests for vectorized environment and rollout collector.

SubprocVecEnv tests run as subprocesses to avoid Windows multiprocessing
issues inside the pytest process. Buffer per-episode bootstrap tests
run in-process since they are pure numpy.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import torch

from crowdrl_train.buffer import RolloutBuffer


def _run_python(code: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run Python code in a subprocess."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


class TestSubprocVecEnv:
    """Tests for SubprocVecEnv — all run as subprocesses."""

    def test_create_and_close(self):
        """Create 2 workers and close without error."""
        result = _run_python("""
            from crowdrl_env.crowd_env import CrowdEnvConfig
            from crowdrl_env.spawner import SpawnConfig
            from crowdrl_train.vec_env import SubprocVecEnv

            config = CrowdEnvConfig(spawn=SpawnConfig(n_agents_range=(3, 5)))
            vec_env = SubprocVecEnv(config, seeds=[42, 43])
            assert vec_env.n_envs == 2
            vec_env.close()
            print("OK")
        """)
        assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        assert "OK" in result.stdout

    def test_reset_and_step(self):
        """Reset all envs and step with random actions."""
        result = _run_python("""
            import numpy as np
            from crowdrl_env.crowd_env import CrowdEnvConfig
            from crowdrl_env.spawner import SpawnConfig
            from crowdrl_train.vec_env import SubprocVecEnv

            config = CrowdEnvConfig(
                spawn=SpawnConfig(n_agents_range=(3, 5)),
                max_steps=10,
            )
            vec_env = SubprocVecEnv(config, seeds=[10, 11])

            # Reset
            results = vec_env.reset_all()
            assert len(results) == 2
            for obs, info in results:
                assert obs.ndim == 2
                assert info["n_agents"] >= 3

            # Step with random actions
            actions = []
            for obs, info in results:
                n = info["n_agents"]
                actions.append(np.random.uniform(-1, 1, (n, 4)))

            step_results = vec_env.step(actions)
            assert len(step_results) == 2
            for sr in step_results:
                assert sr.obs.ndim == 2
                assert sr.rewards.ndim == 1

            vec_env.close()
            print("OK")
        """)
        assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        assert "OK" in result.stdout

    def test_episode_completion(self):
        """Run until at least one env reaches episode_over."""
        result = _run_python("""
            import numpy as np
            from crowdrl_env.crowd_env import CrowdEnvConfig
            from crowdrl_env.spawner import SpawnConfig
            from crowdrl_train.vec_env import SubprocVecEnv

            config = CrowdEnvConfig(
                spawn=SpawnConfig(n_agents_range=(2, 3)),
                max_steps=5,  # very short episodes
            )
            vec_env = SubprocVecEnv(config, seeds=[42, 43])
            results = vec_env.reset_all()

            episode_over_seen = False
            for step in range(10):
                actions = []
                for obs, info in results:
                    n = obs.shape[0]
                    actions.append(np.random.uniform(-1, 1, (n, 4)))

                step_results = vec_env.step(actions)
                for sr in step_results:
                    if sr.info.get("episode_over", False):
                        episode_over_seen = True

                # Update obs for next step
                results = [(sr.obs, sr.info) for sr in step_results]

                if episode_over_seen:
                    break

            assert episode_over_seen, "Expected at least one episode to complete in 10 steps"
            vec_env.close()
            print("OK")
        """)
        assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        assert "OK" in result.stdout

    def test_reconfigure(self):
        """Reconfigure all workers with a new config."""
        result = _run_python("""
            from crowdrl_env.crowd_env import CrowdEnvConfig
            from crowdrl_env.spawner import SpawnConfig
            from crowdrl_train.vec_env import SubprocVecEnv

            config1 = CrowdEnvConfig(spawn=SpawnConfig(n_agents_range=(2, 3)))
            vec_env = SubprocVecEnv(config1, seeds=[42, 43])
            vec_env.reset_all()

            # Reconfigure with different agent range
            config2 = CrowdEnvConfig(spawn=SpawnConfig(n_agents_range=(5, 8)))
            results = vec_env.update_all_configs(config2, base_seed=100)

            # After reconfigure, all envs should have more agents
            for obs, info in results:
                assert info["n_agents"] >= 5, f"Expected >= 5 agents, got {info['n_agents']}"

            vec_env.close()
            print("OK")
        """)
        assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        assert "OK" in result.stdout


class TestRolloutCollector:
    """Test the multi-env rollout collector — run as subprocess."""

    def test_collect_returns_episodes(self):
        """Collect agent-steps and verify completed episodes are returned."""
        result = _run_python(
            """
            import torch
            import numpy as np
            from crowdrl_env.crowd_env import CrowdEnvConfig
            from crowdrl_env.spawner import SpawnConfig
            from crowdrl_train.config import NetworkConfig
            from crowdrl_train.networks import ActorCritic
            from crowdrl_train.normalizer import RunningNormalizer
            from crowdrl_train.vec_env import SubprocVecEnv
            from crowdrl_train.rollout_collector import RolloutCollector

            config = CrowdEnvConfig(
                spawn=SpawnConfig(n_agents_range=(3, 5)),
                max_steps=20,  # short episodes
            )
            vec_env = SubprocVecEnv(config, seeds=[42, 43])

            net_config = NetworkConfig(obs_dim=79, action_dim=4,
                                       actor_hidden_sizes=(32, 32),
                                       critic_hidden_sizes=(32, 32))
            actor_critic = ActorCritic(net_config)
            obs_norm = RunningNormalizer(shape=(79,))

            collector = RolloutCollector(
                vec_env, actor_critic, obs_norm, None, torch.device("cpu"),
                obs_dim=79, action_dim=4,
            )

            episodes = collector.collect(n_agent_steps=200)

            # Should have completed at least 1 episode
            assert len(episodes) >= 1, f"Expected completed episodes, got {len(episodes)}"

            # Each episode dict has expected keys
            for ep in episodes:
                assert "goal_rate" in ep
                assert "n_agents" in ep
                assert "episode_length" in ep
                assert ep["n_agents"] >= 3

            # Per-env buffers should have data
            assert collector.total_active_agent_steps > 0

            # GAE + flatten should work
            flat = collector.compute_gae_and_flatten(gamma=0.99, gae_lambda=0.95)
            assert flat.batch_size > 0

            vec_env.close()
            print(f"OK: {len(episodes)} episodes collected, batch_size={flat.batch_size}")
        """,
            timeout=180,
        )
        assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        assert "OK" in result.stdout


class TestBufferPerEpisodeBootstrap:
    """Test compute_gae with per-episode bootstrap values (in-process)."""

    def test_per_episode_bootstrap_zeros_matches_legacy(self):
        """Per-episode list of zeros should match legacy single-array behavior."""
        buf = RolloutBuffer(2, 1, torch.device("cpu"))
        rng = np.random.default_rng(42)

        # Add 2 episodes with different n_agents
        for ep in range(2):
            n_agents = 3 + ep  # 3 and 4
            for t in range(5):
                buf.add(
                    obs=rng.standard_normal((n_agents, 2)),
                    actions_raw=rng.standard_normal((n_agents, 1)),
                    log_probs=rng.standard_normal(n_agents),
                    rewards=rng.standard_normal(n_agents),
                    values=rng.standard_normal(n_agents),
                    dones=np.zeros(n_agents, dtype=bool),
                    active_mask=np.ones(n_agents, dtype=bool),
                )
            buf.mark_episode_end()

        # Compute GAE with per-episode bootstrap (all zeros = completed)
        bootstrap_vals = [
            np.zeros(3, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
        ]
        bootstrap_dones = [
            np.ones(3, dtype=np.bool_),
            np.ones(4, dtype=np.bool_),
        ]
        buf.compute_gae(bootstrap_vals, bootstrap_dones, gamma=0.99, gae_lambda=0.95)

        # All advantages should be finite
        for adv in buf._advantages:
            assert np.all(np.isfinite(adv))

        # Returns = advantages + values
        for i, (adv, val) in enumerate(zip(buf._advantages, buf._returns)):
            np.testing.assert_allclose(val, adv + buf._values[i])

    def test_nonzero_bootstrap_increases_returns(self):
        """Non-zero bootstrap values should increase returns for last episode."""
        buf = RolloutBuffer(2, 1, torch.device("cpu"))
        rng = np.random.default_rng(42)

        n_agents = 3
        for t in range(5):
            buf.add(
                obs=rng.standard_normal((n_agents, 2)),
                actions_raw=rng.standard_normal((n_agents, 1)),
                log_probs=rng.standard_normal(n_agents),
                rewards=np.zeros(n_agents),  # zero rewards
                values=np.ones(n_agents),  # constant value
                dones=np.zeros(n_agents, dtype=bool),
                active_mask=np.ones(n_agents, dtype=bool),
            )
        # Don't mark_episode_end — this is an incomplete episode

        # Zero bootstrap
        buf_zero = RolloutBuffer(2, 1, torch.device("cpu"))
        buf_zero._obs = [o.copy() for o in buf._obs]
        buf_zero._actions_raw = [a.copy() for a in buf._actions_raw]
        buf_zero._log_probs = [lp.copy() for lp in buf._log_probs]
        buf_zero._rewards = [r.copy() for r in buf._rewards]
        buf_zero._values = [v.copy() for v in buf._values]
        buf_zero._dones = [d.copy() for d in buf._dones]
        buf_zero._active_masks = [m.copy() for m in buf._active_masks]
        buf_zero._episode_starts = buf._episode_starts.copy()

        # GAE with zero bootstrap
        buf_zero.compute_gae(
            [np.zeros(n_agents, dtype=np.float64)],
            [np.ones(n_agents, dtype=np.bool_)],
            gamma=0.99,
            gae_lambda=0.95,
        )

        # GAE with positive bootstrap
        buf.compute_gae(
            [np.ones(n_agents, dtype=np.float64) * 5.0],
            [np.zeros(n_agents, dtype=np.bool_)],
            gamma=0.99,
            gae_lambda=0.95,
        )

        # With positive bootstrap, last-step return should be higher
        assert np.all(buf._returns[-1] > buf_zero._returns[-1])
