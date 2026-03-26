"""Integration tests: end-to-end training loop on tiny environments.

These tests verify that all components work together correctly, but do NOT
test convergence (too expensive for CI). They verify:
- The training loop completes without errors
- Losses are finite (not NaN)
- Checkpoint save/load roundtrips produce identical weights
- The collect → GAE → update pipeline handles real env data

NOTE: Some tests run as subprocesses to avoid PyTorch MKL/LAPACK access
violations that occur inside the pytest process on certain Windows builds.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch

from crowdrl_env.crowd_env import CrowdEnv

from crowdrl_train.buffer import RolloutBuffer
from crowdrl_train.config import PPOConfig
from crowdrl_train.curriculum import CurriculumManager
from crowdrl_train.mappo import MAPPOUpdater
from crowdrl_train.networks import ActorCritic
from crowdrl_train.normalizer import RunningNormalizer
from crowdrl_train.train import collect_episode, save_checkpoint, load_checkpoint


def _run_python(code: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run Python code in a subprocess (avoids pytest+torch Windows crashes)."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


class TestCollectAndUpdate:
    """Test the collect episode → PPO update pipeline with a real env."""

    def test_collect_episode(self, tiny_env: CrowdEnv, tiny_actor_critic: ActorCritic):
        """Collect one full episode and verify stats."""
        buffer = RolloutBuffer(79, 4, torch.device("cpu"))
        normalizer = RunningNormalizer(shape=(79,))

        ep_stats = collect_episode(
            tiny_env, tiny_actor_critic, buffer, normalizer, None, torch.device("cpu")
        )

        assert ep_stats["n_agents"] > 0
        assert ep_stats["episode_length"] > 0
        assert 0.0 <= ep_stats["goal_rate"] <= 1.0
        assert buffer.total_steps == ep_stats["episode_length"]

    def test_collect_and_update_cycle(self):
        """Full collect → GAE → PPO update cycle (subprocess to avoid Windows crash)."""
        result = _run_python("""
            import torch, numpy as np
            from crowdrl_env.crowd_env import CrowdEnv, CrowdEnvConfig
            from crowdrl_env.geometry_generator import GeometryConfig, GeometryTier
            from crowdrl_env.spawner import SpawnConfig
            from crowdrl_train.networks import ActorCritic
            from crowdrl_train.config import NetworkConfig, PPOConfig
            from crowdrl_train.buffer import RolloutBuffer
            from crowdrl_train.normalizer import RunningNormalizer
            from crowdrl_train.mappo import MAPPOUpdater
            from crowdrl_train.train import collect_episode

            cfg = CrowdEnvConfig(
                geometry=GeometryConfig(tier=GeometryTier.TIER_0, min_side=10.0, max_side=12.0),
                spawn=SpawnConfig(n_agents_range=(3, 5)), max_steps=20,
            )
            env = CrowdEnv(config=cfg, seed=42)
            nc = NetworkConfig(obs_dim=79, action_dim=4,
                               actor_hidden_sizes=(32, 32), critic_hidden_sizes=(32, 32))
            ac = ActorCritic(nc)
            buf = RolloutBuffer(79, 4, torch.device('cpu'))
            norm = RunningNormalizer(shape=(79,))
            updater = MAPPOUpdater(ac, PPOConfig(n_epochs=2, n_minibatches=1), torch.device('cpu'))

            stats = collect_episode(env, ac, buf, norm, None, torch.device('cpu'))
            n = stats['n_agents']
            buf.compute_gae(np.zeros(n), np.ones(n, dtype=np.bool_), 0.99, 0.95)
            batch = buf.flatten()
            assert batch.batch_size > 0
            metrics = updater.update(batch)
            assert np.isfinite(metrics['policy_loss'])
            assert np.isfinite(metrics['value_loss'])
            assert np.isfinite(metrics['entropy'])
            print('PASS')
        """)
        assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
        assert "PASS" in result.stdout

    def test_multiple_cycles(self):
        """Run 3 collect-update cycles without errors (subprocess)."""
        result = _run_python(
            """
            import torch, numpy as np
            from crowdrl_env.crowd_env import CrowdEnv, CrowdEnvConfig
            from crowdrl_env.geometry_generator import GeometryConfig, GeometryTier
            from crowdrl_env.spawner import SpawnConfig
            from crowdrl_train.networks import ActorCritic
            from crowdrl_train.config import NetworkConfig, PPOConfig
            from crowdrl_train.buffer import RolloutBuffer
            from crowdrl_train.normalizer import RunningNormalizer
            from crowdrl_train.mappo import MAPPOUpdater
            from crowdrl_train.train import collect_episode

            cfg = CrowdEnvConfig(
                geometry=GeometryConfig(tier=GeometryTier.TIER_0, min_side=10.0, max_side=12.0),
                spawn=SpawnConfig(n_agents_range=(3, 5)), max_steps=20,
            )
            env = CrowdEnv(config=cfg, seed=42)
            nc = NetworkConfig(obs_dim=79, action_dim=4,
                               actor_hidden_sizes=(32, 32), critic_hidden_sizes=(32, 32))
            ac = ActorCritic(nc)
            buf = RolloutBuffer(79, 4, torch.device('cpu'))
            norm = RunningNormalizer(shape=(79,))
            updater = MAPPOUpdater(ac, PPOConfig(n_epochs=2), torch.device('cpu'))

            for cycle in range(3):
                stats = collect_episode(env, ac, buf, norm, None, torch.device('cpu'))
                n = stats['n_agents']
                buf.compute_gae(np.zeros(n), np.ones(n, dtype=np.bool_), 0.99, 0.95)
                batch = buf.flatten()
                if batch.batch_size > 0:
                    metrics = updater.update(batch)
                    assert np.isfinite(metrics['policy_loss']), f'NaN at cycle {cycle}'
                buf.clear()
            print('PASS')
        """,
            timeout=120,
        )
        assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
        assert "PASS" in result.stdout


class TestCheckpointing:
    def test_save_load_roundtrip(
        self, tmp_path: Path, tiny_actor_critic: ActorCritic, tiny_env: CrowdEnv
    ):
        """Checkpoint save/load should preserve network weights."""
        from crowdrl_train.config import CurriculumConfig

        updater = MAPPOUpdater(tiny_actor_critic, PPOConfig(), torch.device("cpu"))
        normalizer = RunningNormalizer(shape=(79,))
        curriculum = CurriculumManager(CurriculumConfig())

        # Feed some data to normalizer
        normalizer.update(np.random.randn(50, 79))

        # Save
        path = tmp_path / "test_ckpt.pt"
        save_checkpoint(path, tiny_actor_critic, updater, normalizer, None, curriculum, 1000, 10)

        # Load into fresh instances
        from crowdrl_train.config import NetworkConfig

        fresh_ac = ActorCritic(
            NetworkConfig(
                obs_dim=79, action_dim=4, actor_hidden_sizes=(32, 32), critic_hidden_sizes=(32, 32)
            )
        )
        fresh_updater = MAPPOUpdater(fresh_ac, PPOConfig(), torch.device("cpu"))
        fresh_normalizer = RunningNormalizer(shape=(79,))
        fresh_curriculum = CurriculumManager(CurriculumConfig())

        total_steps, rollout_count = load_checkpoint(
            path, fresh_ac, fresh_updater, fresh_normalizer, None, fresh_curriculum
        )

        assert total_steps == 1000
        assert rollout_count == 10

        # Verify weights match
        for p1, p2 in zip(tiny_actor_critic.parameters(), fresh_ac.parameters()):
            assert torch.allclose(p1, p2)

        # Verify normalizer stats match
        np.testing.assert_array_equal(normalizer.mean, fresh_normalizer.mean)
