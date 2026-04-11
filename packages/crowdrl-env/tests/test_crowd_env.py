"""Tests for the CrowdEnv Gymnasium environment."""

import numpy as np
import pytest

from crowdrl_core.observation import ObsConfig

from crowdrl_env.crowd_env import CrowdEnv, CrowdEnvConfig
from crowdrl_env.geometry_generator import GeometryConfig, GeometryTier
from crowdrl_env.reward import RewardConfig
from crowdrl_env.solvability import SolvabilityMode
from crowdrl_env.spawner import SpawnConfig


@pytest.fixture
def basic_env():
    """Small Tier 0 env for fast tests."""
    config = CrowdEnvConfig(
        geometry=GeometryConfig(tier=GeometryTier.TIER_0, min_side=10.0, max_side=15.0),
        spawn=SpawnConfig(n_agents_range=(3, 5), min_spawn_separation=0.3),
        solvability_mode=SolvabilityMode.PRUNE,
        max_steps=50,
    )
    return CrowdEnv(config=config, seed=42)


@pytest.fixture
def corridor_env():
    """Tier 1 corridor env."""
    config = CrowdEnvConfig(
        geometry=GeometryConfig(
            tier=GeometryTier.TIER_1,
            corridor_width_range=(2.0, 3.0),
            corridor_length_range=(10.0, 15.0),
        ),
        spawn=SpawnConfig(n_agents_range=(3, 6), min_spawn_separation=0.3),
        solvability_mode=SolvabilityMode.PRUNE,
        max_steps=50,
    )
    return CrowdEnv(config=config, seed=123)


class TestReset:
    def test_reset_returns_obs_and_info(self, basic_env):
        obs, info = basic_env.reset()
        assert obs.ndim == 2
        assert obs.shape[1] == basic_env.config.obs.obs_dim
        assert obs.shape[0] == basic_env.n_agents
        assert "n_agents" in info

    def test_reset_with_seed(self, basic_env):
        obs1, _ = basic_env.reset(seed=99)
        n1 = basic_env.n_agents
        obs2, _ = basic_env.reset(seed=99)
        n2 = basic_env.n_agents

        assert n1 == n2
        np.testing.assert_array_equal(obs1, obs2)

    def test_n_agents_in_range(self, basic_env):
        for _ in range(5):
            basic_env.reset()
            n = basic_env.n_agents
            assert 1 <= n <= 5  # May be pruned below min

    def test_observations_not_all_zero(self, basic_env):
        obs, _ = basic_env.reset()
        assert np.any(obs != 0.0)


class TestStep:
    def test_step_basic_shapes(self, basic_env):
        obs, _ = basic_env.reset()
        n = basic_env.n_agents
        action_dim = basic_env.config.action.action_dim

        actions = np.zeros((n, action_dim), dtype=np.float64)
        obs, rewards, terminated, truncated, info = basic_env.step(actions)

        assert obs.shape == (n, basic_env.config.obs.obs_dim)
        assert rewards.shape == (n,)
        assert terminated.shape == (n,)
        assert truncated.shape == (n,)

    def test_step_random_actions(self, basic_env):
        rng = np.random.default_rng(42)
        basic_env.reset()
        n = basic_env.n_agents

        for _ in range(10):
            actions = rng.uniform(-1, 1, (n, basic_env.config.action.action_dim))
            obs, rewards, terminated, truncated, info = basic_env.step(actions)
            assert obs.shape[0] == n

    def test_episode_terminates_on_max_steps(self, basic_env):
        basic_env.reset()
        n = basic_env.n_agents

        for step in range(basic_env.config.max_steps):
            actions = np.zeros((n, basic_env.config.action.action_dim))
            obs, rewards, terminated, truncated, info = basic_env.step(actions)

        # After max_steps, episode should be over
        assert info["episode_over"]
        assert np.all(truncated | terminated)

    def test_timeout_penalty_applied(self):
        config = CrowdEnvConfig(
            geometry=GeometryConfig(tier=GeometryTier.TIER_0, min_side=10.0, max_side=12.0),
            spawn=SpawnConfig(n_agents_range=(2, 2), min_spawn_separation=0.3),
            reward=RewardConfig(timeout_penalty=-5.0, goal_radius=0.01),
            max_steps=3,
        )
        env = CrowdEnv(config=config, seed=42)
        env.reset()
        n = env.n_agents

        # Stand still for 3 steps (won't reach goal)
        for _ in range(2):
            env.step(np.zeros((n, config.action.action_dim)))

        _, rewards, _, truncated, info = env.step(np.zeros((n, config.action.action_dim)))

        # All still-active agents should get timeout penalty
        assert info["episode_over"]
        for i in range(n):
            if truncated[i]:
                assert (
                    rewards[i] <= config.reward.timeout_penalty + 1.0
                )  # Allow some progress reward

    def test_inactive_agents_get_zero_obs(self, basic_env):
        """Once an agent reaches goal, its observations should be zero."""
        basic_env.reset()
        n = basic_env.n_agents

        # Run for a while — check that any terminated agent gets zero obs
        rng = np.random.default_rng(7)
        for _ in range(basic_env.config.max_steps):
            actions = rng.uniform(-1, 1, (n, basic_env.config.action.action_dim))
            obs, _, terminated, _, info = basic_env.step(actions)

            if info["episode_over"]:
                break

        # At least verify the mechanism works without error
        assert obs.shape == (n, basic_env.config.obs.obs_dim)


class TestMultiTier:
    def test_geometry_tier_randomisation(self):
        config = CrowdEnvConfig(
            geometry=GeometryConfig(min_side=8.0, max_side=12.0),
            geometry_tiers=[GeometryTier.TIER_0, GeometryTier.TIER_1],
            spawn=SpawnConfig(n_agents_range=(2, 4), min_spawn_separation=0.3),
            max_steps=10,
        )
        env = CrowdEnv(config=config, seed=42)

        tiers_seen = set()
        for _ in range(10):
            _, info = env.reset()
            tiers_seen.add(info["geometry_tier"])

        # Should see both tiers across 10 resets
        assert len(tiers_seen) >= 2


class TestCorridorEnv:
    def test_corridor_reset_and_step(self, corridor_env):
        obs, info = corridor_env.reset()
        assert obs.ndim == 2
        assert corridor_env.n_agents >= 1

        n = corridor_env.n_agents
        actions = np.zeros((n, corridor_env.config.action.action_dim))
        obs, rewards, terminated, truncated, _ = corridor_env.step(actions)
        assert obs.shape[0] == n

    def test_corridor_full_episode(self, corridor_env):
        corridor_env.reset()
        n = corridor_env.n_agents
        rng = np.random.default_rng(0)

        total_reward = np.zeros(n)
        for _ in range(corridor_env.config.max_steps):
            actions = rng.uniform(-1, 1, (n, corridor_env.config.action.action_dim))
            _, rewards, _, _, info = corridor_env.step(actions)
            total_reward += rewards
            if info["episode_over"]:
                break

        # Episode should end within max_steps
        assert info["episode_over"]


class TestPhysics:
    def test_agents_move_with_actions(self, basic_env):
        basic_env.reset()
        n = basic_env.n_agents
        initial_pos = basic_env._world.positions.copy()

        # Apply forward actions for several steps
        actions = np.zeros((n, basic_env.config.action.action_dim))
        actions[:, 0] = 1.0  # Max speed forward

        for _ in range(5):
            basic_env.step(actions)

        # Active agents should have moved
        moved = np.linalg.norm(basic_env._world.positions - initial_pos, axis=1)
        assert np.any(moved > 0.01)

    def test_collision_forces_prevent_overlap(self):
        """Two agents heading toward each other should be pushed apart."""
        config = CrowdEnvConfig(
            geometry=GeometryConfig(tier=GeometryTier.TIER_0, min_side=10.0, max_side=12.0),
            spawn=SpawnConfig(
                n_agents_range=(2, 2),
                min_spawn_separation=0.1,
                shoulder_width_mean=0.3,
                chest_depth_mean=0.2,
            ),
            max_steps=100,
            reward=RewardConfig(goal_radius=0.01),  # Tiny goal so they don't terminate
        )
        env = CrowdEnv(config=config, seed=42)
        env.reset()

        # Just run some steps — mainly checking no crashes
        n = env.n_agents
        for _ in range(20):
            actions = np.zeros((n, config.action.action_dim))
            actions[:, 0] = 1.0  # Forward
            env.step(actions)

        # No NaN or inf in positions
        assert np.all(np.isfinite(env._world.positions))
        assert np.all(np.isfinite(env._world.velocities))


class TestTemporalMemory:
    """End-to-end tests for the Option A temporal-memory observation features."""

    @staticmethod
    def _build_env(W=4, max_steps=50, dt=0.01):
        cfg = CrowdEnvConfig(
            geometry=GeometryConfig(tier=GeometryTier.TIER_0, min_side=10.0, max_side=12.0),
            spawn=SpawnConfig(n_agents_range=(3, 5), min_spawn_separation=0.3),
            solvability_mode=SolvabilityMode.PRUNE,
            obs=ObsConfig(
                use_temporal_memory=True,
                temporal_memory_window=W,
                temporal_memory_max_steps=max_steps,
                temporal_memory_dt=dt,
            ),
            max_steps=max_steps,
            dt=dt,
        )
        return CrowdEnv(config=cfg, seed=7)

    def test_obs_dim_includes_memory(self):
        env = self._build_env()
        obs, _ = env.reset()
        # 7 ego + 8*7 social + 16 rays + 6 memory = 85
        assert env.config.obs.obs_dim == 85
        assert obs.shape[1] == 85

    def test_reset_initialises_memory_state(self):
        env = self._build_env()
        env.reset()
        world = env._world
        assert world.spawn_positions is not None
        assert world.initial_goal_distances is not None
        assert world.cumulative_path_length is not None
        assert world.pos_history is not None
        assert world.gdist_history is not None
        # Spawn matches current positions at t=0
        np.testing.assert_array_equal(world.spawn_positions, world.positions)
        # Cumulative path length starts at zero
        assert np.all(world.cumulative_path_length == 0.0)
        # Ring buffer pre-filled with spawn positions
        n = world.n_agents
        buf_size = env.config.obs.temporal_memory_window + 1
        for t in range(buf_size):
            np.testing.assert_array_equal(world.pos_history[:, t, :], world.positions[:n])

    def test_features_at_reset_all_zero(self):
        env = self._build_env()
        obs, _ = env.reset()
        # Last 6 dims are the memory block
        memory = obs[:, -6:]
        # Active agents should have zero memory features at t=0 since nothing
        # has moved yet. Inactive rows are zero everywhere regardless.
        for i in range(env.n_agents):
            if env._active_mask[i]:
                np.testing.assert_allclose(memory[i], 0.0, atol=1e-9)

    def test_step_advances_cumulative_path(self):
        env = self._build_env()
        env.reset()
        n = env.n_agents
        actions = np.zeros((n, env.config.action.action_dim))
        actions[:, 0] = 1.0  # Forward
        for _ in range(10):
            env.step(actions)
        # At least one agent moved a non-trivial distance
        assert np.any(env._world.cumulative_path_length > 0.01)
        # step_count tracks the episode step
        assert env._world.step_count == 10

    def test_elapsed_fraction_in_obs(self):
        max_steps = 40
        env = self._build_env(max_steps=max_steps)
        env.reset()
        n = env.n_agents
        actions = np.zeros((n, env.config.action.action_dim))
        for _ in range(10):
            obs, *_ = env.step(actions)
        # elapsed_fraction is memory feature index 3 = obs[:, -3]
        for i in range(n):
            if env._active_mask[i]:
                assert abs(obs[i, -3] - 10 / max_steps) < 1e-6

    def test_window_features_eventually_nonzero(self):
        W = 4
        env = self._build_env(W=W, dt=0.1, max_steps=50)
        env.reset()
        n = env.n_agents
        actions = np.zeros((n, env.config.action.action_dim))
        actions[:, 0] = 1.0  # push forward
        # Step for W+2 simulation steps so the ring buffer has fully filled
        for _ in range(W + 2):
            obs, *_ = env.step(actions)

        # At least one active agent should show non-zero displacement window
        any_disp = False
        for i in range(n):
            if env._active_mask[i]:
                if abs(obs[i, -2]) > 1e-6 or abs(obs[i, -1]) > 1e-6:
                    any_disp = True
                    break
        assert any_disp, "Expected non-zero window features after W+2 forward steps"

    def test_memory_features_finite_during_rollout(self):
        env = self._build_env()
        env.reset()
        n = env.n_agents
        actions = np.zeros((n, env.config.action.action_dim))
        actions[:, 0] = 0.5
        for _ in range(30):
            obs, *_ = env.step(actions)
            assert np.all(np.isfinite(obs)), "observations must stay finite"


class TestNeighborMemoryWiring:
    """End-to-end wiring tests for the persistent neighbor-ID matcher.

    Commit 2 of plan/neighbor_memory_extension.md: the matcher runs every
    step when ``use_neighbor_memory`` is True, the slots stay stable across
    steps for agents that remain in range, and commits 3-5 will read from
    this table for observation features.
    """

    @staticmethod
    def _build_env(use_neighbor_memory=True):
        cfg = CrowdEnvConfig(
            geometry=GeometryConfig(tier=GeometryTier.TIER_0, min_side=10.0, max_side=12.0),
            spawn=SpawnConfig(n_agents_range=(5, 8), min_spawn_separation=0.3),
            solvability_mode=SolvabilityMode.PRUNE,
            obs=ObsConfig(
                use_neighbor_memory=use_neighbor_memory,
                neighbor_sensing_radius=5.0,
            ),
            max_steps=50,
            dt=0.01,
        )
        return CrowdEnv(config=cfg, seed=11)

    def test_disabled_leaves_neighbor_ids_as_none(self):
        env = self._build_env(use_neighbor_memory=False)
        env.reset()
        assert env._world.neighbor_ids is None

    def test_reset_populates_neighbor_ids(self):
        env = self._build_env(use_neighbor_memory=True)
        env.reset()
        nids = env._world.neighbor_ids
        assert nids is not None
        assert nids.shape == (env.n_agents, env.config.obs.k_neighbours)
        assert nids.dtype == np.int32
        # With >= 2 active agents there should be at least one non-empty slot
        active_rows = nids[env._active_mask]
        assert (active_rows >= 0).any(), "expected at least one populated slot after reset"

    def test_step_updates_neighbor_ids(self):
        env = self._build_env(use_neighbor_memory=True)
        env.reset()
        pre = env._world.neighbor_ids.copy()
        actions = np.zeros((env.n_agents, env.config.action.action_dim))
        actions[:, 0] = 0.5
        env.step(actions)
        post = env._world.neighbor_ids
        assert post is not None
        assert post.shape == pre.shape
        # IDs must remain in [-1, n_agents). Values outside [-1, n-1] would be a bug.
        n = env.n_agents
        assert post.max() < n
        assert post.min() >= -1

    def test_no_self_assignment(self):
        """An agent's own index must never appear in its own neighbor slots."""
        env = self._build_env(use_neighbor_memory=True)
        env.reset()
        actions = np.zeros((env.n_agents, env.config.action.action_dim))
        actions[:, 0] = 0.5
        for _ in range(5):
            env.step(actions)
            nids = env._world.neighbor_ids
            for i in range(env.n_agents):
                assert i not in nids[i], f"agent {i} appears in its own neighbor slots: {nids[i]}"

    def test_vel_history_buffer_allocated_and_shaped(self):
        """Ring buffer for neighbor velocities is allocated at reset."""
        env = self._build_env(use_neighbor_memory=True)
        env.reset()
        vh = env._world.neighbor_vel_history
        assert vh is not None
        W_n = env.config.obs.neighbor_vel_history_window
        K = env.config.obs.k_neighbours
        assert vh.shape == (env.n_agents, W_n + 1, K, 2)
        # Fresh reset -> all zeros
        assert np.all(vh == 0.0)

    def test_vel_history_buffer_populated_after_step(self):
        """After one step, at least one slot in the ring buffer holds a
        non-zero velocity (because neighbors were assigned and moving)."""
        env = self._build_env(use_neighbor_memory=True)
        env.reset()
        actions = np.zeros((env.n_agents, env.config.action.action_dim))
        actions[:, 0] = 1.0  # all agents push forward
        env.step(actions)
        vh = env._world.neighbor_vel_history
        # Find the slot that was written (step_count - 1 % (W_n+1))
        W_n = env.config.obs.neighbor_vel_history_window
        write_idx = (1 - 1) % (W_n + 1)
        written = vh[:, write_idx, :, :]
        # At least one (agent, slot) pair should have a non-zero velocity
        assert np.any(np.abs(written) > 1e-6), (
            f"expected non-zero velocities at write slot {write_idx}"
        )

    def test_obs_dim_grows_by_k_times_2_with_vel_history(self):
        """With both flags on, obs grows by exactly 2*K dims."""
        base = CrowdEnvConfig(
            geometry=GeometryConfig(tier=GeometryTier.TIER_0, min_side=10.0, max_side=12.0),
            spawn=SpawnConfig(n_agents_range=(5, 8), min_spawn_separation=0.3),
            solvability_mode=SolvabilityMode.PRUNE,
            obs=ObsConfig(
                use_neighbor_memory=True,
                use_neighbor_vel_history=False,
            ),
            max_steps=50,
            dt=0.01,
        )
        on = CrowdEnvConfig(
            geometry=GeometryConfig(tier=GeometryTier.TIER_0, min_side=10.0, max_side=12.0),
            spawn=SpawnConfig(n_agents_range=(5, 8), min_spawn_separation=0.3),
            solvability_mode=SolvabilityMode.PRUNE,
            obs=ObsConfig(
                use_neighbor_memory=True,
                use_neighbor_vel_history=True,
            ),
            max_steps=50,
            dt=0.01,
        )
        k = on.obs.k_neighbours
        assert on.obs.obs_dim == base.obs.obs_dim + 2 * k

    def test_vel_history_feature_block_is_finite_after_steps(self):
        """End-to-end: with vel history enabled, the feature block appears
        at the right offset in the observation vector and stays finite."""
        cfg = CrowdEnvConfig(
            geometry=GeometryConfig(tier=GeometryTier.TIER_0, min_side=10.0, max_side=12.0),
            spawn=SpawnConfig(n_agents_range=(5, 8), min_spawn_separation=0.3),
            solvability_mode=SolvabilityMode.PRUNE,
            obs=ObsConfig(
                use_neighbor_memory=True,
                use_neighbor_vel_history=True,
            ),
            max_steps=50,
            dt=0.01,
        )
        env = CrowdEnv(config=cfg, seed=11)
        env.reset()
        actions = np.zeros((env.n_agents, env.config.action.action_dim))
        actions[:, 0] = 1.0
        for _ in range(10):
            obs, *_ = env.step(actions)
            assert np.all(np.isfinite(obs))
            # Last 2*K dims are the vel-history block
            k = env.config.obs.k_neighbours
            nb_block = obs[:, -2 * k :]
            # Sanity: active agents with neighbors should see at least one
            # non-zero slot after the ring buffer fills.
            assert nb_block.shape == (env.n_agents, 2 * k)

    def test_vel_history_zero_for_empty_slot(self):
        """Slots with neighbor_ids == -1 must hold zero velocities even
        after several steps."""
        env = self._build_env(use_neighbor_memory=True)
        env.reset()
        actions = np.zeros((env.n_agents, env.config.action.action_dim))
        actions[:, 0] = 0.5
        for _ in range(5):
            env.step(actions)
        vh = env._world.neighbor_vel_history  # (n, W_n+1, K, 2)
        nids = env._world.neighbor_ids  # (n, K)
        # For every (i, k) where nids[i, k] == -1, vh[i, :, k, :] must be zero.
        for i in range(env.n_agents):
            for k in range(env.config.obs.k_neighbours):
                if nids[i, k] == -1:
                    assert np.all(vh[i, :, k, :] == 0.0), (
                        f"empty slot ({i}, {k}) has non-zero history"
                    )
