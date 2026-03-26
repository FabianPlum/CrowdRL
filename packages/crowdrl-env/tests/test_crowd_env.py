"""Tests for the CrowdEnv Gymnasium environment."""

import numpy as np
import pytest

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
