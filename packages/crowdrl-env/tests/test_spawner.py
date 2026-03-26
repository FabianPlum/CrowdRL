"""Tests for the crowd composition sampler and agent spawner."""

import numpy as np
import pytest
from shapely.geometry import box

from crowdrl_env.spawner import SpawnConfig, spawn_agents


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_regions():
    """Simple spawn and goal regions for testing."""
    spawn = [box(0, 0, 5, 5)]
    goal = [box(15, 0, 20, 5)]
    return spawn, goal


class TestSpawnAgents:
    def test_basic_spawn(self, rng, simple_regions):
        spawn_r, goal_r = simple_regions
        result = spawn_agents(rng, spawn_r, goal_r, n_agents=10)

        assert result.n_agents == 10
        assert result.positions.shape == (10, 2)
        assert result.velocities.shape == (10, 2)
        assert result.torso_orientations.shape == (10,)
        assert result.head_orientations.shape == (10,)
        assert result.shoulder_widths.shape == (10,)
        assert result.chest_depths.shape == (10,)
        assert result.goal_positions.shape == (10, 2)
        assert result.preferred_speeds.shape == (10,)

    def test_initial_velocity_zero(self, rng, simple_regions):
        spawn_r, goal_r = simple_regions
        result = spawn_agents(rng, spawn_r, goal_r, n_agents=5)
        np.testing.assert_array_equal(result.velocities, 0.0)

    def test_orientations_face_goal(self, rng, simple_regions):
        spawn_r, goal_r = simple_regions
        result = spawn_agents(rng, spawn_r, goal_r, n_agents=5)

        for i in range(result.n_agents):
            diff = result.goal_positions[i] - result.positions[i]
            expected_angle = np.arctan2(diff[1], diff[0])
            np.testing.assert_almost_equal(
                result.torso_orientations[i], expected_angle, decimal=10
            )
            np.testing.assert_almost_equal(result.head_orientations[i], expected_angle, decimal=10)

    def test_body_dimensions_positive(self, rng, simple_regions):
        spawn_r, goal_r = simple_regions
        config = SpawnConfig(min_body_dim=0.05)
        result = spawn_agents(rng, spawn_r, goal_r, config=config, n_agents=20)

        assert np.all(result.shoulder_widths >= config.min_body_dim)
        assert np.all(result.chest_depths >= config.min_body_dim)

    def test_preferred_speeds_in_range(self, rng, simple_regions):
        spawn_r, goal_r = simple_regions
        config = SpawnConfig(min_speed=0.5, max_speed=2.0)
        result = spawn_agents(rng, spawn_r, goal_r, config=config, n_agents=20)

        assert np.all(result.preferred_speeds >= config.min_speed)
        assert np.all(result.preferred_speeds <= config.max_speed)

    def test_spawn_positions_in_region(self, rng, simple_regions):
        spawn_r, goal_r = simple_regions
        result = spawn_agents(rng, spawn_r, goal_r, n_agents=10)

        from shapely.geometry import Point

        for i in range(result.n_agents):
            p = Point(result.positions[i])
            assert any(region.contains(p) for region in spawn_r)

    def test_goal_positions_in_region(self, rng, simple_regions):
        spawn_r, goal_r = simple_regions
        result = spawn_agents(rng, spawn_r, goal_r, n_agents=10)

        from shapely.geometry import Point

        for i in range(result.n_agents):
            p = Point(result.goal_positions[i])
            assert any(region.contains(p) for region in goal_r)

    def test_minimum_separation(self, rng, simple_regions):
        spawn_r, goal_r = simple_regions
        config = SpawnConfig(min_spawn_separation=0.8)
        result = spawn_agents(rng, spawn_r, goal_r, config=config, n_agents=10)

        for i in range(result.n_agents):
            for j in range(i + 1, result.n_agents):
                dist = np.linalg.norm(result.positions[i] - result.positions[j])
                assert dist >= config.min_spawn_separation - 1e-10

    def test_tight_region_fewer_agents(self, rng):
        """When spawn region is tiny, fewer agents than requested may be placed."""
        tiny_spawn = [box(0, 0, 0.5, 0.5)]
        goal = [box(10, 0, 15, 5)]
        config = SpawnConfig(min_spawn_separation=0.3)
        result = spawn_agents(rng, tiny_spawn, goal, config=config, n_agents=100)

        # Should place some but not all 100
        assert result.n_agents > 0
        assert result.n_agents < 100

    def test_random_agent_count_from_range(self, rng, simple_regions):
        spawn_r, goal_r = simple_regions
        config = SpawnConfig(n_agents_range=(5, 15))
        result = spawn_agents(rng, spawn_r, goal_r, config=config)

        assert 5 <= result.n_agents <= 15

    def test_multiple_spawn_regions(self, rng):
        spawns = [box(0, 0, 3, 3), box(7, 0, 10, 3)]
        goals = [box(15, 0, 20, 5)]
        result = spawn_agents(rng, spawns, goals, n_agents=10)

        assert result.n_agents > 0

    def test_multiple_goal_regions(self, rng):
        spawns = [box(0, 0, 5, 5)]
        goals = [box(15, 0, 20, 3), box(15, 3, 20, 6)]
        result = spawn_agents(rng, spawns, goals, n_agents=10)

        assert result.n_agents == 10

    def test_reproducibility(self, simple_regions):
        spawn_r, goal_r = simple_regions
        r1 = spawn_agents(np.random.default_rng(123), spawn_r, goal_r, n_agents=5)
        r2 = spawn_agents(np.random.default_rng(123), spawn_r, goal_r, n_agents=5)

        np.testing.assert_array_equal(r1.positions, r2.positions)
        np.testing.assert_array_equal(r1.goal_positions, r2.goal_positions)
