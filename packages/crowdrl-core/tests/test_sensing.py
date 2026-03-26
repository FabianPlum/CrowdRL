"""Tests for sensing: raycasts and KNN social query."""

import numpy as np
from shapely.geometry import Polygon

from crowdrl_core.sensing import RaycastConfig, cast_rays, knn_social
from conftest import make_world_state


class TestCastRays:
    def test_all_clear_in_open_space(self):
        """Agent in centre of large polygon, all rays should read ~1.0."""
        world = make_world_state(
            n_agents=1,
            polygon=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
            positions=np.array([[50.0, 50.0]]),
            goal_positions=np.array([[90.0, 50.0]]),
        )
        config = RaycastConfig(n_rays=16, fov_deg=200, max_range=5.0)
        readings = cast_rays(world, 0, config)
        assert readings.shape == (16,)
        # All should be 1.0 (no walls within 5m in a 100m square)
        np.testing.assert_allclose(readings, 1.0)

    def test_wall_detection(self):
        """Agent near a wall should detect it."""
        world = make_world_state(
            n_agents=1,
            polygon=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            positions=np.array([[1.0, 5.0]]),
            torso_orientations=np.array([np.pi]),  # Facing -x (toward wall)
            head_orientations=np.array([np.pi]),
            goal_positions=np.array([[9.0, 5.0]]),
        )
        config = RaycastConfig(n_rays=1, fov_deg=0, max_range=5.0)
        readings = cast_rays(world, 0, config)
        # Should detect wall at ~1m distance
        assert readings[0] < 0.5  # 1m / 5m = 0.2

    def test_agent_detection(self):
        """Ray should detect another agent."""
        world = make_world_state(
            n_agents=2,
            positions=np.array([[1.0, 5.0], [3.0, 5.0]]),
            torso_orientations=np.array([0.0, 0.0]),
            head_orientations=np.array([0.0, 0.0]),
        )
        config = RaycastConfig(n_rays=1, fov_deg=0, max_range=5.0)
        readings = cast_rays(world, 0, config)
        # Agent 1 is ~2m away, should be detected
        assert readings[0] < 1.0

    def test_two_channel_output(self):
        """Two-channel raycasts should return (distance, hit_type) pairs."""
        world = make_world_state(
            n_agents=1,
            positions=np.array([[1.0, 5.0]]),
            torso_orientations=np.array([np.pi]),
            head_orientations=np.array([np.pi]),
            goal_positions=np.array([[9.0, 5.0]]),
        )
        config = RaycastConfig(n_rays=4, fov_deg=90, max_range=5.0, two_channel=True)
        readings = cast_rays(world, 0, config)
        assert readings.shape == (4, 2)

    def test_inactive_agents_ignored(self):
        """Inactive agents should not be detected by rays."""
        world = make_world_state(
            n_agents=2,
            polygon=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
            positions=np.array([[50.0, 50.0], [52.0, 50.0]]),
            head_orientations=np.array([0.0, 0.0]),
            goal_positions=np.array([[90.0, 50.0], [90.0, 50.0]]),
        )
        world.active_mask = np.array([True, False])
        config = RaycastConfig(n_rays=1, fov_deg=0, max_range=5.0)
        readings = cast_rays(world, 0, config)
        assert readings[0] == 1.0  # Agent 1 is inactive, not detected

    def test_fov_restricts_rays(self):
        """Rays should only cover the configured FOV."""
        world = make_world_state(
            n_agents=1,
            polygon=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            positions=np.array([[5.0, 5.0]]),
            head_orientations=np.array([0.0]),  # Facing +x
            goal_positions=np.array([[9.0, 5.0]]),
        )
        config = RaycastConfig(n_rays=16, fov_deg=200, max_range=10.0)
        readings = cast_rays(world, 0, config)
        # Front rays (facing +x, 5m to wall) should be shorter than back rays
        # The wall at x=10 is 5m away
        assert readings.shape == (16,)


class TestKNNSocial:
    def test_single_agent_empty(self):
        """With only one agent, social features should be all zeros."""
        world = make_world_state(n_agents=1)
        social = knn_social(world, 0, k=8)
        assert social.shape == (8, 7)
        np.testing.assert_array_equal(social, np.zeros((8, 7)))

    def test_two_agents(self):
        """Two agents — one neighbour should be populated."""
        world = make_world_state(
            n_agents=2,
            positions=np.array([[0.0, 0.0], [3.0, 0.0]]),
            velocities=np.array([[1.0, 0.0], [-1.0, 0.0]]),
        )
        social = knn_social(world, 0, k=8)
        assert social.shape == (8, 7)
        # First neighbour should be populated
        assert np.linalg.norm(social[0, :2]) > 0  # Non-zero relative position
        # Remaining 7 should be zero-padded
        np.testing.assert_array_equal(social[1:], np.zeros((7, 7)))

    def test_egocentric_frame(self):
        """Relative position should be in ego frame."""
        # Agent 0 at origin facing +x, agent 1 directly ahead
        world = make_world_state(
            n_agents=2,
            positions=np.array([[0.0, 0.0], [3.0, 0.0]]),
            torso_orientations=np.array([0.0, 0.0]),  # Facing +x
        )
        social = knn_social(world, 0, k=8)
        # In ego frame (facing +x), agent 1 should be at ~(3, 0)
        assert social[0, 0] > 0  # Positive x (ahead)
        assert abs(social[0, 1]) < 1e-6  # Zero y

    def test_egocentric_rotated(self):
        """Agent facing +y — neighbour at +x should appear to the right."""
        world = make_world_state(
            n_agents=2,
            positions=np.array([[0.0, 0.0], [3.0, 0.0]]),
            torso_orientations=np.array([np.pi / 2, 0.0]),  # Agent 0 facing +y
        )
        social = knn_social(world, 0, k=8)
        # In ego frame (facing +y), agent 1 at global +x should be at ego (x>0 means right of heading)
        # With rotation by -pi/2: global [3,0] → ego [0, -3]
        assert abs(social[0, 0]) < 1e-6  # ~0 in ego x
        assert social[0, 1] < 0  # Negative ego y (to the right)

    def test_k_limiting(self):
        """With more agents than K, only K nearest should be reported."""
        n = 20
        positions = np.random.RandomState(42).randn(n, 2) * 3
        positions[0] = [0, 0]  # Ego at origin
        world = make_world_state(
            n_agents=n,
            positions=positions,
            goal_positions=np.zeros((n, 2)),
        )
        social = knn_social(world, 0, k=4)
        assert social.shape == (4, 7)
        # All 4 should be populated (non-zero)
        for i in range(4):
            assert np.linalg.norm(social[i, :2]) > 0

    def test_body_dimensions_included(self):
        """Social features should include neighbour body dimensions."""
        world = make_world_state(
            n_agents=2,
            positions=np.array([[0.0, 0.0], [3.0, 0.0]]),
            shoulder_widths=np.array([0.23, 0.30]),
            chest_depths=np.array([0.15, 0.20]),
        )
        social = knn_social(world, 0, k=8)
        assert abs(social[0, 5] - 0.30) < 1e-6  # shoulder_width
        assert abs(social[0, 6] - 0.20) < 1e-6  # chest_depth

    def test_inactive_agents_excluded(self):
        """Inactive agents should not appear in social sensing."""
        world = make_world_state(
            n_agents=3,
            positions=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            goal_positions=np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]),
        )
        world.active_mask = np.array([True, False, True])
        social = knn_social(world, 0, k=8)
        # Only agent 2 should appear (agent 1 is inactive)
        # Agent 2 is at distance 2 in +x direction
        assert social[0, 0] > 0
        assert abs(social[0, 0] - 2.0) < 1e-6
        # Second slot should be empty
        np.testing.assert_array_equal(social[1], np.zeros(7))
