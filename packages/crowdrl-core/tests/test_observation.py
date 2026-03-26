"""Tests for the observation builder — the critical transfer guarantee."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from crowdrl_core.observation import ObsConfig, build_observation, build_observations_batch
from crowdrl_core.sensing import RaycastConfig
from conftest import make_world_state


class TestObsConfig:
    def test_default_dims(self):
        config = ObsConfig()
        # 7 ego + 56 social (8×7) + 16 rays = 79
        assert config.obs_dim == 79

    def test_two_channel_dims(self):
        config = ObsConfig(raycast=RaycastConfig(two_channel=True))
        # 7 + 56 + 32 = 95
        assert config.obs_dim == 95

    def test_navmesh_dims(self):
        config = ObsConfig(use_navmesh=True)
        # 7 + 56 + 16 + 3 = 82
        assert config.obs_dim == 82

    def test_full_dims(self):
        config = ObsConfig(
            raycast=RaycastConfig(two_channel=True),
            use_navmesh=True,
        )
        # 7 + 56 + 32 + 3 = 98
        assert config.obs_dim == 98

    def test_custom_k(self):
        config = ObsConfig(k_neighbours=4)
        # 7 + 28 (4×7) + 16 = 51
        assert config.obs_dim == 51


class TestBuildObservation:
    def test_output_shape(self):
        config = ObsConfig()
        world = make_world_state(n_agents=2)
        obs = build_observation(world, 0, config)
        assert obs.shape == (config.obs_dim,)
        assert obs.dtype == np.float64

    def test_deterministic(self):
        """Same world state should produce identical observations."""
        config = ObsConfig()
        world = make_world_state(n_agents=2)
        obs1 = build_observation(world, 0, config)
        obs2 = build_observation(world, 0, config)
        np.testing.assert_array_equal(obs1, obs2)

    def test_goal_direction_egocentric(self):
        """Goal direction should be in ego frame."""
        # Agent at origin facing +x, goal directly ahead
        world = make_world_state(
            n_agents=1,
            positions=np.array([[0.0, 0.0]]),
            torso_orientations=np.array([0.0]),
            goal_positions=np.array([[5.0, 0.0]]),
        )
        config = ObsConfig(k_neighbours=0, raycast=RaycastConfig(n_rays=1))
        obs = build_observation(world, 0, config)
        # First two elements are goal direction in ego frame
        goal_dir = obs[:2]
        assert goal_dir[0] > 0.9  # Pointing forward
        assert abs(goal_dir[1]) < 0.1

    def test_goal_direction_rotated(self):
        """Goal to the right of a north-facing agent."""
        world = make_world_state(
            n_agents=1,
            positions=np.array([[0.0, 0.0]]),
            torso_orientations=np.array([np.pi / 2]),  # Facing +y
            head_orientations=np.array([np.pi / 2]),
            goal_positions=np.array([[5.0, 0.0]]),  # Goal is to the right
        )
        config = ObsConfig(k_neighbours=0, raycast=RaycastConfig(n_rays=1))
        obs = build_observation(world, 0, config)
        goal_dir = obs[:2]
        # In ego frame (facing +y), goal at +x global = -y ego (to the right)
        assert goal_dir[1] < -0.5

    def test_velocity_in_ego_frame(self):
        """Velocity should be rotated to ego frame."""
        world = make_world_state(
            n_agents=1,
            positions=np.array([[0.0, 0.0]]),
            velocities=np.array([[1.0, 0.0]]),  # Moving +x
            torso_orientations=np.array([0.0]),  # Facing +x
            goal_positions=np.array([[5.0, 0.0]]),
        )
        config = ObsConfig(k_neighbours=0, raycast=RaycastConfig(n_rays=1))
        obs = build_observation(world, 0, config)
        # Velocity is obs[2:4]
        vel = obs[2:4]
        assert vel[0] > 0.9  # Moving forward in ego frame
        assert abs(vel[1]) < 0.1

    def test_head_rel_torso_in_obs(self):
        """Head angle relative to torso should be in the observation."""
        world = make_world_state(
            n_agents=1,
            positions=np.array([[5.0, 5.0]]),
            torso_orientations=np.array([0.0]),
            head_orientations=np.array([0.5]),  # Head turned 0.5 rad
            goal_positions=np.array([[9.0, 5.0]]),
        )
        config = ObsConfig(k_neighbours=0, raycast=RaycastConfig(n_rays=1))
        obs = build_observation(world, 0, config)
        # head_rel_torso is the last element of ego state (index 6)
        assert abs(obs[6] - 0.5) < 1e-6

    def test_two_channel_rays(self):
        config = ObsConfig(raycast=RaycastConfig(n_rays=8, two_channel=True))
        world = make_world_state(n_agents=1)
        obs = build_observation(world, 0, config)
        assert obs.shape == (config.obs_dim,)

    def test_with_navmesh(self):
        config = ObsConfig(use_navmesh=True)
        world = make_world_state(n_agents=1, build_nav=True)
        obs = build_observation(world, 0, config)
        assert obs.shape == (config.obs_dim,)
        # Last 3 elements should be navmesh signals
        nav_signal = obs[-3:]
        # Waypoint direction should be unit or zero
        nav_dir_norm = np.linalg.norm(nav_signal[:2])
        assert nav_dir_norm <= 1.0 + 1e-6

    def test_different_agents_different_obs(self):
        """Two agents at different positions should have different observations."""
        config = ObsConfig()
        world = make_world_state(
            n_agents=2,
            positions=np.array([[2.0, 5.0], [8.0, 5.0]]),
        )
        obs0 = build_observation(world, 0, config)
        obs1 = build_observation(world, 1, config)
        assert not np.allclose(obs0, obs1)


class TestBuildObservationsBatch:
    def test_batch_shape(self):
        config = ObsConfig()
        world = make_world_state(n_agents=5)
        # Need proper 5-agent world state
        n = 5
        world = make_world_state(
            n_agents=n,
            positions=np.random.randn(n, 2) * 3 + 5,
            goal_positions=np.random.randn(n, 2) * 3 + 5,
        )
        obs = build_observations_batch(world, config)
        assert obs.shape == (5, config.obs_dim)

    def test_batch_matches_individual(self):
        config = ObsConfig()
        n = 3
        world = make_world_state(
            n_agents=n,
            positions=np.array([[2.0, 5.0], [5.0, 5.0], [8.0, 5.0]]),
            goal_positions=np.array([[8.0, 5.0], [2.0, 5.0], [5.0, 5.0]]),
        )
        batch = build_observations_batch(world, config)
        for i in range(n):
            individual = build_observation(world, i, config)
            np.testing.assert_array_equal(batch[i], individual)

    def test_active_mask(self):
        config = ObsConfig()
        world = make_world_state(
            n_agents=3,
            positions=np.array([[2.0, 5.0], [5.0, 5.0], [8.0, 5.0]]),
            goal_positions=np.array([[8.0, 5.0], [2.0, 5.0], [5.0, 5.0]]),
        )
        world.active_mask = np.array([True, False, True])
        batch = build_observations_batch(world, config)
        assert batch.shape == (2, config.obs_dim)  # Only 2 active

    def test_empty_world(self):
        config = ObsConfig()
        world = make_world_state(n_agents=2)
        world.active_mask = np.array([False, False])
        batch = build_observations_batch(world, config)
        assert batch.shape == (0, config.obs_dim)
