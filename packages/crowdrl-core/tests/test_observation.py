"""Tests for the observation builder — the critical transfer guarantee."""

import numpy as np

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

    def test_temporal_memory_dims(self):
        config = ObsConfig(use_temporal_memory=True)
        # 7 + 56 + 16 + 6 = 85
        assert config.obs_dim == 85

    def test_temporal_memory_with_navmesh_dims(self):
        config = ObsConfig(use_navmesh=True, use_temporal_memory=True)
        # 7 + 56 + 16 + 3 + 6 = 88
        assert config.obs_dim == 88

    def test_neighbor_vel_history_dims(self):
        config = ObsConfig(
            use_navmesh=True,
            use_temporal_memory=True,
            use_neighbor_memory=True,
            use_neighbor_vel_history=True,
        )
        # 88 + 8*2 = 104
        assert config.obs_dim == 104

    def test_neighbor_vel_history_requires_memory_flag(self):
        """Without ``use_neighbor_memory``, the vel history flag alone
        should NOT add dimensions -- the matcher isn't running so the
        buffer can't be read."""
        config = ObsConfig(
            use_neighbor_memory=False,
            use_neighbor_vel_history=True,
        )
        # Still the base 79 (no navmesh, no temporal memory, no neighbor mem)
        assert config.obs_dim == 79


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
            np.testing.assert_allclose(batch[i], individual, atol=1e-12)

    def test_active_mask(self):
        config = ObsConfig()
        world = make_world_state(
            n_agents=3,
            positions=np.array([[2.0, 5.0], [5.0, 5.0], [8.0, 5.0]]),
            goal_positions=np.array([[8.0, 5.0], [2.0, 5.0], [5.0, 5.0]]),
        )
        world.active_mask = np.array([True, False, True])
        batch = build_observations_batch(world, config)
        # Returns all agents; inactive agents have zero obs
        assert batch.shape == (3, config.obs_dim)
        np.testing.assert_array_equal(batch[1], np.zeros(config.obs_dim))
        assert np.any(batch[0] != 0)
        assert np.any(batch[2] != 0)

    def test_empty_world(self):
        config = ObsConfig()
        world = make_world_state(n_agents=2)
        world.active_mask = np.array([False, False])
        batch = build_observations_batch(world, config)
        assert batch.shape == (2, config.obs_dim)
        np.testing.assert_array_equal(batch, np.zeros((2, config.obs_dim)))


def _attach_memory_state(world, W=50, preferred_speeds=None):
    """Helper: initialise the temporal-memory state on a WorldState at t=0.

    Ring buffers are pre-filled with the current position / goal distance
    so reads before the window fills return the spawn value.
    """
    n = world.n_agents
    buf = W + 1
    goal_d = np.linalg.norm(world.goal_positions - world.positions, axis=1)
    world.spawn_positions = world.positions.copy()
    world.initial_goal_distances = goal_d.copy()
    world.cumulative_path_length = np.zeros(n, dtype=np.float64)
    world.pos_history = np.broadcast_to(world.positions[:, np.newaxis, :], (n, buf, 2)).copy()
    world.gdist_history = np.broadcast_to(goal_d[:, np.newaxis], (n, buf)).copy()
    world.preferred_speeds = (
        preferred_speeds if preferred_speeds is not None else np.full(n, 1.3, dtype=np.float64)
    )
    world.step_count = 0
    return world


class TestTemporalMemory:
    """Tests for the 6 temporal-memory observation features (Option A)."""

    def _make_config(self, W=4, max_steps=100, dt=0.01):
        return ObsConfig(
            k_neighbours=0,
            raycast=RaycastConfig(n_rays=1),
            use_temporal_memory=True,
            temporal_memory_window=W,
            temporal_memory_max_steps=max_steps,
            temporal_memory_dt=dt,
        )

    def test_memory_features_zero_at_spawn(self):
        """At t=0 (just spawned), all temporal features should be 0 or sensible defaults."""
        config = self._make_config()
        world = make_world_state(
            n_agents=1,
            positions=np.array([[2.0, 5.0]]),
            goal_positions=np.array([[8.0, 5.0]]),
        )
        _attach_memory_state(world, W=config.temporal_memory_window)

        obs = build_observation(world, 0, config)
        memory = obs[-6:]
        disp_spawn, cum_path, path_eff, elapsed, disp_win, goal_win = memory

        # Spawn == position, so displacement-from-spawn = 0
        assert disp_spawn == 0.0
        # Haven't walked any path yet
        assert cum_path == 0.0
        # path_eff is 0/eps -> 0 at spawn
        assert path_eff == 0.0
        # step_count = 0
        assert elapsed == 0.0
        # Ring buffer slots all = spawn_pos, displacement over window = 0
        assert disp_win == 0.0
        # gdist_window = initial_gdist, so goal progress = 0
        assert goal_win == 0.0

    def test_displacement_from_spawn(self):
        """Moving away from spawn increases displacement_from_spawn."""
        config = self._make_config()
        world = make_world_state(
            n_agents=1,
            positions=np.array([[2.0, 5.0]]),
            goal_positions=np.array([[8.0, 5.0]]),
        )
        _attach_memory_state(world, W=config.temporal_memory_window)

        # Initial goal distance = 6.0. Move agent 3m forward (halfway).
        world.positions[0] = [5.0, 5.0]
        obs = build_observation(world, 0, config)
        disp_spawn = obs[-6]
        # 3.0 / 6.0 = 0.5
        assert abs(disp_spawn - 0.5) < 1e-9

    def test_path_efficiency_straight_line(self):
        """Straight-line motion should give path_efficiency near 1.0."""
        config = self._make_config()
        world = make_world_state(
            n_agents=1,
            positions=np.array([[0.0, 0.0]]),
            goal_positions=np.array([[10.0, 0.0]]),
        )
        _attach_memory_state(world, W=config.temporal_memory_window)

        # Simulate moving in a straight line: cum_path = displacement
        world.positions[0] = [5.0, 0.0]
        world.cumulative_path_length[0] = 5.0
        obs = build_observation(world, 0, config)
        path_eff = obs[-4]
        assert abs(path_eff - 1.0) < 1e-9

    def test_path_efficiency_looping(self):
        """Looping motion should give path_efficiency much less than 1.0."""
        config = self._make_config()
        world = make_world_state(
            n_agents=1,
            positions=np.array([[0.0, 0.0]]),
            goal_positions=np.array([[10.0, 0.0]]),
        )
        _attach_memory_state(world, W=config.temporal_memory_window)

        # Walked 10m but ended up 2m from spawn: efficiency = 0.2
        world.positions[0] = [2.0, 0.0]
        world.cumulative_path_length[0] = 10.0
        obs = build_observation(world, 0, config)
        path_eff = obs[-4]
        assert abs(path_eff - 0.2) < 1e-9

    def test_elapsed_fraction(self):
        """elapsed_fraction = step_count / max_steps."""
        config = self._make_config(W=4, max_steps=100)
        world = make_world_state(
            n_agents=1,
            positions=np.array([[0.0, 0.0]]),
            goal_positions=np.array([[10.0, 0.0]]),
        )
        _attach_memory_state(world, W=config.temporal_memory_window)
        world.step_count = 25
        obs = build_observation(world, 0, config)
        elapsed = obs[-3]
        assert abs(elapsed - 0.25) < 1e-9

    def test_disp_window_after_movement(self):
        """Manually step the ring buffer: displacement over window reflects motion."""
        W = 4
        config = self._make_config(W=W, max_steps=100, dt=0.1)
        # Use dt=0.1 to make ring-buffer math easy to reason about: v_pref=1
        # gives expected_window = 1 * 4 * 0.1 = 0.4 m over the W-step window.
        world = make_world_state(
            n_agents=1,
            positions=np.array([[0.0, 0.0]]),
            goal_positions=np.array([[10.0, 0.0]]),
        )
        _attach_memory_state(
            world,
            W=W,
            preferred_speeds=np.array([1.0], dtype=np.float64),
        )

        # Simulate 5 steps of forward motion, delta = 0.1m per step
        buf_size = W + 1  # 5
        positions_seq = [0.1, 0.2, 0.3, 0.4, 0.5]
        for step_idx, x in enumerate(positions_seq):
            world.positions[0] = [x, 0.0]
            world.cumulative_path_length[0] = x
            # Write current pos into ring buffer at pre-step cursor
            write_idx = step_idx % buf_size
            world.pos_history[0, write_idx] = world.positions[0]
            world.gdist_history[0, write_idx] = 10.0 - x
            world.step_count = step_idx + 1

        obs = build_observation(world, 0, config)
        disp_win = obs[-2]
        goal_win = obs[-1]

        # After 5 writes, the window reads at index step_count % buf_size = 5%5 = 0
        # Index 0 contains pos[0] = 0.1 (the oldest entry still in buffer).
        # So window displacement = |0.5 - 0.1| = 0.4.
        # Expected window = v_pref * W * dt = 1 * 4 * 0.1 = 0.4.
        # Normalised = 0.4 / 0.4 = 1.0.
        assert abs(disp_win - 1.0) < 1e-9
        # Goal progress: gdist[0] was (10 - 0.1) = 9.9 at step 0. Now 9.5.
        # progress = 9.9 - 9.5 = 0.4, normalised = 1.0
        assert abs(goal_win - 1.0) < 1e-9

    def test_memory_features_batched_matches_per_agent(self):
        """build_observations_batch should produce the same memory features as build_observation."""
        W = 4
        config = self._make_config(W=W, max_steps=100)
        n = 3
        world = make_world_state(
            n_agents=n,
            positions=np.array([[0.0, 0.0], [5.0, 5.0], [2.0, 8.0]]),
            goal_positions=np.array([[10.0, 0.0], [5.0, 0.0], [2.0, 0.0]]),
        )
        _attach_memory_state(world, W=W)

        # Simulate some motion and state manipulation
        world.step_count = 10
        world.positions[0] = [3.0, 0.0]
        world.cumulative_path_length[0] = 3.5
        world.pos_history[0, 10 % (W + 1)] = [3.0, 0.0]
        world.gdist_history[0, 10 % (W + 1)] = 7.0

        batch = build_observations_batch(world, config)
        for i in range(n):
            individual = build_observation(world, i, config)
            np.testing.assert_allclose(batch[i], individual, atol=1e-12)
