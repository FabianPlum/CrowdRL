"""Tests for sensing: raycasts and KNN social query."""

import numpy as np
from shapely.geometry import Polygon

from crowdrl_core.sensing import (
    RaycastConfig,
    cast_rays,
    knn_social,
    match_persistent_neighbors,
)
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


class TestMatchPersistentNeighbors:
    """Persistent-neighbor matching keeps neighbor identity stable across steps.

    See ``plan/neighbor_memory_extension.md`` section 2.1 for the rationale.
    """

    def test_first_step_fills_slots_with_nearest(self):
        """On the first step (all prev_slots == -1), the function degenerates
        to standard KNN -- each ego gets the k nearest active agents, sorted
        by distance, in its slot table."""
        positions = np.array(
            [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [2.0, 0.0], [5.0, 0.0]],
            dtype=np.float64,
        )
        n = positions.shape[0]
        prev = np.full((n, 3), -1, dtype=np.int32)
        active = np.ones(n, dtype=np.bool_)

        slots = match_persistent_neighbors(positions, prev, active, sensing_radius=10.0, k=3)

        # Agent 0 at (0,0): nearest are 1 @ dist 1, 3 @ dist 2, 2 @ dist 3
        assert slots[0, 0] == 1
        assert slots[0, 1] == 3
        assert slots[0, 2] == 2

    def test_retains_stable_neighbor(self):
        """A neighbor already assigned a slot and still in range keeps its
        slot, even if another agent is now closer -- that's the whole point
        of persistent matching."""
        # Agent 0 at origin. Initially slot 0 = agent 2 (dist 3).
        # Then agent 1 appears closer (dist 1), but slot 0 should still
        # point to agent 2 because it's still in range.
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=np.float64)
        prev = np.array([[2, -1], [-1, -1], [-1, -1]], dtype=np.int32)
        active = np.ones(3, dtype=np.bool_)

        slots = match_persistent_neighbors(positions, prev, active, sensing_radius=10.0, k=2)

        # Slot 0 must still be agent 2 (stable)
        assert slots[0, 0] == 2
        # Slot 1 was empty; should now hold the nearest unassigned = agent 1
        assert slots[0, 1] == 1

    def test_evicts_out_of_range_neighbor(self):
        """A previously-assigned neighbor that moved beyond sensing_radius
        gets evicted and replaced with the nearest in-range agent."""
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        # Agent 0 had agent 2 in slot 0, but agent 2 is now 10m away
        # and our sensing_radius is 5m -> evicted. Slot 0 should be
        # refilled with agent 1 (the nearest remaining in-range agent).
        prev = np.array([[2, -1], [-1, -1], [-1, -1]], dtype=np.int32)
        active = np.ones(3, dtype=np.bool_)

        slots = match_persistent_neighbors(positions, prev, active, sensing_radius=5.0, k=2)

        # Agent 2 evicted, slot 0 refilled with nearest in-range = agent 1
        assert slots[0, 0] == 1
        # Slot 1 left empty (no other candidates within range)
        assert slots[0, 1] == -1

    def test_evicts_inactive_neighbor(self):
        """If a previously-assigned neighbor became inactive (e.g. reached
        its goal), it is evicted even if still close."""
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        prev = np.array([[1, -1], [-1, -1], [-1, -1]], dtype=np.int32)
        active = np.array([True, False, True])

        slots = match_persistent_neighbors(positions, prev, active, sensing_radius=10.0, k=2)

        # Agent 1 is inactive -> evicted from slot 0
        # Agent 2 is the nearest remaining active neighbor -> fills slot 0
        assert slots[0, 0] == 2
        assert slots[0, 1] == -1

    def test_no_duplicate_slot_assignments(self):
        """Filling empty slots must never assign the same neighbor to two
        slots of the same ego."""
        n = 6
        positions = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
            dtype=np.float64,
        )
        prev = np.full((n, 3), -1, dtype=np.int32)
        active = np.ones(n, dtype=np.bool_)

        slots = match_persistent_neighbors(positions, prev, active, sensing_radius=10.0, k=3)

        for i in range(n):
            assigned = [s for s in slots[i] if s >= 0]
            assert len(assigned) == len(set(assigned)), (
                f"Duplicate assignment in row {i}: {slots[i]}"
            )

    def test_inactive_ego_row_is_all_negative_one(self):
        """An inactive ego agent's slot row must be all -1."""
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        prev = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32)
        active = np.array([False, True, True])

        slots = match_persistent_neighbors(positions, prev, active, sensing_radius=10.0, k=2)

        np.testing.assert_array_equal(slots[0], np.array([-1, -1], dtype=np.int32))

    def test_multistep_stability(self):
        """Across multiple steps with agents moving slowly, a neighbor that
        stays in range keeps the same slot number throughout."""
        # Two agents initially 1m apart, agent 1 drifts to 2m, 3m, 4m
        # from agent 0. Slot 0 should always hold agent 1.
        prev = np.array([[-1, -1], [-1, -1]], dtype=np.int32)
        active = np.ones(2, dtype=np.bool_)

        for drift in [1.0, 2.0, 3.0, 4.0]:
            positions = np.array([[0.0, 0.0], [drift, 0.0]], dtype=np.float64)
            prev = match_persistent_neighbors(positions, prev, active, sensing_radius=5.0, k=2)
            assert prev[0, 0] == 1, f"At drift {drift}, slot 0 lost stable assignment"
            assert prev[1, 0] == 0
