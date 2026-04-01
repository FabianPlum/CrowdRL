"""Tests for the solvability verifier."""

import numpy as np
import pytest
from shapely.geometry import box

from crowdrl_core.geometry import build_navmesh
from crowdrl_core.navmesh import is_passable, is_reachable
from crowdrl_env.solvability import (
    SolvabilityMode,
    filter_by_solvability,
    verify_solvability,
)


@pytest.fixture
def simple_navmesh():
    """Navmesh for a 10x5 rectangle -- all portals are wide."""
    polygon = box(0, 0, 10, 5)
    return build_navmesh(polygon)


@pytest.fixture
def bottleneck_navmesh():
    """Navmesh for a corridor with a narrow bottleneck (~1m gap)."""
    exterior = box(0, 0, 20, 6)
    obstacle_top = box(9, 3.5, 11, 6)
    obstacle_bottom = box(9, 0, 11, 2.5)
    polygon = exterior.difference(obstacle_top).difference(obstacle_bottom)
    return build_navmesh(polygon)


@pytest.fixture
def close_obstacles_navmesh():
    """Navmesh with two close but non-touching obstacles creating a narrow gap.

    The gap between the two obstacles is 0.4m -- too narrow for most agents
    but the diagonal portal edge may be longer, fooling a portal-only check.
    """
    exterior = box(0, 0, 10, 6)
    obs_a = box(4, 0.5, 5, 2.8)
    obs_b = box(4, 3.2, 5, 5.5)
    polygon = exterior.difference(obs_a).difference(obs_b)
    return build_navmesh(polygon)


# ---- Core is_passable tests ----


class TestIsPassable:
    def test_open_rectangle_any_radius(self, simple_navmesh):
        """Wide open rectangle -- even large agents pass."""
        start = np.array([1.0, 2.5])
        goal = np.array([9.0, 2.5])
        assert is_passable(simple_navmesh, start, goal, agent_radius=0.3)

    def test_zero_radius_equals_reachable(self, simple_navmesh):
        start = np.array([1.0, 2.5])
        goal = np.array([9.0, 2.5])
        assert is_passable(simple_navmesh, start, goal, agent_radius=0.0)
        assert is_reachable(simple_navmesh, start, goal)

    def test_bottleneck_small_agent_passes(self, bottleneck_navmesh):
        """Agent smaller than the ~1m gap fits through."""
        start = np.array([2.0, 3.0])
        goal = np.array([18.0, 3.0])
        assert is_passable(bottleneck_navmesh, start, goal, agent_radius=0.2)

    def test_bottleneck_large_agent_rejected(self, bottleneck_navmesh):
        """Agent wider than the ~1m gap cannot pass."""
        start = np.array([2.0, 3.0])
        goal = np.array([18.0, 3.0])
        # Gap is ~1m (from y=2.5 to y=3.5), so agent with radius 0.6 (diameter 1.2m) won't fit
        assert not is_passable(bottleneck_navmesh, start, goal, agent_radius=0.6)

    def test_unreachable_goal(self, simple_navmesh):
        """Goal outside the navmesh is unreachable regardless of radius."""
        start = np.array([1.0, 2.5])
        goal = np.array([50.0, 50.0])
        assert not is_passable(simple_navmesh, start, goal, agent_radius=0.0)

    def test_same_triangle_always_passable(self, simple_navmesh):
        """Start and goal in the same triangle -- no portals to check."""
        start = np.array([1.0, 2.5])
        goal = np.array([1.5, 2.5])
        assert is_passable(simple_navmesh, start, goal, agent_radius=1.0)


class TestClearanceFactor:
    """Tests for the clearance_factor parameter."""

    def test_clearance_factor_rejects_borderline_agent(self, bottleneck_navmesh):
        """Agent that barely fits with factor=1.0 is rejected with factor=1.2."""
        start = np.array([2.0, 3.0])
        goal = np.array([18.0, 3.0])
        # Gap is ~1m, agent radius 0.45 -> diameter 0.9m fits at factor=1.0
        assert is_passable(
            bottleneck_navmesh, start, goal, agent_radius=0.45, clearance_factor=1.0
        )
        # With 1.2 factor: effective diameter = 0.9 * 1.2 = 1.08m > 1.0m gap
        assert not is_passable(
            bottleneck_navmesh, start, goal, agent_radius=0.45, clearance_factor=1.2
        )

    def test_clearance_factor_1_is_default(self, simple_navmesh):
        """Factor=1.0 should behave identically to the original check."""
        start = np.array([1.0, 2.5])
        goal = np.array([9.0, 2.5])
        assert is_passable(simple_navmesh, start, goal, agent_radius=0.3, clearance_factor=1.0)


class TestGeometricClearance:
    """Tests for the geometric path clearance validation (Stage 3)."""

    def test_close_obstacles_reject_wide_agent(self, close_obstacles_navmesh):
        """Agent too wide for 0.4m gap between close obstacles is rejected.

        This specifically tests the geometric clearance check -- the portal
        width check alone might pass if the portal edge is diagonal.
        """
        start = np.array([2.0, 3.0])
        goal = np.array([8.0, 3.0])
        # Agent radius 0.25 -> diameter 0.5m > 0.4m gap
        assert not is_passable(close_obstacles_navmesh, start, goal, agent_radius=0.25)

    def test_close_obstacles_allow_thin_agent(self, close_obstacles_navmesh):
        """Thin agent fits through the 0.4m gap."""
        start = np.array([2.0, 3.0])
        goal = np.array([8.0, 3.0])
        # Agent radius 0.1 -> diameter 0.2m < 0.4m gap
        assert is_passable(close_obstacles_navmesh, start, goal, agent_radius=0.1)

    def test_geometric_clearance_with_clearance_factor(self, close_obstacles_navmesh):
        """Clearance factor makes borderline agent fail geometric check."""
        start = np.array([2.0, 3.0])
        goal = np.array([8.0, 3.0])
        # Agent radius 0.15 -> diameter 0.3m fits in 0.4m gap at factor=1.0
        assert is_passable(
            close_obstacles_navmesh, start, goal, agent_radius=0.15, clearance_factor=1.0
        )
        # With 1.2 factor: effective diameter = 0.36m, still fits
        # With 1.5 factor: effective diameter = 0.45m > 0.4m gap
        assert not is_passable(
            close_obstacles_navmesh, start, goal, agent_radius=0.15, clearance_factor=1.5
        )

    def test_wide_open_passes_geometric_check(self, simple_navmesh):
        """Wide open rectangle passes geometric clearance trivially."""
        start = np.array([1.0, 2.5])
        goal = np.array([9.0, 2.5])
        assert is_passable(simple_navmesh, start, goal, agent_radius=0.3, clearance_factor=1.5)


# ---- Solvability verifier tests ----


class TestVerifySolvability:
    def test_all_solvable_prune(self, simple_navmesh):
        positions = np.array([[1.0, 2.5], [2.0, 2.5]], dtype=np.float64)
        goals = np.array([[8.0, 2.5], [9.0, 2.5]], dtype=np.float64)
        radii = np.array([0.2, 0.2], dtype=np.float64)

        mask = verify_solvability(simple_navmesh, positions, goals, radii, SolvabilityMode.PRUNE)
        assert mask is not None
        assert np.all(mask)

    def test_all_solvable_strict(self, simple_navmesh):
        positions = np.array([[1.0, 2.5], [2.0, 2.5]], dtype=np.float64)
        goals = np.array([[8.0, 2.5], [9.0, 2.5]], dtype=np.float64)
        radii = np.array([0.2, 0.2], dtype=np.float64)

        mask = verify_solvability(simple_navmesh, positions, goals, radii, SolvabilityMode.STRICT)
        assert mask is not None
        assert np.all(mask)

    def test_unreachable_goal_prune(self, simple_navmesh):
        """Agent with goal outside the navmesh is marked unsolvable."""
        positions = np.array([[1.0, 2.5], [2.0, 2.5]], dtype=np.float64)
        goals = np.array([[8.0, 2.5], [50.0, 50.0]], dtype=np.float64)
        radii = np.array([0.2, 0.2], dtype=np.float64)

        mask = verify_solvability(simple_navmesh, positions, goals, radii, SolvabilityMode.PRUNE)
        assert mask is not None
        assert mask[0]
        assert not mask[1]

    def test_too_wide_for_bottleneck_prune(self, bottleneck_navmesh):
        """Agent too wide for the bottleneck is marked unsolvable."""
        positions = np.array([[2.0, 3.0], [2.0, 3.0]], dtype=np.float64)
        goals = np.array([[18.0, 3.0], [18.0, 3.0]], dtype=np.float64)
        radii = np.array([0.2, 0.6], dtype=np.float64)  # second agent too wide

        mask = verify_solvability(
            bottleneck_navmesh, positions, goals, radii, SolvabilityMode.PRUNE
        )
        assert mask is not None
        assert mask[0]  # Small agent passes
        assert not mask[1]  # Large agent blocked

    def test_strict_returns_none_on_any_unsolvable(self, simple_navmesh):
        positions = np.array([[1.0, 2.5], [2.0, 2.5]], dtype=np.float64)
        goals = np.array([[8.0, 2.5], [50.0, 50.0]], dtype=np.float64)
        radii = np.array([0.2, 0.2], dtype=np.float64)

        mask = verify_solvability(simple_navmesh, positions, goals, radii, SolvabilityMode.STRICT)
        assert mask is None

    def test_regenerate_below_threshold(self, simple_navmesh):
        """One out of 5 unsolvable -- below 30% threshold -- returns mask."""
        positions = np.array([[1, 2.5], [2, 2.5], [3, 2.5], [4, 2.5], [5, 2.5]], dtype=np.float64)
        goals = np.array([[8, 2.5], [8, 2.5], [8, 2.5], [8, 2.5], [50, 50]], dtype=np.float64)
        radii = np.full(5, 0.2, dtype=np.float64)

        mask = verify_solvability(
            simple_navmesh, positions, goals, radii, SolvabilityMode.REGENERATE, 0.3
        )
        assert mask is not None
        assert np.sum(mask) == 4

    def test_regenerate_above_threshold(self, simple_navmesh):
        """Three out of 5 unsolvable -- above 30% threshold -- regenerate."""
        positions = np.array([[1, 2.5], [2, 2.5], [3, 2.5], [4, 2.5], [5, 2.5]], dtype=np.float64)
        goals = np.array([[8, 2.5], [8, 2.5], [50, 50], [50, 50], [50, 50]], dtype=np.float64)
        radii = np.full(5, 0.2, dtype=np.float64)

        mask = verify_solvability(
            simple_navmesh, positions, goals, radii, SolvabilityMode.REGENERATE, 0.3
        )
        assert mask is None

    def test_clearance_factor_passed_through(self, bottleneck_navmesh):
        """Clearance factor is propagated to is_passable."""
        positions = np.array([[2.0, 3.0]], dtype=np.float64)
        goals = np.array([[18.0, 3.0]], dtype=np.float64)
        # radius 0.45 -> diameter 0.9m fits in 1m gap at factor=1.0
        radii = np.array([0.45], dtype=np.float64)

        mask_loose = verify_solvability(
            bottleneck_navmesh, positions, goals, radii, clearance_factor=1.0
        )
        assert mask_loose is not None
        assert mask_loose[0]

        mask_strict = verify_solvability(
            bottleneck_navmesh, positions, goals, radii, clearance_factor=1.2
        )
        assert mask_strict is not None
        assert not mask_strict[0]


class TestFilterBySolvability:
    def test_filter_arrays(self):
        mask = np.array([True, False, True, False], dtype=np.bool_)
        positions = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        speeds = np.array([1.0, 1.5, 1.2, 1.3], dtype=np.float64)

        filtered_pos, filtered_speeds = filter_by_solvability(mask, positions, speeds)
        assert filtered_pos.shape == (2, 2)
        assert filtered_speeds.shape == (2,)
        np.testing.assert_array_equal(filtered_pos, [[1, 2], [5, 6]])
        np.testing.assert_array_equal(filtered_speeds, [1.0, 1.2])

    def test_filter_all_solvable(self):
        mask = np.ones(3, dtype=np.bool_)
        arr = np.arange(9, dtype=np.float64).reshape(3, 3)
        (result,) = filter_by_solvability(mask, arr)
        np.testing.assert_array_equal(result, arr)

    def test_filter_none_solvable(self):
        mask = np.zeros(3, dtype=np.bool_)
        arr = np.arange(6, dtype=np.float64).reshape(3, 2)
        (result,) = filter_by_solvability(mask, arr)
        assert result.shape == (0, 2)
