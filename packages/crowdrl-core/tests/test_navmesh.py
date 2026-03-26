"""Tests for navmesh A* pathfinding."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from crowdrl_core.geometry import build_navmesh
from crowdrl_core.navmesh import (
    astar_triangle_path,
    find_path,
    is_reachable,
    next_waypoint_direction,
    path_deviation,
)
from conftest import make_world_state


class TestAstarTrianglePath:
    def test_same_triangle(self, simple_square_polygon):
        nm = build_navmesh(simple_square_polygon)
        path = astar_triangle_path(nm, 0, 0)
        assert path == [0]

    def test_adjacent_triangles(self, simple_square_polygon):
        nm = build_navmesh(simple_square_polygon)
        if len(nm.adjacency[0]) > 0:
            neighbour = nm.adjacency[0][0]
            path = astar_triangle_path(nm, 0, neighbour)
            assert path == [0, neighbour]

    def test_path_exists_in_square(self, simple_square_polygon):
        nm = build_navmesh(simple_square_polygon)
        n_tris = len(nm.adjacency)
        # All triangles should be reachable from triangle 0
        for i in range(n_tris):
            path = astar_triangle_path(nm, 0, i)
            assert path is not None, f"No path from 0 to {i}"
            assert path[0] == 0
            assert path[-1] == i


class TestFindPath:
    def test_simple_path(self, simple_square_polygon):
        nm = build_navmesh(simple_square_polygon)
        start = np.array([1.0, 1.0])
        goal = np.array([9.0, 9.0])
        path = find_path(nm, start, goal)
        assert path is not None
        assert len(path) >= 1

    def test_point_outside_returns_none(self, simple_square_polygon):
        nm = build_navmesh(simple_square_polygon)
        start = np.array([1.0, 1.0])
        goal = np.array([15.0, 15.0])  # Outside
        path = find_path(nm, start, goal)
        assert path is None

    def test_corridor_end_to_end(self, corridor_polygon):
        nm = build_navmesh(corridor_polygon)
        start = np.array([1.0, 1.5])
        goal = np.array([19.0, 1.5])
        path = find_path(nm, start, goal)
        assert path is not None
        assert len(path) >= 2  # Should traverse multiple triangles

    def test_around_obstacle(self, square_with_obstacle):
        nm = build_navmesh(square_with_obstacle)
        start = np.array([1.0, 5.0])
        goal = np.array([9.0, 5.0])
        path = find_path(nm, start, goal)
        assert path is not None

    def test_through_bottleneck(self, bottleneck_polygon):
        nm = build_navmesh(bottleneck_polygon)
        start = np.array([1.0, 3.0])
        goal = np.array([9.0, 3.0])
        path = find_path(nm, start, goal)
        assert path is not None


class TestIsReachable:
    def test_reachable_in_square(self, simple_square_polygon):
        nm = build_navmesh(simple_square_polygon)
        assert is_reachable(nm, np.array([1.0, 1.0]), np.array([9.0, 9.0]))

    def test_unreachable_outside(self, simple_square_polygon):
        nm = build_navmesh(simple_square_polygon)
        assert not is_reachable(nm, np.array([1.0, 1.0]), np.array([15.0, 15.0]))


class TestNextWaypointDirection:
    def test_returns_unit_vector(self, simple_square_polygon):
        nm = build_navmesh(simple_square_polygon)
        pos = np.array([1.0, 1.0])
        goal = np.array([9.0, 9.0])
        direction = next_waypoint_direction(nm, pos, goal)
        assert direction is not None
        norm = np.linalg.norm(direction)
        assert abs(norm - 1.0) < 1e-6 or norm < 1e-10

    def test_points_generally_toward_goal(self, simple_square_polygon):
        nm = build_navmesh(simple_square_polygon)
        pos = np.array([1.0, 1.0])
        goal = np.array([9.0, 9.0])
        direction = next_waypoint_direction(nm, pos, goal)
        assert direction is not None
        # Direction should have positive x and y components (toward goal)
        goal_dir = goal - pos
        goal_dir = goal_dir / np.linalg.norm(goal_dir)
        dot = np.dot(direction, goal_dir)
        assert dot > 0, "Waypoint direction should generally face the goal"

    def test_unreachable_returns_none(self, simple_square_polygon):
        nm = build_navmesh(simple_square_polygon)
        pos = np.array([1.0, 1.0])
        goal = np.array([15.0, 15.0])
        assert next_waypoint_direction(nm, pos, goal) is None


class TestPathDeviation:
    def test_straight_line_low_deviation(self, simple_square_polygon):
        nm = build_navmesh(simple_square_polygon)
        pos = np.array([1.0, 5.0])
        goal = np.array([9.0, 5.0])
        dev = path_deviation(nm, pos, goal)
        assert dev is not None
        # In a convex polygon, path deviation should be small
        assert dev >= 0.0

    def test_obstacle_increases_deviation(self, square_with_obstacle):
        nm = build_navmesh(square_with_obstacle)
        # Path that must go around obstacle
        pos = np.array([1.0, 5.0])
        goal = np.array([9.0, 5.0])
        dev = path_deviation(nm, pos, goal)
        assert dev is not None
        assert dev >= 0.0

    def test_unreachable_returns_none(self, simple_square_polygon):
        nm = build_navmesh(simple_square_polygon)
        dev = path_deviation(nm, np.array([1.0, 1.0]), np.array([15.0, 15.0]))
        assert dev is None

    def test_same_position_zero_deviation(self, simple_square_polygon):
        nm = build_navmesh(simple_square_polygon)
        pos = np.array([5.0, 5.0])
        dev = path_deviation(nm, pos, pos)
        assert dev is not None
        assert dev == 0.0
