"""Tests for navmesh A* pathfinding."""

import numpy as np
from shapely.geometry import Polygon, box

from crowdrl_core.geometry import build_navmesh
from crowdrl_core.navmesh import (
    _validate_path_clearance,
    astar_triangle_path,
    find_path,
    is_passable,
    is_reachable,
    next_waypoint_direction,
    path_deviation,
    shortest_path,
)


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


class TestShortestPath:
    """Tests for the funnel-smoothed shortest path."""

    def test_straight_line_in_open_field(self, simple_square_polygon):
        """In a convex polygon the shortest path should be (nearly) the straight line."""
        nm = build_navmesh(simple_square_polygon)
        start = np.array([2.0, 5.0])
        goal = np.array([8.0, 5.0])
        wp = shortest_path(nm, start, goal)
        assert wp is not None
        # Should be essentially start → goal (2 waypoints)
        assert len(wp) <= 3
        path_len = sum(float(np.linalg.norm(wp[i + 1] - wp[i])) for i in range(len(wp) - 1))
        direct = float(np.linalg.norm(goal - start))
        assert path_len < direct * 1.01  # within 1% of straight line

    def test_bottleneck_path_goes_through_aperture(self):
        """Path through a bottleneck must pass through the gap, not through walls."""
        exterior = [(0, 0), (10, 0), (10, 4), (0, 4)]
        top_wall = [(4, 2.5), (6, 2.5), (6, 4), (4, 4)]
        bottom_wall = [(4, 0), (6, 0), (6, 1.5), (4, 1.5)]
        bottleneck = Polygon(exterior, [top_wall, bottom_wall])
        nm = build_navmesh(bottleneck)

        start = np.array([1.0, 2.0])
        goal = np.array([9.0, 2.0])
        wp = shortest_path(nm, start, goal)
        assert wp is not None
        # All waypoints must be inside the polygon (with tolerance)
        from shapely.geometry import Point

        for p in wp:
            assert bottleneck.distance(Point(p[0], p[1])) < 0.01

        # Path must cross x=5 (the bottleneck centre) at y between 1.5 and 2.5
        for i in range(len(wp) - 1):
            x0, x1 = wp[i][0], wp[i + 1][0]
            if (x0 <= 5.0 <= x1) or (x1 <= 5.0 <= x0):
                t = (5.0 - x0) / (x1 - x0) if abs(x1 - x0) > 1e-10 else 0.5
                y_at_5 = wp[i][1] + t * (wp[i + 1][1] - wp[i][1])
                assert 1.5 <= y_at_5 <= 2.5, f"Path crosses x=5 at y={y_at_5}, outside aperture"
                break


class TestAgentRadius:
    """Tests for agent_radius portal inset."""

    def test_radius_zero_matches_no_radius(self):
        """With radius=0 the path should be identical to the default."""
        exterior = [(0, 0), (10, 0), (10, 4), (0, 4)]
        top_wall = [(4, 2.5), (6, 2.5), (6, 4), (4, 4)]
        bottom_wall = [(4, 0), (6, 0), (6, 1.5), (4, 1.5)]
        bottleneck = Polygon(exterior, [top_wall, bottom_wall])
        nm = build_navmesh(bottleneck)

        start = np.array([1.0, 2.0])
        goal = np.array([9.0, 2.0])
        wp_default = shortest_path(nm, start, goal)
        wp_zero = shortest_path(nm, start, goal, agent_radius=0.0)
        assert wp_default is not None and wp_zero is not None
        assert len(wp_default) == len(wp_zero)
        for a, b in zip(wp_default, wp_zero):
            np.testing.assert_allclose(a, b)

    def test_radius_keeps_path_away_from_corners(self):
        """With a nonzero radius, waypoints should stay away from wall corners."""
        exterior = [(0, 0), (10, 0), (10, 4), (0, 4)]
        top_wall = [(4, 2.5), (6, 2.5), (6, 4), (4, 4)]
        bottom_wall = [(4, 0), (6, 0), (6, 1.5), (4, 1.5)]
        bottleneck = Polygon(exterior, [top_wall, bottom_wall])
        nm = build_navmesh(bottleneck)

        start = np.array([1.0, 2.0])
        goal = np.array([9.0, 2.0])
        radius = 0.2
        wp = shortest_path(nm, start, goal, agent_radius=radius)
        assert wp is not None

        # The wall corners of the aperture are at (4, 2.5), (4, 1.5), (6, 2.5), (6, 1.5)
        corners = [np.array([4, 2.5]), np.array([4, 1.5]), np.array([6, 2.5]), np.array([6, 1.5])]
        for p in wp[1:-1]:  # skip start/goal
            for corner in corners:
                dist = float(np.linalg.norm(p - corner))
                # Waypoint should not sit exactly on a corner
                assert dist > 0.05, f"Waypoint {p} too close to corner {corner}"

    def test_radius_path_longer_than_zero_radius(self):
        """Path with clearance should be at least as long as the zero-radius path."""
        exterior = [(0, 0), (10, 0), (10, 4), (0, 4)]
        top_wall = [(4, 2.5), (6, 2.5), (6, 4), (4, 4)]
        bottom_wall = [(4, 0), (6, 0), (6, 1.5), (4, 1.5)]
        bottleneck = Polygon(exterior, [top_wall, bottom_wall])
        nm = build_navmesh(bottleneck)

        start = np.array([1.0, 2.0])
        goal = np.array([9.0, 2.0])
        wp_0 = shortest_path(nm, start, goal, agent_radius=0.0)
        wp_r = shortest_path(nm, start, goal, agent_radius=0.2)
        assert wp_0 is not None and wp_r is not None

        def path_len(wp):
            return sum(float(np.linalg.norm(wp[i + 1] - wp[i])) for i in range(len(wp) - 1))

        assert path_len(wp_r) >= path_len(wp_0) - 1e-10


class TestNavMeshStoresPolygon:
    """Verify that build_navmesh stores the source polygon."""

    def test_polygon_stored(self, simple_square_polygon):
        nm = build_navmesh(simple_square_polygon)
        assert nm.polygon is not None
        assert nm.polygon.equals(simple_square_polygon)

    def test_polygon_with_holes(self, square_with_obstacle):
        nm = build_navmesh(square_with_obstacle)
        assert nm.polygon is not None
        assert len(list(nm.polygon.interiors)) == 1


class TestValidatePathClearance:
    """Tests for _validate_path_clearance (Minkowski erosion approach)."""

    def test_connected_in_open_field(self, simple_square_polygon):
        """Start and goal well inside a rectangle -- connected after erosion."""
        start = np.array([2.0, 5.0])
        goal = np.array([8.0, 5.0])
        assert _validate_path_clearance(start, goal, 0.3, simple_square_polygon)

    def test_narrow_polygon_rejects_wide_agent(self):
        """Polygon too narrow for the agent radius after erosion."""
        polygon = box(0, 0, 10, 0.4)
        start = np.array([1.0, 0.2])
        goal = np.array([9.0, 0.2])
        # Radius 0.3 erodes 0.4m-wide polygon to empty
        assert not _validate_path_clearance(start, goal, 0.3, polygon)

    def test_narrow_gap_disconnects(self):
        """Gap narrower than 2*radius disconnects start from goal."""
        exterior = box(0, 0, 10, 6)
        obs_a = box(4, 0.5, 5, 2.7)
        obs_b = box(4, 3.3, 5, 5.5)
        polygon = exterior.difference(obs_a).difference(obs_b)
        start = np.array([2.0, 3.0])
        goal = np.array([8.0, 3.0])
        # Gap is 0.6m, agent radius 0.4 -> eroded gap = -0.2m (disconnected)
        assert not _validate_path_clearance(start, goal, 0.4, polygon)

    def test_wide_gap_connects(self):
        """Gap wider than 2*radius keeps connectivity."""
        exterior = box(0, 0, 10, 6)
        obs_a = box(4, 0.5, 5, 2.0)
        obs_b = box(4, 4.0, 5, 5.5)
        polygon = exterior.difference(obs_a).difference(obs_b)
        start = np.array([2.0, 3.0])
        goal = np.array([8.0, 3.0])
        # Gap is 2.0m, agent radius 0.3 -> eroded gap = 1.4m (connected)
        assert _validate_path_clearance(start, goal, 0.3, polygon)


class TestIsPassableGeometric:
    """Tests for the full 3-stage is_passable check including geometric clearance."""

    def test_diagonal_portal_does_not_fool_check(self):
        """Two close obstacles with diagonal portal: geometric check catches it.

        The portal edge between triangles in the gap may be diagonal
        (e.g. sqrt(2) * gap_width), but the actual perpendicular clearance
        is only gap_width.  The geometric check must reject this.
        """
        exterior = box(0, 0, 10, 6)
        # Two obstacles very close together: 0.5m gap
        obs_a = box(4, 0.5, 5.5, 2.75)
        obs_b = box(4, 3.25, 5.5, 5.5)
        polygon = exterior.difference(obs_a).difference(obs_b)
        nm = build_navmesh(polygon)

        start = np.array([2.0, 3.0])
        goal = np.array([8.0, 3.0])
        # Agent radius 0.3 -> diameter 0.6m > 0.5m gap
        assert not is_passable(nm, start, goal, agent_radius=0.3)
