"""Tests for elliptical collision detection and contact forces."""

import numpy as np

from crowdrl_core.collision import (
    compute_contact_forces,
    detect_collisions,
    ellipse_overlap,
    enforce_wall_boundaries,
    ray_ellipse_intersection,
)
from conftest import make_world_state


class TestEllipseOverlap:
    def test_no_overlap(self):
        overlap = ellipse_overlap(
            np.array([0.0, 0.0]),
            0.23,
            0.15,
            0.0,
            np.array([2.0, 0.0]),
            0.23,
            0.15,
            0.0,
        )
        assert overlap == 0.0

    def test_touching(self):
        # Two circles (equal semi-axes) just touching
        overlap = ellipse_overlap(
            np.array([0.0, 0.0]),
            0.5,
            0.5,
            0.0,
            np.array([1.0, 0.0]),
            0.5,
            0.5,
            0.0,
        )
        # Should be at boundary — very small or zero overlap
        assert overlap <= 0.05

    def test_overlapping(self):
        overlap = ellipse_overlap(
            np.array([0.0, 0.0]),
            0.5,
            0.5,
            0.0,
            np.array([0.3, 0.0]),
            0.5,
            0.5,
            0.0,
        )
        assert overlap > 0

    def test_same_position(self):
        overlap = ellipse_overlap(
            np.array([0.0, 0.0]),
            0.23,
            0.15,
            0.0,
            np.array([0.0, 0.0]),
            0.23,
            0.15,
            0.0,
        )
        assert overlap > 0

    def test_rotated_ellipses_no_overlap(self):
        # Two thin ellipses side by side, not overlapping
        overlap = ellipse_overlap(
            np.array([0.0, 0.0]),
            0.5,
            0.1,
            0.0,  # Wide, thin
            np.array([0.0, 1.5]),
            0.5,
            0.1,
            0.0,
        )
        assert overlap == 0.0

    def test_rotated_ellipses_overlap(self):
        # Two ellipses rotated 90° to each other, close enough to overlap
        overlap = ellipse_overlap(
            np.array([0.0, 0.0]),
            0.5,
            0.2,
            0.0,
            np.array([0.3, 0.0]),
            0.5,
            0.2,
            np.pi / 2,
        )
        assert overlap > 0


class TestDetectCollisions:
    def test_no_collisions(self):
        world = make_world_state(
            n_agents=2,
            positions=np.array([[0.0, 0.0], [5.0, 5.0]]),
        )
        collisions = detect_collisions(world)
        assert len(collisions) == 0

    def test_agents_overlapping(self):
        world = make_world_state(
            n_agents=2,
            positions=np.array([[0.0, 0.0], [0.1, 0.0]]),
        )
        collisions = detect_collisions(world)
        assert len(collisions) == 1
        i, j, overlap = collisions[0]
        assert {i, j} == {0, 1}
        assert overlap > 0

    def test_inactive_agents_skipped(self):
        world = make_world_state(
            n_agents=2,
            positions=np.array([[0.0, 0.0], [0.1, 0.0]]),
        )
        world.active_mask = np.array([True, False])
        collisions = detect_collisions(world)
        assert len(collisions) == 0

    def test_multiple_collisions(self):
        world = make_world_state(
            n_agents=3,
            positions=np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]]),
            goal_positions=np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]),
        )
        collisions = detect_collisions(world)
        assert len(collisions) >= 2  # At least two pairs overlapping


class TestContactForces:
    def test_no_agent_collision_negligible_forces(self):
        # Agents at centre of 10×10 polygon — no agent-agent overlap,
        # wall repulsion is exponentially small at 3m+ distance
        world = make_world_state(
            n_agents=2,
            positions=np.array([[3.0, 5.0], [7.0, 5.0]]),
        )
        forces = compute_contact_forces(world)
        assert forces.shape == (2, 2)
        # Forces should be negligible (only tiny exponential wall repulsion)
        assert np.all(np.abs(forces) < 0.1)

    def test_colliding_agents_get_repulsive_forces(self):
        world = make_world_state(
            n_agents=2,
            positions=np.array([[5.0, 5.0], [5.1, 5.0]]),
        )
        forces = compute_contact_forces(world)
        # Agent 0 should be pushed in -x direction, agent 1 in +x
        assert forces[0, 0] < 0
        assert forces[1, 0] > 0

    def test_agent_agent_forces_symmetric(self):
        # At the centre of a symmetric polygon, wall forces on both agents
        # cancel by symmetry, so net force should be equal and opposite
        world = make_world_state(
            n_agents=2,
            positions=np.array([[4.9, 5.0], [5.1, 5.0]]),
        )
        forces = compute_contact_forces(world)
        np.testing.assert_allclose(forces[0] + forces[1], [0, 0], atol=1e-6)

    def test_wall_repulsion_increases_near_wall(self):
        # Agent near wall should get larger force than agent far from wall
        world_near = make_world_state(
            n_agents=1,
            positions=np.array([[0.5, 5.0]]),
            goal_positions=np.array([[8.0, 5.0]]),
        )
        world_far = make_world_state(
            n_agents=1,
            positions=np.array([[5.0, 5.0]]),
            goal_positions=np.array([[8.0, 5.0]]),
        )
        force_near = compute_contact_forces(world_near)
        force_far = compute_contact_forces(world_far)
        assert np.linalg.norm(force_near[0]) > np.linalg.norm(force_far[0])


class TestEnforceWallBoundaries:
    def test_agent_inside_near_wall_pushed_to_radius(self):
        # Agent at x=0.05 (inside polygon), body radius=0.23
        world = make_world_state(
            n_agents=1,
            positions=np.array([[0.05, 5.0]]),
            goal_positions=np.array([[8.0, 5.0]]),
        )
        enforce_wall_boundaries(world)
        radius = max(world.shoulder_widths[0], world.chest_depths[0])
        assert world.positions[0, 0] >= radius - 1e-6

    def test_agent_outside_polygon_snapped_back_inside(self):
        # Agent at x=-0.5 (outside polygon boundary at x=0)
        world = make_world_state(
            n_agents=1,
            positions=np.array([[-0.5, 5.0]]),
            velocities=np.array([[-1.0, 0.0]]),
            goal_positions=np.array([[8.0, 5.0]]),
        )
        enforce_wall_boundaries(world)
        radius = max(world.shoulder_widths[0], world.chest_depths[0])
        # Must be back inside at radius distance from boundary
        assert world.positions[0, 0] >= radius - 1e-6
        # Velocity into wall should be zeroed
        assert world.velocities[0, 0] >= 0

    def test_agent_crossed_through_wall_corrected(self):
        # Agent at y=4.15 (just past the y=4.0 boundary of a 4m-tall corridor)
        from shapely.geometry import Polygon

        corridor = Polygon([(0, 0), (10, 0), (10, 4), (0, 4)])
        world = make_world_state(
            n_agents=1,
            positions=np.array([[5.0, 4.15]]),
            velocities=np.array([[0.0, 1.0]]),
            goal_positions=np.array([[8.0, 2.0]]),
            polygon=corridor,
        )
        enforce_wall_boundaries(world)
        radius = max(world.shoulder_widths[0], world.chest_depths[0])
        # Must be inside the corridor with clearance
        assert world.positions[0, 1] <= 4.0 - radius + 1e-6
        assert world.positions[0, 1] > 0  # Still in corridor
        # Velocity into wall should be zeroed
        assert world.velocities[0, 1] <= 0

    def test_velocity_into_wall_zeroed(self):
        world = make_world_state(
            n_agents=1,
            positions=np.array([[0.1, 5.0]]),
            velocities=np.array([[-2.0, 0.5]]),
            goal_positions=np.array([[8.0, 5.0]]),
        )
        enforce_wall_boundaries(world)
        assert world.velocities[0, 0] >= 0

    def test_no_effect_when_far_from_walls(self):
        world = make_world_state(
            n_agents=1,
            positions=np.array([[5.0, 5.0]]),
            goal_positions=np.array([[8.0, 5.0]]),
        )
        pos_before = world.positions.copy()
        enforce_wall_boundaries(world)
        np.testing.assert_array_equal(world.positions, pos_before)

    def test_no_effect_without_polygon(self):
        world = make_world_state(
            n_agents=1,
            positions=np.array([[0.05, 5.0]]),
            goal_positions=np.array([[8.0, 5.0]]),
        )
        world.walkable_polygon = None
        pos_before = world.positions.copy()
        enforce_wall_boundaries(world)
        np.testing.assert_array_equal(world.positions, pos_before)

    def test_agent_inside_hole_pushed_out(self):
        # Polygon with a hole — agent placed inside the hole (obstacle)
        from shapely.geometry import Polygon

        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(4, 4), (6, 4), (6, 6), (4, 6)]
        poly = Polygon(exterior, [hole])
        world = make_world_state(
            n_agents=1,
            positions=np.array([[5.0, 5.0]]),  # Centre of hole
            goal_positions=np.array([[8.0, 8.0]]),
            polygon=poly,
        )
        enforce_wall_boundaries(world)
        # Agent should be pushed outside the hole
        assert poly.contains(Point(world.positions[0, 0], world.positions[0, 1]))


from shapely.geometry import Point  # noqa: E402


class TestRayEllipseIntersection:
    def test_hit_circle(self):
        # Ray from left hitting a unit circle at origin
        t = ray_ellipse_intersection(
            np.array([-3.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 0.0]),
            semi_width=1.0,
            semi_depth=1.0,
            ellipse_angle=0.0,
        )
        assert t is not None
        assert abs(t - 2.0) < 1e-6  # Hit at x = -1

    def test_miss(self):
        t = ray_ellipse_intersection(
            np.array([-3.0, 5.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 0.0]),
            semi_width=1.0,
            semi_depth=1.0,
            ellipse_angle=0.0,
        )
        assert t is None

    def test_hit_ellipse(self):
        # Ray from left, ellipse wider than deep
        t = ray_ellipse_intersection(
            np.array([-3.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 0.0]),
            semi_width=0.5,
            semi_depth=1.0,
            ellipse_angle=0.0,
        )
        assert t is not None
        assert abs(t - 2.0) < 1e-6

    def test_ray_from_inside(self):
        # Ray starting inside the ellipse
        t = ray_ellipse_intersection(
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 0.0]),
            semi_width=1.0,
            semi_depth=1.0,
            ellipse_angle=0.0,
        )
        assert t is not None
        assert abs(t - 1.0) < 1e-6  # Exits at x = 1

    def test_rotated_ellipse(self):
        # Ellipse rotated 90°: depth axis is now along y
        t = ray_ellipse_intersection(
            np.array([0.0, -3.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            semi_width=0.5,
            semi_depth=2.0,
            ellipse_angle=np.pi / 2,
        )
        assert t is not None
        assert t < 3.0  # Should hit before reaching centre
