"""Tests for elliptical collision detection and contact forces."""

import numpy as np

from crowdrl_core.collision import (
    compute_contact_forces,
    detect_collisions,
    ellipse_overlap,
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
    def test_no_collision_zero_forces(self):
        world = make_world_state(
            n_agents=2,
            positions=np.array([[0.0, 0.0], [5.0, 5.0]]),
        )
        forces = compute_contact_forces(world)
        assert forces.shape == (2, 2)
        np.testing.assert_array_equal(forces, np.zeros((2, 2)))

    def test_colliding_agents_get_repulsive_forces(self):
        world = make_world_state(
            n_agents=2,
            positions=np.array([[0.0, 0.0], [0.1, 0.0]]),
        )
        forces = compute_contact_forces(world)
        # Agent 0 should be pushed in -x direction, agent 1 in +x
        assert forces[0, 0] < 0
        assert forces[1, 0] > 0

    def test_forces_are_equal_and_opposite(self):
        world = make_world_state(
            n_agents=2,
            positions=np.array([[0.0, 0.0], [0.1, 0.0]]),
        )
        forces = compute_contact_forces(world)
        np.testing.assert_allclose(forces[0] + forces[1], [0, 0], atol=1e-10)


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
