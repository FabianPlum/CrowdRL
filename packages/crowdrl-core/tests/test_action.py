"""Tests for the action interpreter."""

import numpy as np
import pytest

from crowdrl_core.action import (
    ActionConfig,
    ActionResult,
    interpret_action,
    interpret_actions_batch,
)


class TestInterpretAction:
    def test_zero_action(self):
        """Zero action should produce zero speed (since speed maps [-1,1] → [0, max])."""
        result = interpret_action(
            np.array([0.0, 0.0, 0.0, 0.0]),
            current_heading=0.0,
            current_torso=0.0,
            current_head=0.0,
        )
        # Speed: (0 + 1) / 2 * 1.5 = 0.75
        expected_speed = 0.75
        actual_speed = np.linalg.norm(result.desired_velocity)
        assert abs(actual_speed - expected_speed) < 1e-6

    def test_max_speed(self):
        """Action[0] = 1.0 should produce max speed."""
        result = interpret_action(
            np.array([1.0, 0.0, 0.0, 0.0]),
            current_heading=0.0,
            current_torso=0.0,
            current_head=0.0,
        )
        speed = np.linalg.norm(result.desired_velocity)
        assert abs(speed - 1.5) < 1e-6

    def test_zero_speed(self):
        """Action[0] = -1.0 should produce zero speed."""
        result = interpret_action(
            np.array([-1.0, 0.0, 0.0, 0.0]),
            current_heading=0.0,
            current_torso=0.0,
            current_head=0.0,
        )
        speed = np.linalg.norm(result.desired_velocity)
        assert speed < 1e-6

    def test_heading_change(self):
        """Positive heading action should turn left (CCW)."""
        config = ActionConfig(max_heading_change=np.pi / 4)
        result = interpret_action(
            np.array([0.0, 1.0, 0.0, 0.0]),
            current_heading=0.0,
            current_torso=0.0,
            current_head=0.0,
            config=config,
        )
        assert abs(result.new_heading - np.pi / 4) < 1e-6

    def test_heading_wraps(self):
        """Heading should wrap around ±π."""
        result = interpret_action(
            np.array([0.0, 1.0, 0.0, 0.0]),
            current_heading=3.0,  # Near π
            current_torso=3.0,
            current_head=3.0,
        )
        assert -np.pi <= result.new_heading <= np.pi

    def test_torso_independent(self):
        """Torso should change independently of heading."""
        config = ActionConfig()
        result = interpret_action(
            np.array([0.0, 0.0, 1.0, 0.0]),
            current_heading=0.0,
            current_torso=0.0,
            current_head=0.0,
            config=config,
        )
        assert abs(result.new_heading) < 1e-6  # Heading unchanged
        assert abs(result.new_torso_orientation - config.max_torso_change) < 1e-6

    def test_head_constraint(self):
        """Head should be clamped to ±90° relative to torso."""
        config = ActionConfig(head_limit=np.pi / 2)
        result = interpret_action(
            np.array([0.0, 0.0, 0.0, 1.0]),
            current_heading=0.0,
            current_torso=0.0,
            current_head=np.pi / 2 - 0.01,  # Already near limit
            config=config,
        )
        head_rel = result.new_head_orientation - result.new_torso_orientation
        assert abs(head_rel) <= np.pi / 2 + 1e-6

    def test_head_cannot_exceed_90_degrees(self):
        """Even with extreme action, head stays within ±90° of torso."""
        config = ActionConfig()
        # Try to push head far beyond limit
        result = interpret_action(
            np.array([0.0, 0.0, 0.0, 1.0]),
            current_heading=0.0,
            current_torso=0.0,
            current_head=np.pi / 2,  # Already at limit
            config=config,
        )
        head_rel = result.new_head_orientation - result.new_torso_orientation
        assert abs(head_rel) <= np.pi / 2 + 1e-6

    def test_velocity_direction_matches_heading(self):
        """Desired velocity direction should match new heading."""
        result = interpret_action(
            np.array([1.0, 0.0, 0.0, 0.0]),
            current_heading=np.pi / 4,
            current_torso=np.pi / 4,
            current_head=np.pi / 4,
        )
        vel_angle = np.arctan2(
            result.desired_velocity[1], result.desired_velocity[0]
        )
        assert abs(vel_angle - result.new_heading) < 1e-6

    def test_action_clipping(self):
        """Actions outside [-1, 1] should be clipped."""
        result = interpret_action(
            np.array([5.0, -5.0, 3.0, -3.0]),
            current_heading=0.0,
            current_torso=0.0,
            current_head=0.0,
        )
        speed = np.linalg.norm(result.desired_velocity)
        assert speed <= 1.5 + 1e-6

    def test_2d_action_mode(self):
        """With action_dim=2, torso and head should fuse with heading."""
        config = ActionConfig(action_dim=2)
        result = interpret_action(
            np.array([0.0, 0.5]),
            current_heading=0.0,
            current_torso=0.0,
            current_head=0.0,
            config=config,
        )
        assert abs(result.new_torso_orientation - result.new_heading) < 1e-6
        assert abs(result.new_head_orientation - result.new_heading) < 1e-6

    def test_3d_action_mode(self):
        """With action_dim=3, head should fuse with torso."""
        config = ActionConfig(action_dim=3)
        result = interpret_action(
            np.array([0.0, 0.0, 0.5]),
            current_heading=0.0,
            current_torso=0.0,
            current_head=0.0,
            config=config,
        )
        assert abs(result.new_head_orientation - result.new_torso_orientation) < 1e-6


class TestInterpretActionsBatch:
    def test_batch_matches_individual(self):
        n = 5
        rng = np.random.default_rng(42)
        actions = rng.uniform(-1, 1, (n, 4))
        headings = rng.uniform(-np.pi, np.pi, n)
        torsos = rng.uniform(-np.pi, np.pi, n)
        heads = rng.uniform(-np.pi, np.pi, n)

        batch_results = interpret_actions_batch(actions, headings, torsos, heads)
        assert len(batch_results) == n

        for i in range(n):
            individual = interpret_action(actions[i], headings[i], torsos[i], heads[i])
            np.testing.assert_allclose(
                batch_results[i].desired_velocity, individual.desired_velocity
            )
            assert abs(batch_results[i].new_heading - individual.new_heading) < 1e-10
