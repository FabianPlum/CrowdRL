"""Tests for the action interpreter."""

import numpy as np

from crowdrl_core.action import (
    ActionConfig,
    interpret_action,
    interpret_actions_batch,
)


class TestInterpretAction:
    def test_zero_action_midpoint(self):
        """action[0]=0 lands at midpoint of [-max_back, +max_fwd] = +0.75 m/s."""
        result = interpret_action(
            np.array([0.0, 0.0, 0.0, 0.0]),
            current_heading=0.0,
            current_torso=0.0,
            current_head=0.0,
        )
        # Linear remap: -0.5 + (0 + 1) / 2 * 2.5 = +0.75 m/s
        expected_speed = 0.75
        # Magnitude only (negative desired_speed would also give positive magnitude)
        actual_speed = np.linalg.norm(result.desired_velocity)
        assert abs(actual_speed - expected_speed) < 1e-6

    def test_max_forward_speed(self):
        """action[0]=+1 produces max_forward_speed (default 2.0 m/s)."""
        result = interpret_action(
            np.array([1.0, 0.0, 0.0, 0.0]),
            current_heading=0.0,
            current_torso=0.0,
            current_head=0.0,
        )
        speed = np.linalg.norm(result.desired_velocity)
        assert abs(speed - 2.0) < 1e-6

    def test_max_backward_speed(self):
        """action[0]=-1 produces max_backward_speed magnitude (default 0.5 m/s),
        with the velocity vector pointing OPPOSITE to heading."""
        result = interpret_action(
            np.array([-1.0, 0.0, 0.0, 0.0]),
            current_heading=0.0,  # facing +x
            current_torso=0.0,
            current_head=0.0,
        )
        # Magnitude should equal max_backward_speed
        speed = np.linalg.norm(result.desired_velocity)
        assert abs(speed - 0.5) < 1e-6
        # Direction should be -x (opposite to heading)
        assert result.desired_velocity[0] < 0.0
        assert abs(result.desired_velocity[1]) < 1e-9

    def test_zero_speed_at_inverse_midpoint(self):
        """action[0] that maps to desired_speed=0 sits at the asymmetric
        zero-crossing (a0 = -max_back / (max_fwd + max_back) * 2 - 1)."""
        cfg = ActionConfig()
        # Inverse of: desired = -max_back + (a0+1)/2 * (max_fwd + max_back)
        speed_range = cfg.max_forward_speed + cfg.max_backward_speed
        a0_zero = 2.0 * cfg.max_backward_speed / speed_range - 1.0
        result = interpret_action(
            np.array([a0_zero, 0.0, 0.0, 0.0]),
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
        vel_angle = np.arctan2(result.desired_velocity[1], result.desired_velocity[0])
        assert abs(vel_angle - result.new_heading) < 1e-6

    def test_action_clipping(self):
        """Actions outside [-1, 1] should be clipped to max_forward_speed."""
        result = interpret_action(
            np.array([5.0, -5.0, 3.0, -3.0]),
            current_heading=0.0,
            current_torso=0.0,
            current_head=0.0,
        )
        speed = np.linalg.norm(result.desired_velocity)
        cfg = ActionConfig()
        assert speed <= cfg.max_forward_speed + 1e-6

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


class TestBiomechanicalEnvelope:
    """Layer 1 of agent_dynamics_refactor: default per-step caps should
    correspond to angular velocities within the human envelope at the
    default simulation rate (dt=0.01s).
    """

    def test_default_caps_within_human_envelope(self):
        """Heading/torso/head rates < 200 deg/s at dt=0.01s.

        Human walking yaw envelope (Hicheur 2007): 30-60 deg/s comfortable,
        ~120 deg/s aggressive cornering, ~360 deg/s only when pivoting in
        place. Layer 1 targets the walking envelope, not the pivot.
        """
        config = ActionConfig()
        dt = 0.01

        heading_rate_deg = np.degrees(config.max_heading_change) / dt
        torso_rate_deg = np.degrees(config.max_torso_change) / dt
        head_rate_deg = np.degrees(config.max_head_change) / dt

        assert heading_rate_deg < 200.0, (
            f"max_heading_change implies {heading_rate_deg:.0f} deg/s at "
            f"dt={dt}, above 200 deg/s human envelope"
        )
        assert torso_rate_deg < 200.0, (
            f"max_torso_change implies {torso_rate_deg:.0f} deg/s at dt={dt}"
        )
        assert head_rate_deg < 200.0, (
            f"max_head_change implies {head_rate_deg:.0f} deg/s at dt={dt}"
        )

    def test_default_caps_ordering_torso_heading_head(self):
        """Torso changes slower than heading; head changes fastest.

        Biomechanical ordering: hips constrain torso rotation independent
        of feet, so torso is the slowest axis. Head can scan freely on
        the neck, so it is the fastest. Heading sits between them.
        """
        config = ActionConfig()
        assert config.max_torso_change < config.max_heading_change
        assert config.max_heading_change < config.max_head_change


class TestInterpretActionsBatch:
    def test_batch_matches_individual(self):
        n = 5
        rng = np.random.default_rng(42)
        actions = rng.uniform(-1, 1, (n, 4))
        headings = rng.uniform(-np.pi, np.pi, n)
        torsos = rng.uniform(-np.pi, np.pi, n)
        heads = rng.uniform(-np.pi, np.pi, n)

        batch_results = interpret_actions_batch(actions, headings, torsos, heads)
        assert batch_results.desired_velocities.shape == (n, 2)
        assert batch_results.new_headings.shape == (n,)

        for i in range(n):
            individual = interpret_action(actions[i], headings[i], torsos[i], heads[i])
            np.testing.assert_allclose(
                batch_results.desired_velocities[i], individual.desired_velocity
            )
            assert abs(batch_results.new_headings[i] - individual.new_heading) < 1e-10
            assert (
                abs(batch_results.new_torso_orientations[i] - individual.new_torso_orientation)
                < 1e-10
            )
            assert (
                abs(batch_results.new_head_orientations[i] - individual.new_head_orientation)
                < 1e-10
            )


class TestSpeedTurnCoupling:
    """Speed-coupled yaw envelope: omega_max(v) = min(pivot, a_lat / v)."""

    def _cfg(self, **kw):
        params = dict(
            speed_turn_coupling=True,
            turn_lat_accel=2.0,
            turn_pivot_rate=2.0943951023931953,  # 120 deg/s
            dt=0.01,
            max_heading_change=1.0,  # large flat cap so coupling is the binding constraint
            max_torso_change=1.0,
        )
        params.update(kw)
        return ActionConfig(**params)

    def test_off_by_default_ignores_speed(self):
        """With coupling off (default), current_speed has no effect (regression)."""
        a = np.array([0.0, 1.0, 0.0, 0.0])
        r_slow = interpret_action(a, 0.0, 0.0, 0.0, current_speed=0.0)
        r_fast = interpret_action(a, 0.0, 0.0, 0.0, current_speed=5.0)
        assert abs(r_slow.new_heading - r_fast.new_heading) < 1e-12

    def test_high_speed_clamps_to_lateral_accel(self):
        """At speed v the max heading change is (a_lat / v) * dt."""
        cfg = self._cfg()
        v = 2.0
        r = interpret_action(
            np.array([0.0, 1.0, 0.0, 0.0]), 0.0, 0.0, 0.0, config=cfg, current_speed=v
        )
        expected = (cfg.turn_lat_accel / v) * cfg.dt  # 2.0/2.0*0.01 = 0.01 rad
        assert abs(r.new_heading - expected) < 1e-9

    def test_low_speed_capped_by_pivot_rate(self):
        """Near standstill a_lat/v explodes, so the pivot rate caps the turn."""
        cfg = self._cfg()
        r = interpret_action(
            np.array([0.0, 1.0, 0.0, 0.0]), 0.0, 0.0, 0.0, config=cfg, current_speed=1e-4
        )
        expected = cfg.turn_pivot_rate * cfg.dt
        assert abs(r.new_heading - expected) < 1e-9

    def test_slower_allows_sharper_turn(self):
        """Monotonic: lower speed -> larger admissible heading change."""
        cfg = self._cfg()
        a = np.array([0.0, 1.0, 0.0, 0.0])
        d = [
            interpret_action(a, 0.0, 0.0, 0.0, config=cfg, current_speed=v).new_heading
            for v in (0.5, 1.0, 2.0, 3.0)
        ]
        assert d[0] > d[1] > d[2] > d[3]

    def test_torso_obeys_same_envelope(self):
        """Torso change is clamped by the same speed envelope as heading."""
        cfg = self._cfg()
        v = 2.0
        r = interpret_action(
            np.array([0.0, 0.0, 1.0, 0.0]), 0.0, 0.0, 0.0, config=cfg, current_speed=v
        )
        assert abs(r.new_torso_orientation - (cfg.turn_lat_accel / v) * cfg.dt) < 1e-9

    def test_flat_cap_binds_when_smaller(self):
        """If the flat cap is tighter than the envelope, the flat cap wins."""
        cfg = self._cfg(max_heading_change=0.001)
        r = interpret_action(
            np.array([0.0, 1.0, 0.0, 0.0]), 0.0, 0.0, 0.0, config=cfg, current_speed=0.5
        )
        assert abs(r.new_heading - 0.001) < 1e-9

    def test_batch_matches_scalar(self):
        """Batch interpreter applies the same envelope as the scalar path."""
        cfg = self._cfg()
        speeds = np.array([0.5, 1.0, 2.0])
        actions = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (3, 1))
        zeros = np.zeros(3)
        batch = interpret_actions_batch(
            actions, zeros, zeros, zeros, config=cfg, current_speeds=speeds
        )
        for i, v in enumerate(speeds):
            scalar = interpret_action(
                actions[i], 0.0, 0.0, 0.0, config=cfg, current_speed=float(v)
            )
            assert abs(batch.new_headings[i] - scalar.new_heading) < 1e-9
