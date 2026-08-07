"""Tests for the reward module."""

import numpy as np
import pytest

from crowdrl_env.reward import RewardConfig, RewardState, compute_rewards


@pytest.fixture
def default_config():
    return RewardConfig()


@pytest.fixture
def no_smoothness_config():
    return RewardConfig(use_smoothness=False)


def _make_state(n_agents, goal_distances=None):
    state = RewardState()
    if goal_distances is None:
        goal_distances = np.ones(n_agents) * 10.0
    state.reset(n_agents, goal_distances)
    return state


class TestTier1Sparse:
    def test_goal_reaching_bonus(self, no_smoothness_config):
        cfg = no_smoothness_config
        n = 3
        positions = np.array([[10.0, 0.0], [0.0, 0.0], [5.0, 0.0]])
        goals = np.array([[10.1, 0.0], [20.0, 0.0], [5.2, 0.0]])
        velocities = np.ones((n, 2))
        headings = np.zeros(n)
        preferred_speeds = np.ones(n) * 1.34
        active = np.ones(n, dtype=np.bool_)
        collision = np.zeros(n, dtype=np.bool_)
        state = _make_state(n)

        rewards, reached = compute_rewards(
            positions,
            velocities,
            headings,
            goals,
            preferred_speeds,
            active,
            collision,
            state,
            cfg,
            dt=0.01,
        )

        # Agents 0 and 2 are within goal_radius (0.5m)
        assert reached[0]
        assert not reached[1]
        assert reached[2]
        assert rewards[0] > cfg.goal_bonus * 0.5  # At least the bonus
        assert rewards[2] > cfg.goal_bonus * 0.5

    def test_collision_penalty(self, no_smoothness_config):
        cfg = no_smoothness_config
        n = 2
        positions = np.array([[0.0, 0.0], [5.0, 0.0]])
        goals = np.array([[10.0, 0.0], [10.0, 0.0]])
        velocities = np.ones((n, 2))
        headings = np.zeros(n)
        preferred_speeds = np.ones(n) * 1.34
        active = np.ones(n, dtype=np.bool_)
        collision = np.array([True, False], dtype=np.bool_)
        state = _make_state(n)

        rewards, _ = compute_rewards(
            positions,
            velocities,
            headings,
            goals,
            preferred_speeds,
            active,
            collision,
            state,
            cfg,
            dt=0.01,
        )

        # Agent 0 has collision penalty, agent 1 does not
        assert rewards[0] < rewards[1]

    def test_inactive_agents_get_no_reward(self, no_smoothness_config):
        cfg = no_smoothness_config
        n = 2
        positions = np.array([[0.0, 0.0], [5.0, 0.0]])
        goals = np.array([[10.0, 0.0], [10.0, 0.0]])
        velocities = np.ones((n, 2))
        headings = np.zeros(n)
        preferred_speeds = np.ones(n) * 1.34
        active = np.array([True, False], dtype=np.bool_)
        collision = np.zeros(n, dtype=np.bool_)
        state = _make_state(n)

        rewards, _ = compute_rewards(
            positions,
            velocities,
            headings,
            goals,
            preferred_speeds,
            active,
            collision,
            state,
            cfg,
            dt=0.01,
        )

        assert rewards[1] == 0.0

    def test_progress_reward(self, no_smoothness_config):
        """Agent moving toward goal gets positive progress reward."""
        cfg = no_smoothness_config
        n = 1
        # Start at (0, 0), goal at (10, 0), prev distance was 10
        positions = np.array([[1.0, 0.0]])  # Now closer
        goals = np.array([[10.0, 0.0]])
        velocities = np.array([[1.0, 0.0]])
        headings = np.zeros(n)
        preferred_speeds = np.ones(n) * 1.34
        active = np.ones(n, dtype=np.bool_)
        collision = np.zeros(n, dtype=np.bool_)
        state = _make_state(n, goal_distances=np.array([10.0]))

        rewards, _ = compute_rewards(
            positions,
            velocities,
            headings,
            goals,
            preferred_speeds,
            active,
            collision,
            state,
            cfg,
            dt=0.01,
        )

        # Progress = 10 - 9 = 1, reward += 0.1 * 1 = 0.1
        assert rewards[0] > 0


class TestSpeedDeviationIndependentOfSmoothness:
    """``speed_deviation_weight`` must work with ``use_smoothness=False``.

    It used to be nested inside the ``use_smoothness`` block, so every config that
    turned smoothness off -- which is the baseline setting -- silently trained with
    NO speed matching no matter what weight it asked for. That is invisible in the
    logs: the smoothness channel just reads +0.00. Policies aimed at JuPedSim
    (constant 1.0 m/s preferred speed) depend on this term, so pin it here.

    Must stay in lockstep with the torch twin in crowdrl_torch.reward.
    """

    def _reward(self, weight: float) -> np.ndarray:
        # Every other term zeroed (incl. the collision penalty, which would not
        # fire anyway -- these agents never touch) so the delta isolates speed_dev.
        cfg = _collision_only_config(
            collision_penalty=0.0,
            use_smoothness=False,
            speed_deviation_weight=weight,
        )
        n = 3
        velocities = np.array([[0.2, 0.0], [1.0, 0.0], [1.8, 0.0]])
        state = _make_state(n)
        state.prev_velocities = velocities.copy()
        rewards, _ = compute_rewards(
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            velocities,
            np.zeros(n),
            np.array([[100.0, 0.0]] * n),
            np.array([1.0, 1.0, 1.0]),
            np.ones(n, dtype=np.bool_),
            np.zeros(n, dtype=np.bool_),
            state,
            cfg,
            dt=0.01,
        )
        return np.asarray(rewards)

    def test_applies_when_smoothness_disabled(self):
        delta = self._reward(-0.5) - self._reward(0.0)
        # |speed - 1.0| = [0.8, 0.0, 0.8]  ->  -0.5 * that
        assert np.allclose(delta, [-0.4, 0.0, -0.4])

    def test_agent_at_preferred_speed_is_unpenalised(self):
        """The middle agent moves at exactly its preferred speed."""
        assert self._reward(-0.5)[1] == pytest.approx(0.0)

    def test_zero_weight_is_a_no_op(self):
        assert np.allclose(self._reward(0.0), 0.0)


class TestTier2Smoothness:
    def test_speed_deviation_penalty(self, default_config):
        """Agent moving much faster than preferred gets penalised."""
        cfg = default_config
        n = 1
        positions = np.array([[0.0, 0.0]])
        goals = np.array([[10.0, 0.0]])
        # Very fast velocity (speed ~2.83 vs preferred 1.34)
        velocities = np.array([[2.0, 2.0]])
        headings = np.zeros(n)
        preferred_speeds = np.array([1.34])
        active = np.ones(n, dtype=np.bool_)
        collision = np.zeros(n, dtype=np.bool_)
        state = _make_state(n)

        # First step: sets prev_velocities
        compute_rewards(
            positions,
            velocities,
            headings,
            goals,
            preferred_speeds,
            active,
            collision,
            state,
            cfg,
            dt=0.01,
        )

        # Second step: smoothness kicks in
        rewards, _ = compute_rewards(
            positions,
            velocities,
            headings,
            goals,
            preferred_speeds,
            active,
            collision,
            state,
            cfg,
            dt=0.01,
        )

        # Speed deviation penalty should make reward negative
        # (speed ~2.83 - 1.34 = ~1.49, penalty = -0.1 * 1.49 ≈ -0.15)
        assert rewards[0] < 0

    def test_smooth_motion_less_penalty(self, default_config):
        """Constant velocity has less smoothness penalty than jerky motion."""
        cfg = default_config
        n = 1
        dt = 0.01

        # Constant velocity run
        state_smooth = _make_state(n)
        vel_const = np.array([[1.0, 0.0]])
        positions = np.array([[0.0, 0.0]])
        goals = np.array([[10.0, 0.0]])
        headings = np.zeros(n)
        preferred_speeds = np.array([1.0])  # Match velocity for minimal penalty
        active = np.ones(n, dtype=np.bool_)
        collision = np.zeros(n, dtype=np.bool_)

        for _ in range(3):
            r_smooth, _ = compute_rewards(
                positions,
                vel_const,
                headings,
                goals,
                preferred_speeds,
                active,
                collision,
                state_smooth,
                cfg,
                dt,
            )

        # Jerky velocity run
        state_jerky = _make_state(n)
        vels = [
            np.array([[1.0, 0.0]]),
            np.array([[0.0, 1.0]]),
            np.array([[-1.0, 0.0]]),
        ]

        for v in vels:
            r_jerky, _ = compute_rewards(
                positions,
                v,
                headings,
                goals,
                preferred_speeds,
                active,
                collision,
                state_jerky,
                cfg,
                dt,
            )

        # Smooth motion should have less negative reward from smoothness terms
        # (though progress might differ, the smoothness penalty should be visible)
        # Just check jerk produces more penalty overall
        assert r_smooth[0] >= r_jerky[0]


class TestWallProximityPenalty:
    def test_wall_proximity_penalty_applied(self):
        """Agents close to walls receive a penalty."""
        cfg = RewardConfig(use_smoothness=False, wall_proximity_penalty=-0.3)
        n = 2
        positions = np.array([[0.0, 0.0], [5.0, 0.0]])
        goals = np.array([[10.0, 0.0], [10.0, 0.0]])
        velocities = np.ones((n, 2))
        headings = np.zeros(n)
        preferred_speeds = np.ones(n) * 1.34
        active = np.ones(n, dtype=np.bool_)
        collision = np.zeros(n, dtype=np.bool_)
        state = _make_state(n)

        # Agent 0 is very close to a wall, agent 1 is far
        wall_distances = np.array([0.1, 5.0])
        agent_radii = np.array([0.22, 0.22])  # threshold = 0.22 * 1.5 = 0.33

        rewards, _ = compute_rewards(
            positions,
            velocities,
            headings,
            goals,
            preferred_speeds,
            active,
            collision,
            state,
            cfg,
            dt=0.01,
            wall_distances=wall_distances,
            agent_radii=agent_radii,
        )

        # Agent 0 should have the wall penalty, agent 1 should not
        assert rewards[0] < rewards[1]

    def test_wall_proximity_disabled_when_zero(self):
        """No wall penalty when weight is 0."""
        cfg = RewardConfig(use_smoothness=False, wall_proximity_penalty=0.0)
        n = 1
        positions = np.array([[0.0, 0.0]])
        goals = np.array([[10.0, 0.0]])
        velocities = np.ones((n, 2))
        headings = np.zeros(n)
        preferred_speeds = np.ones(n) * 1.34
        active = np.ones(n, dtype=np.bool_)
        collision = np.zeros(n, dtype=np.bool_)
        state = _make_state(n)

        wall_distances = np.array([0.05])
        agent_radii = np.array([0.22])

        rewards_with, _ = compute_rewards(
            positions,
            velocities,
            headings,
            goals,
            preferred_speeds,
            active,
            collision,
            state,
            cfg,
            dt=0.01,
            wall_distances=wall_distances,
            agent_radii=agent_radii,
        )

        state2 = _make_state(n)
        rewards_without, _ = compute_rewards(
            positions,
            velocities,
            headings,
            goals,
            preferred_speeds,
            active,
            collision,
            state2,
            cfg,
            dt=0.01,
        )

        np.testing.assert_allclose(rewards_with, rewards_without)


class TestActionRatePenalty:
    def test_action_rate_penalty_applied(self):
        """Large action changes are penalised."""
        cfg = RewardConfig(use_smoothness=False, action_rate_weight=-0.05)
        n = 1
        positions = np.array([[0.0, 0.0]])
        goals = np.array([[10.0, 0.0]])
        velocities = np.ones((n, 2))
        headings = np.zeros(n)
        preferred_speeds = np.ones(n) * 1.34
        active = np.ones(n, dtype=np.bool_)
        collision = np.zeros(n, dtype=np.bool_)
        state = _make_state(n)

        # First step: set prev_actions
        actions_t0 = np.array([[0.0, 0.0, 0.0, 0.0]])
        compute_rewards(
            positions,
            velocities,
            headings,
            goals,
            preferred_speeds,
            active,
            collision,
            state,
            cfg,
            dt=0.01,
            actions=actions_t0,
        )

        # Second step: large change
        actions_t1 = np.array([[1.0, 1.0, 1.0, 1.0]])
        rewards_big, _ = compute_rewards(
            positions,
            velocities,
            headings,
            goals,
            preferred_speeds,
            active,
            collision,
            state,
            cfg,
            dt=0.01,
            actions=actions_t1,
        )

        # Reset and do small change
        state2 = _make_state(n)
        compute_rewards(
            positions,
            velocities,
            headings,
            goals,
            preferred_speeds,
            active,
            collision,
            state2,
            cfg,
            dt=0.01,
            actions=actions_t0,
        )
        actions_t1_small = np.array([[0.01, 0.01, 0.01, 0.01]])
        rewards_small, _ = compute_rewards(
            positions,
            velocities,
            headings,
            goals,
            preferred_speeds,
            active,
            collision,
            state2,
            cfg,
            dt=0.01,
            actions=actions_t1_small,
        )

        # Big action change should produce more penalty
        assert rewards_big[0] < rewards_small[0]

    def test_action_rate_no_penalty_first_step(self):
        """No action rate penalty on the first step (no prev_actions)."""
        cfg = RewardConfig(use_smoothness=False, action_rate_weight=-0.05)
        n = 1
        positions = np.array([[0.0, 0.0]])
        goals = np.array([[10.0, 0.0]])
        velocities = np.ones((n, 2))
        headings = np.zeros(n)
        preferred_speeds = np.ones(n) * 1.34
        active = np.ones(n, dtype=np.bool_)
        collision = np.zeros(n, dtype=np.bool_)
        state = _make_state(n)

        actions = np.array([[1.0, 1.0, 1.0, 1.0]])
        rewards_with, _ = compute_rewards(
            positions,
            velocities,
            headings,
            goals,
            preferred_speeds,
            active,
            collision,
            state,
            cfg,
            dt=0.01,
            actions=actions,
        )

        state2 = _make_state(n)
        rewards_without, _ = compute_rewards(
            positions,
            velocities,
            headings,
            goals,
            preferred_speeds,
            active,
            collision,
            state2,
            cfg,
            dt=0.01,
        )

        # First step should have no action rate penalty
        np.testing.assert_allclose(rewards_with, rewards_without)


class TestRewardState:
    def test_reset_clears_state(self):
        state = RewardState()
        state.prev_velocities = np.zeros((3, 2))
        state.prev_accelerations = np.zeros((3, 2))

        state.reset(5, np.ones(5) * 10.0)
        assert state.prev_velocities is None
        assert state.prev_accelerations is None
        assert state.prev_nav_distances is not None
        assert len(state.prev_nav_distances) == 5


def _collision_only_config(**overrides):
    """RewardConfig with every term except the collision/wall penalties zeroed,
    so a single compute_rewards call isolates the collision reward."""
    base = dict(
        goal_bonus=0.0,
        collision_penalty=-2.0,
        timeout_penalty=0.0,
        wall_proximity_penalty=0.0,
        wall_collision_penalty=0.0,
        agent_proximity_penalty_near=0.0,
        agent_proximity_penalty_far=0.0,
        action_rate_weight=0.0,
        use_smoothness=False,
        speed_deviation_weight=0.0,
        existence_penalty=0.0,
        progress_weight=0.0,
        inverse_distance_weight=0.0,
    )
    base.update(overrides)
    return RewardConfig(**base)


class TestVelocityWeightedCollision:
    """Impact-speed weighting of the collision / wall-contact penalties (P1)."""

    # Two agents at 0.5 m separation; contact = r_i+r_j = 0.6, 1.2x = 0.72 > 0.5.
    _POS = np.array([[0.0, 0.0], [0.5, 0.0]])
    _GOALS = np.array([[100.0, 0.0], [-100.0, 0.0]])  # far -> no goal bonus
    _RADII = np.array([0.3, 0.3])
    _HEAD = np.zeros(2)
    _PREF = np.ones(2) * 1.34

    def _collide(self, cfg, velocities, collision_velocities=None):
        rewards, _ = compute_rewards(
            self._POS,
            velocities,
            self._HEAD,
            self._GOALS,
            self._PREF,
            np.ones(2, dtype=np.bool_),
            np.ones(2, dtype=np.bool_),  # both flagged in collision
            _make_state(2),
            cfg,
            dt=0.01,
            agent_radii=self._RADII,
            collision_velocities=collision_velocities,
        )
        return rewards

    def test_off_reproduces_binary_penalty(self):
        cfg = _collision_only_config(use_velocity_weighted_collision=False)
        # Even at high speed, OFF is the flat binary penalty.
        rewards = self._collide(cfg, np.array([[2.0, 0.0], [-2.0, 0.0]]))
        assert rewards[0] == pytest.approx(-2.0)
        assert rewards[1] == pytest.approx(-2.0)

    def test_scales_with_closing_speed(self):
        cfg = _collision_only_config(
            use_velocity_weighted_collision=True,
            collision_speed_floor=0.5,
            collision_speed_scale=0.5,
        )
        # Head-on, closing 4 m/s -> 0.5 + 0.5*4 = 2.5 -> -5.0
        fast = self._collide(cfg, np.array([[2.0, 0.0], [-2.0, 0.0]]))
        # Head-on, closing 0.2 m/s -> 0.5 + 0.5*0.2 = 0.6 -> -1.2
        slow = self._collide(cfg, np.array([[0.1, 0.0], [-0.1, 0.0]]))
        assert fast[0] == pytest.approx(-5.0)
        assert slow[0] == pytest.approx(-1.2)
        assert fast[0] < slow[0]

    def test_resting_contact_pays_only_floor(self):
        cfg = _collision_only_config(
            use_velocity_weighted_collision=True,
            collision_speed_floor=0.5,
            collision_speed_scale=0.5,
        )
        # Both at rest, closing 0 -> scale = floor 0.5 -> -1.0 (cheaper than -2).
        rewards = self._collide(cfg, np.zeros((2, 2)))
        assert rewards[0] == pytest.approx(-1.0)
        assert rewards[1] == pytest.approx(-1.0)

    def test_uses_relative_not_own_velocity(self):
        """Two agents moving FAST but together (zero relative velocity) are not
        closing, so they pay only the floor -- proving the weight is the pairwise
        CLOSING speed, not either agent's own speed."""
        cfg = _collision_only_config(
            use_velocity_weighted_collision=True,
            collision_speed_floor=0.5,
            collision_speed_scale=0.5,
        )
        # Both +x at 2 m/s: own speed 2, but relative velocity 0.
        rewards = self._collide(cfg, np.array([[2.0, 0.0], [2.0, 0.0]]))
        # Own-speed weighting would give 0.5 + 0.5*2 = 1.5 -> -3.0 (WRONG).
        # Closing-speed weighting gives floor 0.5 -> -1.0 (CORRECT).
        assert rewards[0] == pytest.approx(-1.0)
        assert rewards[0] != pytest.approx(-3.0)

    def test_falls_back_to_velocities_when_no_snapshot(self):
        cfg = _collision_only_config(
            use_velocity_weighted_collision=True,
            collision_speed_floor=0.5,
            collision_speed_scale=0.5,
        )
        v = np.array([[2.0, 0.0], [-2.0, 0.0]])
        with_snap = self._collide(cfg, np.zeros((2, 2)), collision_velocities=v)
        without_snap = self._collide(cfg, v, collision_velocities=None)
        assert with_snap[0] == pytest.approx(without_snap[0])
        assert with_snap[0] == pytest.approx(-5.0)

    def test_wall_contact_weighted_by_own_speed(self):
        # A wall is static, so "relative velocity" reduces to the agent's own
        # speed; ramming a wall at 2 m/s -> -0.5 * (0.5 + 0.5*2) = -0.75.
        cfg = _collision_only_config(
            collision_penalty=0.0,
            wall_collision_penalty=-0.5,
            use_velocity_weighted_collision=True,
            collision_speed_floor=0.5,
            collision_speed_scale=0.5,
        )
        rewards, _ = compute_rewards(
            np.array([[0.0, 0.0]]),
            np.array([[2.0, 0.0]]),
            np.zeros(1),
            np.array([[100.0, 0.0]]),
            np.ones(1) * 1.34,
            np.ones(1, dtype=np.bool_),
            np.zeros(1, dtype=np.bool_),  # no agent collision
            _make_state(1),
            cfg,
            dt=0.01,
            wall_collision_mask=np.ones(1, dtype=np.bool_),
        )
        assert rewards[0] == pytest.approx(-0.75)

    def test_wall_contact_off_is_binary(self):
        cfg = _collision_only_config(
            collision_penalty=0.0,
            wall_collision_penalty=-0.5,
            use_velocity_weighted_collision=False,
        )
        rewards, _ = compute_rewards(
            np.array([[0.0, 0.0]]),
            np.array([[2.0, 0.0]]),
            np.zeros(1),
            np.array([[100.0, 0.0]]),
            np.ones(1) * 1.34,
            np.ones(1, dtype=np.bool_),
            np.zeros(1, dtype=np.bool_),
            _make_state(1),
            cfg,
            dt=0.01,
            wall_collision_mask=np.ones(1, dtype=np.bool_),
        )
        assert rewards[0] == pytest.approx(-0.5)

    def test_nan_velocity_snapshot_yields_finite_reward(self):
        # A degenerate pileup / transient non-finite policy output can hand the
        # weighting a NaN/Inf pre-contact velocity; it must be sanitized, not
        # propagated into the reward (which would poison training -- the r855 bug).
        cfg = _collision_only_config(
            use_velocity_weighted_collision=True,
            collision_speed_floor=0.1,
            collision_speed_scale=0.5,
        )
        bad = np.array([[np.nan, 0.0], [np.inf, -np.inf]])
        rewards = self._collide(cfg, np.zeros((2, 2)), collision_velocities=bad)
        assert np.all(np.isfinite(rewards))
        # sanitized to 0 closing -> floor only -> -2 * 0.1 = -0.2
        assert rewards[0] == pytest.approx(-0.2)

    def test_huge_closing_speed_penalty_is_bounded(self):
        # A huge (finite) closing speed must not blow the penalty up -- the
        # impact speed is capped at max_impact_speed (10 m/s).
        cfg = _collision_only_config(
            use_velocity_weighted_collision=True,
            collision_speed_floor=0.1,
            collision_speed_scale=0.5,
        )
        huge = np.array([[1e6, 0.0], [-1e6, 0.0]])  # closing ~2e6 m/s
        rewards = self._collide(cfg, np.zeros((2, 2)), collision_velocities=huge)
        assert np.all(np.isfinite(rewards))
        # capped: -2 * (0.1 + 0.5*10) = -10.2
        assert rewards[0] == pytest.approx(-10.2)


class TestGradedWallProximity:
    """Graded ramp + closing-speed weighting of the wall-proximity band.

    One agent, radius 0.22, threshold 1.5 -> band [0.22, 0.33]. The nearest
    wall sits BELOW the agent, so ``wall_directions`` = (0, -1) and a velocity
    of (0, -v) approaches it at v m/s.
    """

    _RADII = np.array([0.22])
    _DOWN = np.array([[0.0, -1.0]])

    def _wall(self, cfg, wall_distance, velocity, wall_directions=None, collision_velocities=None):
        rewards, _ = compute_rewards(
            np.array([[0.0, 0.0]]),
            np.array([velocity]),
            np.zeros(1),
            np.array([[100.0, 0.0]]),  # far goal -> no bonus
            np.ones(1) * 1.34,
            np.ones(1, dtype=np.bool_),
            np.zeros(1, dtype=np.bool_),  # no agent collision
            _make_state(1),
            cfg,
            dt=0.01,
            wall_distances=np.array([wall_distance]),
            wall_directions=wall_directions,
            agent_radii=self._RADII,
            collision_velocities=collision_velocities,
        )
        return rewards[0]

    def test_flags_off_reproduce_flat_band(self):
        # near/far set to sentinel garbage: with the flag OFF they must be inert
        # and the flat band must apply byte-for-byte.
        cfg = _collision_only_config(
            collision_penalty=0.0,
            wall_proximity_penalty=-0.1,
            wall_proximity_penalty_near=-9.9,
            wall_proximity_penalty_far=-9.9,
        )
        assert self._wall(cfg, 0.25, [0.0, -2.0]) == pytest.approx(-0.1)
        assert self._wall(cfg, 0.5, [0.0, -2.0]) == pytest.approx(0.0)

    def test_graded_ramp_values(self):
        cfg = _collision_only_config(
            collision_penalty=0.0,
            use_graded_wall_proximity=True,
            wall_proximity_penalty_near=-0.2,
            wall_proximity_penalty_far=0.0,
        )
        v = [0.0, 0.0]
        # At body contact (d == radius): full near penalty.
        assert self._wall(cfg, 0.22, v) == pytest.approx(-0.2)
        # Mid-band: mean of near and far.
        assert self._wall(cfg, 0.275, v) == pytest.approx(-0.1)
        # Near the outer edge: ramp approaches far (0.0) -> continuous onset.
        assert self._wall(cfg, 0.329, v) == pytest.approx(-0.2 * (1 - 0.109 / 0.11))
        # Outside the band: nothing.
        assert self._wall(cfg, 0.34, v) == pytest.approx(0.0)
        # Below radius (transient penetration): clamps to near.
        assert self._wall(cfg, 0.1, v) == pytest.approx(-0.2)

    def test_closing_speed_weighting_floor_zero(self):
        cfg = _collision_only_config(
            collision_penalty=0.0,
            use_graded_wall_proximity=True,
            wall_proximity_penalty_near=-0.2,
            wall_proximity_penalty_far=0.0,
            use_velocity_weighted_wall_proximity=True,
            wall_proximity_speed_floor=0.0,
            wall_proximity_speed_scale=0.5,
        )
        # Approaching at 1 m/s, mid-band: -0.1 * (0 + 0.5*1) = -0.05
        assert self._wall(cfg, 0.275, [0.0, -1.0], self._DOWN) == pytest.approx(-0.05)
        # Standing beside the wall: FREE (this is the yield state).
        assert self._wall(cfg, 0.275, [0.0, 0.0], self._DOWN) == pytest.approx(0.0)
        # Moving parallel to the wall: zero closing -> free.
        assert self._wall(cfg, 0.275, [2.0, 0.0], self._DOWN) == pytest.approx(0.0)
        # Receding: negative closing clamps the weight to 0 -> free.
        assert self._wall(cfg, 0.275, [0.0, 2.0], self._DOWN) == pytest.approx(0.0)

    def test_floor_charges_at_zero_closing(self):
        cfg = _collision_only_config(
            collision_penalty=0.0,
            use_graded_wall_proximity=True,
            wall_proximity_penalty_near=-0.2,
            wall_proximity_penalty_far=0.0,
            use_velocity_weighted_wall_proximity=True,
            wall_proximity_speed_floor=0.25,
            wall_proximity_speed_scale=0.5,
        )
        # Standing mid-band with a non-zero floor: -0.1 * 0.25
        assert self._wall(cfg, 0.275, [0.0, 0.0], self._DOWN) == pytest.approx(-0.025)

    def test_weighting_skipped_without_directions(self):
        cfg = _collision_only_config(
            collision_penalty=0.0,
            use_graded_wall_proximity=True,
            wall_proximity_penalty_near=-0.2,
            wall_proximity_penalty_far=0.0,
            use_velocity_weighted_wall_proximity=True,
            wall_proximity_speed_floor=0.0,
            wall_proximity_speed_scale=0.5,
        )
        # No wall_directions -> unweighted ramp, even at speed.
        assert self._wall(cfg, 0.275, [0.0, -2.0], None) == pytest.approx(-0.1)

    def test_flat_band_composes_with_weighting(self):
        # The two flags are orthogonal: weighting also applies to the flat band.
        cfg = _collision_only_config(
            collision_penalty=0.0,
            wall_proximity_penalty=-0.1,
            use_velocity_weighted_wall_proximity=True,
            wall_proximity_speed_floor=0.0,
            wall_proximity_speed_scale=0.5,
        )
        assert self._wall(cfg, 0.25, [0.0, -1.0], self._DOWN) == pytest.approx(-0.05)
        assert self._wall(cfg, 0.25, [0.0, 0.0], self._DOWN) == pytest.approx(0.0)

    def test_weighting_uses_precontact_snapshot(self):
        cfg = _collision_only_config(
            collision_penalty=0.0,
            use_graded_wall_proximity=True,
            wall_proximity_penalty_near=-0.2,
            wall_proximity_penalty_far=0.0,
            use_velocity_weighted_wall_proximity=True,
            wall_proximity_speed_floor=0.0,
            wall_proximity_speed_scale=0.5,
        )
        # Post-contact velocity says "approaching fast" but the pre-contact
        # snapshot says "standing": the snapshot must win -> free.
        r = self._wall(
            cfg,
            0.275,
            [0.0, -2.0],
            self._DOWN,
            collision_velocities=np.zeros((1, 2)),
        )
        assert r == pytest.approx(0.0)


class TestWallNormalImpact:
    """Wall-contact impact speed = into-wall velocity component (opt-in)."""

    _DOWN = np.array([[0.0, -1.0]])

    def _contact(self, cfg, velocity, wall_directions=None):
        rewards, _ = compute_rewards(
            np.array([[0.0, 0.0]]),
            np.array([velocity]),
            np.zeros(1),
            np.array([[100.0, 0.0]]),
            np.ones(1) * 1.34,
            np.ones(1, dtype=np.bool_),
            np.zeros(1, dtype=np.bool_),
            _make_state(1),
            cfg,
            dt=0.01,
            wall_collision_mask=np.ones(1, dtype=np.bool_),
            wall_directions=wall_directions,
        )
        return rewards[0]

    def _cfg(self, **overrides):
        base = dict(
            collision_penalty=0.0,
            wall_collision_penalty=-2.0,
            use_velocity_weighted_collision=True,
            collision_speed_floor=0.25,
            collision_speed_scale=0.5,
            use_wall_normal_impact=True,
        )
        base.update(overrides)
        return _collision_only_config(**base)

    def test_head_on_pays_full_impact(self):
        # Into-wall component 2 m/s -> -2 * (0.25 + 0.5*2) = -2.5
        assert self._contact(self._cfg(), [0.0, -2.0], self._DOWN) == pytest.approx(-2.5)

    def test_sliding_parallel_pays_only_floor(self):
        # Parallel slide at 2 m/s: zero into-wall component -> -2 * 0.25 = -0.5
        assert self._contact(self._cfg(), [2.0, 0.0], self._DOWN) == pytest.approx(-0.5)

    def test_diagonal_pays_normal_component(self):
        # 45 deg into the wall at |v| = 2: component sqrt(2)
        expected = -2.0 * (0.25 + 0.5 * np.sqrt(2.0))
        r = self._contact(self._cfg(), [np.sqrt(2.0), -np.sqrt(2.0)], self._DOWN)
        assert r == pytest.approx(expected)

    def test_receding_component_clamps_to_floor(self):
        # Moving away from the wall while still flagged: impact clamps to 0.
        assert self._contact(self._cfg(), [0.0, 2.0], self._DOWN) == pytest.approx(-0.5)

    def test_flag_off_uses_full_speed(self):
        # Legacy weighting: sliding parallel at 2 m/s costs the same as head-on.
        cfg = self._cfg(use_wall_normal_impact=False)
        assert self._contact(cfg, [2.0, 0.0], self._DOWN) == pytest.approx(-2.5)

    def test_falls_back_to_full_speed_without_directions(self):
        assert self._contact(self._cfg(), [2.0, 0.0], None) == pytest.approx(-2.5)


class TestWallCollisionPenaltyCap:
    """Wall-side twin of collision_penalty_cap: weighting may DISCOUNT slow
    contact but never AMPLIFY fast contact below the cap. Same harness as
    TestWallNormalImpact (base -2.0, floor 0.25, scale 0.5, wall below)."""

    _DOWN = TestWallNormalImpact._DOWN
    _contact = TestWallNormalImpact._contact
    _cfg = TestWallNormalImpact._cfg

    def test_cap_clamps_amplified_head_on(self):
        # Head-on 2 m/s: uncapped -2.0 * (0.25 + 1.0) = -2.5 -> clamped to -2.0.
        cfg = self._cfg(wall_collision_penalty_cap=-2.0)
        assert self._contact(cfg, [0.0, -2.0], self._DOWN) == pytest.approx(-2.0)

    def test_cap_leaves_discounted_contact_alone(self):
        # Parallel slide pays the floor (-0.5), far above the cap -> untouched.
        cfg = self._cfg(wall_collision_penalty_cap=-2.0)
        assert self._contact(cfg, [2.0, 0.0], self._DOWN) == pytest.approx(-0.5)

    def test_cap_off_preserves_amplification(self):
        # cap 0.0 (the default) -> legacy behaviour, amplified -2.5 stands.
        cfg = self._cfg(wall_collision_penalty_cap=0.0)
        assert self._contact(cfg, [0.0, -2.0], self._DOWN) == pytest.approx(-2.5)

    def test_cap_applies_to_full_speed_weighting_too(self):
        # Independent of use_wall_normal_impact: full-||v|| weighting is capped
        # the same way (parallel slide at 2 m/s -> -2.5 uncapped -> -2.0).
        cfg = self._cfg(use_wall_normal_impact=False, wall_collision_penalty_cap=-2.0)
        assert self._contact(cfg, [2.0, 0.0], self._DOWN) == pytest.approx(-2.0)
