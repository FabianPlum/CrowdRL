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
        assert state.prev_goal_distances is not None
        assert len(state.prev_goal_distances) == 5
