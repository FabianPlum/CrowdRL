"""Tests for the per-component reward decomposition (collapse instrumentation).

``compute_rewards`` returns a ``(E, N, C)`` breakdown whose channels must sum
exactly (within float tolerance) to the total per-agent reward, so the
instrumentation never silently drops or double-counts a reward mode. The
``timeout`` channel is filled by ``batched_step``, not here, so it stays zero
in these direct-call tests.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import torch

from crowdrl_torch.reward import (
    REWARD_COMPONENT_NAMES,
    TIMEOUT_COMPONENT_IDX,
    compute_rewards,
)
from crowdrl_torch.types import EnvConfig


def _idx(name: str) -> int:
    return REWARD_COMPONENT_NAMES.index(name)


def test_component_names_and_timeout_index():
    """timeout is the last channel; the index constant agrees with the tuple."""
    assert REWARD_COMPONENT_NAMES[-1] == "timeout"
    assert TIMEOUT_COMPONENT_IDX == len(REWARD_COMPONENT_NAMES) - 1
    # No duplicate channels.
    assert len(set(REWARD_COMPONENT_NAMES)) == len(REWARD_COMPONENT_NAMES)


def test_components_sum_to_total_all_active():
    """With every optional input supplied, all reward modes are exercised and
    the breakdown sums to the total reward."""
    n = 6
    # Agent 0 sits on its goal (goal bonus); agent 1 is flagged in collision;
    # agents 2 and 3 are within personal space (proximity ramp); agent 4 hugs a
    # wall (wall proximity). Velocities/actions differ from their previous values
    # so action-rate and smoothness channels are non-trivial.
    positions = np.array(
        [
            [5.0, 5.0],
            [1.0, 1.0],
            [3.0, 3.0],
            [3.5, 3.0],  # 0.5 m from agent 2 -> inside personal_space_radius
            [8.0, 8.0],
            [6.0, 2.0],
        ],
        dtype=np.float32,
    )
    velocities = np.array(
        [[0.1, 0.0], [0.5, -0.2], [-0.3, 0.4], [0.2, 0.2], [1.0, 0.0], [-0.5, 0.5]],
        dtype=np.float32,
    )
    goal_positions = positions.copy()
    goal_positions[0] = positions[0] + [0.1, 0.1]  # agent 0 essentially at goal
    goal_positions[1:] += [2.0, 1.5]  # others have distance to cover

    active_mask = np.ones(n, dtype=np.bool_)
    collision_mask = np.zeros(n, dtype=np.bool_)
    collision_mask[1] = True

    prev_goal_distances = np.linalg.norm(
        goal_positions - (positions - velocities * 0.1), axis=1
    ).astype(np.float32)

    agent_radii = np.full(n, 0.25, dtype=np.float32)
    wall_distances = np.full(n, 5.0, dtype=np.float32)
    wall_distances[4] = 0.2  # < 0.25 * 1.5 threshold -> wall proximity penalty

    actions = np.zeros((n, 4), dtype=np.float32)
    prev_actions = np.full((n, 4), 0.3, dtype=np.float32)  # != actions -> action rate
    headings = np.full(n, 0.2, dtype=np.float32)
    preferred_speeds = np.full(n, 1.2, dtype=np.float32)
    prev_velocities = np.zeros((n, 2), dtype=np.float32)  # != velocities -> jerk/accel
    prev_accelerations = np.zeros((n, 2), dtype=np.float32)
    prev_headings = np.full(n, 0.1, dtype=np.float32)
    prev_heading_changes = np.zeros(n, dtype=np.float32)

    config = EnvConfig(max_agents=n, use_smoothness=True)

    def t(x):
        return torch.tensor(x).unsqueeze(0)

    rewards, _reached, _dists, comps = compute_rewards(
        t(positions),
        t(velocities),
        t(goal_positions),
        t(active_mask),
        t(collision_mask),
        t(prev_goal_distances),
        config,
        wall_distances=t(wall_distances),
        agent_radii=t(agent_radii),
        actions=t(actions),
        prev_actions=t(prev_actions),
        headings=t(headings),
        preferred_speeds=t(preferred_speeds),
        prev_velocities=t(prev_velocities),
        prev_accelerations=t(prev_accelerations),
        prev_headings=t(prev_headings),
        prev_heading_changes=t(prev_heading_changes),
    )

    rewards = rewards[0].numpy()
    comps = comps[0].numpy()

    # Exhaustive decomposition: channels sum to the total reward.
    assert comps.shape == (n, len(REWARD_COMPONENT_NAMES))
    npt.assert_allclose(comps.sum(axis=-1), rewards, atol=1e-4, rtol=1e-4)

    # Each mode lands in the expected channel (sign / magnitude sanity).
    npt.assert_allclose(comps[0, _idx("goal")], config.goal_bonus, atol=1e-4)
    npt.assert_allclose(comps[1, _idx("collision_agent")], config.collision_penalty, atol=1e-4)
    npt.assert_allclose(comps[4, _idx("wall_proximity")], config.wall_proximity_penalty, atol=1e-4)
    npt.assert_allclose(
        comps[:, _idx("existence")], np.full(n, config.existence_penalty), atol=1e-4
    )
    # Agents 2 and 3 are inside personal space -> negative proximity contribution.
    assert comps[2, _idx("agent_proximity")] < 0.0
    assert comps[3, _idx("agent_proximity")] < 0.0
    # Action-rate and smoothness are active for everyone (non-zero deltas).
    assert np.all(comps[:, _idx("action_rate")] < 0.0)
    assert np.all(comps[:, _idx("smoothness")] < 0.0)
    # timeout is filled by batched_step, not compute_rewards.
    npt.assert_array_equal(comps[:, TIMEOUT_COMPONENT_IDX], np.zeros(n))


def test_inactive_agents_have_zero_components():
    """Inactive agents contribute zero in every channel."""
    n = 4
    positions = np.random.default_rng(0).uniform(1.0, 9.0, (n, 2)).astype(np.float32)
    velocities = np.zeros((n, 2), dtype=np.float32)
    goal_positions = positions + 1.0
    active_mask = np.array([True, False, True, False], dtype=np.bool_)
    collision_mask = np.zeros(n, dtype=np.bool_)
    prev_goal_distances = np.linalg.norm(goal_positions - positions, axis=1).astype(np.float32)

    config = EnvConfig(max_agents=n)

    def t(x):
        return torch.tensor(x).unsqueeze(0)

    rewards, _reached, _dists, comps = compute_rewards(
        t(positions),
        t(velocities),
        t(goal_positions),
        t(active_mask),
        t(collision_mask),
        t(prev_goal_distances),
        config,
    )
    comps = comps[0].numpy()
    npt.assert_array_equal(comps[~active_mask], np.zeros((2, len(REWARD_COMPONENT_NAMES))))
    npt.assert_allclose(comps.sum(axis=-1), rewards[0].numpy(), atol=1e-4, rtol=1e-4)
