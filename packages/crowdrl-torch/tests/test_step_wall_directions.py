"""batched_step wiring for the wall-shaping reward terms.

The graded / closing-speed-weighted wall-proximity penalty needs the nearest-
wall DIRECTION, which batched_step only materialises when the flags ask for it.
This test drives the full step path (not compute_rewards directly) so a config
knob that fails to reach the step -- the historical silent-no-op failure mode --
turns the build red.
"""

from __future__ import annotations

import numpy.testing as npt
import torch

from crowdrl_torch.reward import REWARD_COMPONENT_NAMES
from crowdrl_torch.step import batched_step
from crowdrl_torch.types import EnvConfig, make_initial_state

_WP = REWARD_COMPONENT_NAMES.index("wall_proximity")
_WC = REWARD_COMPONENT_NAMES.index("wall_collision")


def _room_state(n_active: int = 1):
    """Agents inside a 10x10 room, radius 0.25 -> band [0.25, 0.375].

    A closed polygon (not a bare segment) so ``enforce_wall_boundaries``'s
    even-odd inside test is meaningful and the agent is left untouched.
    """
    state = make_initial_state(
        n_envs=1,
        max_agents=2,
        max_segments=4,
        max_waypoints=4,
        memory_window=5,
        k_neighbours=2,
        neighbor_vel_history_window=2,
        device="cpu",
    )
    state.positions[0, 0] = torch.tensor([0.35, 2.0])
    state.goal_positions[0, 0] = torch.tensor([9.5, 2.0])
    state.shoulder_widths[:] = 0.25  # radius 0.25 -> band [0.25, 0.375] at threshold 1.5
    state.chest_depths[:] = 0.15
    state.preferred_speeds[:] = 1.3
    # Huge mass -> the exponential wall-repulsion impulse is negligible and the
    # post-step position is exactly start + v*dt (hand-computable).
    state.masses[:] = 1e12
    state.active_mask[0] = torch.tensor([True, n_active == 2])
    state.n_agents[:] = n_active
    state.wall_segments[0] = torch.tensor(
        [
            [[0.0, 0.0], [10.0, 0.0]],
            [[10.0, 0.0], [10.0, 10.0]],
            [[10.0, 10.0], [0.0, 10.0]],
            [[0.0, 10.0], [0.0, 0.0]],
        ]
    )
    state.n_segments[:] = 4
    return state


def _wall_state():
    """One active agent, 0.35 m from the left wall (x=0)."""
    return _room_state(n_active=1)


def _config(**overrides):
    base = dict(
        max_agents=2,
        max_segments=4,
        max_waypoints=4,
        n_rays=8,
        k_neighbours=2,
        use_navmesh=False,
        use_temporal_memory=False,
        use_neighbor_memory=False,
        temporal_memory_window=5,
        neighbor_vel_history_window=2,
        # v_new = v_desired exactly, so the commanded approach speed is the
        # closing speed the reward sees.
        desired_velocity_weight=1.0,
        progress_weight=0.0,
    )
    base.update(overrides)
    return EnvConfig(**base)


def test_step_wires_wall_directions_into_reward():
    """The wall-shaping knobs must reach the reward through the full step."""
    # Full-backward speed action: desired velocity (-0.5, 0) -> approaches the
    # wall at 0.5 m/s; after the step the agent sits at x = 0.345, inside the
    # band [0.25, 0.375]: t = (0.345 - 0.25) / 0.125 = 0.76 -> ramp
    # -0.2 * 0.24 = -0.048; closing weight 0 + 0.5*0.5 = 0.25 -> -0.012.
    actions = torch.zeros((1, 2, 4))
    actions[0, :, 0] = -1.0

    shaped = _config(
        use_graded_wall_proximity=True,
        wall_proximity_penalty_near=-0.2,
        wall_proximity_penalty_far=0.0,
        use_velocity_weighted_wall_proximity=True,
        wall_proximity_speed_floor=0.0,
        wall_proximity_speed_scale=0.5,
    )
    _, _, _, _, _, comps = batched_step(_wall_state(), actions, shaped)
    npt.assert_allclose(comps[0, 0, _WP].item(), -0.012, atol=1e-4)

    # Same scene with the flags OFF: the legacy flat band charges -0.1.
    _, _, _, _, _, comps_flat = batched_step(_wall_state(), actions, _config())
    npt.assert_allclose(comps_flat[0, 0, _WP].item(), -0.1, atol=1e-6)


def test_step_standing_beside_wall_is_free_with_floor_zero():
    """The yield state end-to-end: an agent parked inside the band with zero
    commanded velocity pays nothing at floor 0.0 (it paid -0.1/step before)."""
    state = _wall_state()
    state.positions[0, 0] = torch.tensor([0.3, 2.0])  # already mid-band
    # a0 = -0.6 maps to desired speed 0 -> standing.
    actions = torch.zeros((1, 2, 4))
    actions[0, :, 0] = -0.6

    shaped = _config(
        use_graded_wall_proximity=True,
        wall_proximity_penalty_near=-0.2,
        wall_proximity_penalty_far=0.0,
        use_velocity_weighted_wall_proximity=True,
        wall_proximity_speed_floor=0.0,
        wall_proximity_speed_scale=0.5,
    )
    _, _, _, _, _, comps = batched_step(state, actions, shaped)
    npt.assert_allclose(comps[0, 0, _WP].item(), 0.0, atol=1e-6)


def test_step_wires_wall_directions_for_normal_impact_alone():
    """The OTHER half of the ``need_wall_directions`` predicate.

    ``use_wall_normal_impact`` reaches the reward through
    ``use_velocity_weighted_collision``, with the proximity weighting OFF. Two
    agents in wall contact at the SAME speed (0.5 m/s) but different geometry:
    agent 0 drives into the left wall, agent 1 slides along the bottom wall. If
    the directions were not materialised the impact would fall back to full
    speed and both would pay the same -- so the two values differing is what
    proves the wiring.
    """

    def contact_state():
        state = _room_state(n_active=2)
        state.positions[0, 0] = torch.tensor([0.24, 2.0])  # penetrating the left wall
        state.positions[0, 1] = torch.tensor([2.0, 0.24])  # penetrating the bottom wall
        state.goal_positions[0, 1] = torch.tensor([9.5, 0.24])
        return state

    contact_cfg = dict(
        wall_collision_penalty=-2.0,
        use_velocity_weighted_collision=True,
        collision_speed_floor=0.25,
        collision_speed_scale=0.5,
    )
    # Both back up at 0.5 m/s along -x: into the left wall, parallel to the bottom.
    actions = torch.zeros((1, 2, 4))
    actions[0, :, 0] = -1.0

    _, _, _, _, _, comps = batched_step(
        contact_state(),
        actions,
        _config(use_wall_normal_impact=True, **contact_cfg),
    )
    # Head-on: -2.0 * (0.25 + 0.5 * 0.5) = -1.0. Parallel: -2.0 * 0.25 = -0.5.
    npt.assert_allclose(comps[0, 0, _WC].item(), -1.0, atol=1e-5)
    npt.assert_allclose(comps[0, 1, _WC].item(), -0.5, atol=1e-5)

    # Same scene without the normal: full speed weights both alike at -1.0.
    _, _, _, _, _, flat = batched_step(
        contact_state(),
        actions,
        _config(use_wall_normal_impact=False, **contact_cfg),
    )
    npt.assert_allclose(flat[0, 0, _WC].item(), -1.0, atol=1e-5)
    npt.assert_allclose(flat[0, 1, _WC].item(), -1.0, atol=1e-5)
