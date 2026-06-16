"""Waypoint cursor advancement (H2): robust to agents pushed off-route.

The old rule advanced only within ``waypoint_crossing_threshold`` of the current
waypoint, so an agent that cut a corner wide (the common dense-crowd case) never
advanced -> stuck cursor -> negative progress reward + false stuck-termination.
The hardened rule also advances once the agent is closer to the NEXT waypoint.
"""

from __future__ import annotations

import torch

from crowdrl_torch.step import advance_waypoint_cursor
from crowdrl_torch.types import EnvConfig

# Straight path of waypoints along +x: wp0=(2,0) wp1=(4,0) wp2=(6,0); slot 3 unused.
_WAYPOINTS = torch.tensor([[[[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [0.0, 0.0]]]])


def _advance(pos, cursor, n_wp=3, active=True, threshold=0.5):
    positions = torch.tensor([[pos]], dtype=torch.float32)
    n_waypoints = torch.tensor([[n_wp]], dtype=torch.int32)
    cur = torch.tensor([[cursor]], dtype=torch.int32)
    active_mask = torch.tensor([[active]], dtype=torch.bool)
    config = EnvConfig(max_waypoints=4, waypoint_crossing_threshold=threshold)
    new = advance_waypoint_cursor(positions, _WAYPOINTS, n_waypoints, cur, active_mask, config)
    return int(new[0, 0].item())


def test_reached_threshold_advances():
    # Within 0.5 m of wp0 -> advance.
    assert _advance([2.0, 0.3], cursor=0) == 1


def test_corner_cut_advances_even_outside_threshold():
    # 2.1 m from wp0 (would NOT trip the 0.5 m disk) but closer to wp1 -> advance.
    # This is the case the old proximity-only cursor got stuck on.
    assert _advance([3.5, 1.5], cursor=0) == 1


def test_before_waypoint_does_not_advance():
    # Still approaching wp0 (closer to wp0 than wp1, outside threshold) -> hold.
    assert _advance([1.0, 0.0], cursor=0) == 0


def test_advance_is_single_step_monotonic():
    # Even far down-path (closer to wp2) the cursor moves by exactly one per step.
    assert _advance([5.5, 0.0], cursor=0) == 1


def test_holds_at_final_waypoint():
    # At the last waypoint (the goal), reaching it must not overflow the cursor.
    assert _advance([6.0, 0.1], cursor=2, n_wp=3) == 2


def test_inactive_agent_does_not_advance():
    assert _advance([2.0, 0.1], cursor=0, active=False) == 0
