"""Tests for the path-distance metric driving the progress reward + stuck check.

The metric measures remaining navmesh PATH length, not straight-line distance to
the goal, so progress stays positive when an agent correctly follows a route that
bends away from the goal -- exactly where the bee-line metric goes negative and
falsely flags the agent as stuck.
"""

import numpy as np
import torch

from crowdrl_torch.observation import compute_navmesh_signals
from crowdrl_torch.reward import compute_remaining_path, path_distance_metric
from crowdrl_torch.types import EnvConfig


def _t(x):
    return torch.tensor(x, dtype=torch.float32)


def test_remaining_path_is_dist_to_cursor_plus_cumulative():
    positions = _t([[[0.0, 0.0]]])  # (1, 1, 2)
    waypoints = _t([[[[5.0, 0.0], [0.0, 10.0]]]])  # (1, 1, 2, 2)
    seg = float(np.hypot(5.0, 10.0))  # |wp0 -> wp1|
    wpl = _t([[[seg, 0.0]]])  # cumulative remaining from each waypoint
    n_wp = torch.tensor([[2]], dtype=torch.int32)
    cursor = torch.tensor([[0]], dtype=torch.int32)

    rem = compute_remaining_path(positions, waypoints, cursor, n_wp, wpl)
    # dist(origin, wp0) = 5, plus cumulative_from_wp0 = seg
    assert abs(float(rem[0, 0]) - (5.0 + seg)) < 1e-5


def test_path_progress_positive_while_goal_distance_increases():
    # Route forces the agent right (toward wp0 = (5, 0)) before doubling back up
    # to goal = (0, 10). Advancing toward wp0 INCREASES straight-line goal
    # distance but DECREASES remaining path: path-aware progress is positive,
    # bee-line progress would be negative.
    goal = _t([[[0.0, 10.0]]])
    waypoints = _t([[[[5.0, 0.0], [0.0, 10.0]]]])
    seg = float(np.hypot(5.0, 10.0))
    wpl = _t([[[seg, 0.0]]])
    n_wp = torch.tensor([[2]], dtype=torch.int32)
    cursor = torch.tensor([[0]], dtype=torch.int32)

    pos_start = _t([[[0.0, 0.0]]])
    pos_moved = _t([[[2.5, 0.0]]])  # advanced toward wp0, away from the goal

    nav_start = path_distance_metric(pos_start, goal, waypoints, cursor, n_wp, wpl, True)
    nav_moved = path_distance_metric(pos_moved, goal, waypoints, cursor, n_wp, wpl, True)
    gd_start = float((goal - pos_start).norm(dim=-1))
    gd_moved = float((goal - pos_moved).norm(dim=-1))

    assert float(nav_moved) < float(nav_start)  # path progress is POSITIVE
    assert gd_moved > gd_start  # bee-line "progress" would be NEGATIVE


def test_fallback_to_goal_distance():
    positions = _t([[[1.0, 2.0]]])
    goal = _t([[[4.0, 6.0]]])  # straight-line distance 5
    waypoints = torch.zeros((1, 1, 4, 2), dtype=torch.float32)
    wpl = torch.zeros((1, 1, 4), dtype=torch.float32)
    cursor = torch.zeros((1, 1), dtype=torch.int32)

    # navmesh disabled -> straight-line goal distance
    n_wp = torch.tensor([[2]], dtype=torch.int32)
    off = path_distance_metric(positions, goal, waypoints, cursor, n_wp, wpl, False)
    assert abs(float(off) - 5.0) < 1e-5

    # navmesh on but the agent has no path -> straight-line fallback
    n_wp0 = torch.tensor([[0]], dtype=torch.int32)
    none = path_distance_metric(positions, goal, waypoints, cursor, n_wp0, wpl, True)
    assert abs(float(none) - 5.0) < 1e-5


def test_navmesh_signal_points_at_single_waypoint_no_blend():
    # The bearing must point at the SINGLE current waypoint (cursor 0), with no
    # distance-weighted look-ahead blend toward the next waypoint. wp0 is due
    # north and wp1 is north-east; a blend would tilt the bearing eastward.
    cfg = EnvConfig(max_waypoints=4, use_navmesh=True)
    positions = _t([[[0.0, 0.0]]])
    cos_h = _t([[1.0]])  # torso 0 -> ego frame == world frame
    sin_h = _t([[0.0]])
    waypoints = _t([[[[0.0, 5.0], [5.0, 5.0], [0.0, 0.0], [0.0, 0.0]]]])
    n_wp = torch.tensor([[2]], dtype=torch.int32)
    cursor = torch.tensor([[0]], dtype=torch.int32)
    wpl = _t([[[5.0, 0.0, 0.0, 0.0]]])  # |wp0->wp1| = 5, then 0
    goal = _t([[[0.0, 5.0]]])

    nav = compute_navmesh_signals(positions, cos_h, sin_h, waypoints, n_wp, cursor, wpl, goal, cfg)
    # Points straight at wp0 = (0, 1): no eastward (x) component a blend would add.
    assert abs(float(nav[0, 0, 0]) - 0.0) < 1e-5  # dir_ego_x
    assert abs(float(nav[0, 0, 1]) - 1.0) < 1e-5  # dir_ego_y
