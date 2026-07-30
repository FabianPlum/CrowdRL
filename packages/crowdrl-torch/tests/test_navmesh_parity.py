"""numpy <-> GPU navmesh observation parity (guards H1: single-waypoint signal).

The torch obs builder reads a static precomputed funnel path + cursor; the numpy
builder recomputes ``next_waypoint_direction`` every step. At an agent's spawn
(cursor 0, on the route) they must produce the SAME ego-frame next-waypoint
direction and path deviation. If the GPU distance-weighted blend is ever
reintroduced, this test fails.

(Off-route the two intentionally differ -- numpy recomputes from the new
position, the GPU follows the stored path -- so parity is asserted at spawn.)
"""

from __future__ import annotations

import numpy as np
import torch
from shapely.geometry import Polygon

from crowdrl_core.geometry import build_navmesh
from crowdrl_core.navmesh import (
    JUPEDSIM_ROUTER_INSET,
    next_waypoint_direction,
    path_deviation,
    router_next_waypoint,
    shortest_path,
)
from crowdrl_torch.observation import compute_navmesh_signals
from crowdrl_torch.types import EnvConfig


def _l_corridor() -> Polygon:
    # L-shaped walkable area: a straight bee-line from the horizontal arm to the
    # vertical arm leaves the polygon, so the shortest path must bend around the
    # inner corner (2, 2) -> a non-trivial intermediate waypoint.
    return Polygon([(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6)])


def test_navmesh_signal_matches_numpy_at_spawn():
    navmesh = build_navmesh(_l_corridor())
    pos = np.array([5.0, 1.0])
    goal = np.array([1.0, 5.0])
    radius = 0.3
    heading = 0.7  # arbitrary torso angle, exercises the ego rotation

    wps = shortest_path(navmesh, pos, goal, radius)
    assert wps is not None and len(wps) >= 3, "need a bent path for a meaningful test"

    # --- numpy reference: next-waypoint dir + path deviation, rotated to ego ---
    wp_dir = next_waypoint_direction(navmesh, pos, goal, radius)
    p_dev = path_deviation(navmesh, pos, goal, radius)
    cos_h, sin_h = np.cos(-heading), np.sin(-heading)
    np_x = cos_h * wp_dir[0] - sin_h * wp_dir[1]
    np_y = sin_h * wp_dir[0] + cos_h * wp_dir[1]

    # --- torch side: replicate the episode_factory precompute at cursor 0 ---
    stored = [np.asarray(w) for w in wps[1:]]  # drop start; intermediate + goal
    n = len(stored)
    max_wp = 8
    wp_arr = np.zeros((max_wp, 2), dtype=np.float64)
    wp_len = np.zeros(max_wp, dtype=np.float64)
    for k in range(n):
        wp_arr[k] = stored[k]
    for k in range(n - 1, -1, -1):
        wp_len[k] = (
            0.0 if k == n - 1 else wp_len[k + 1] + np.linalg.norm(stored[k + 1] - stored[k])
        )

    waypoints = torch.tensor(wp_arr, dtype=torch.float32).view(1, 1, max_wp, 2)
    n_waypoints = torch.tensor([[n]], dtype=torch.int32)
    cursor = torch.zeros((1, 1), dtype=torch.int32)
    path_lengths = torch.tensor(wp_len, dtype=torch.float32).view(1, 1, max_wp)
    positions = torch.tensor(pos, dtype=torch.float32).view(1, 1, 2)
    goals = torch.tensor(goal, dtype=torch.float32).view(1, 1, 2)
    cos_t = torch.cos(torch.tensor(-heading, dtype=torch.float32)).view(1, 1)
    sin_t = torch.sin(torch.tensor(-heading, dtype=torch.float32)).view(1, 1)
    config = EnvConfig(max_waypoints=max_wp, use_navmesh=True)

    nav = compute_navmesh_signals(
        positions, cos_t, sin_t, waypoints, n_waypoints, cursor, path_lengths, goals, config
    )
    tx, ty, tdev = nav[0, 0].tolist()

    assert np.allclose([tx, ty], [np_x, np_y], atol=1e-3), (
        f"dir mismatch {(tx, ty)} vs {(np_x, np_y)}"
    )
    assert np.allclose(tdev, max(p_dev, 0.0), atol=1e-3), f"path_dev mismatch {tdev} vs {p_dev}"


def test_jupedsim_style_routing_matches_numpy_at_spawn():
    """Flag-on parity: the stored path is precomputed at the router's fixed
    0.2 m inset and the path_dev channel is pinned to 0.0, matching the numpy
    flag-on builder (route-branch math on router_next_waypoint)."""
    navmesh = build_navmesh(_l_corridor())
    pos = np.array([5.0, 1.0])
    goal = np.array([1.0, 5.0])
    heading = 0.7

    # --- numpy reference: the flag-on funnel branch ---
    wp = router_next_waypoint(navmesh, pos, goal)
    assert wp is not None
    unit = (wp - pos) / np.linalg.norm(wp - pos)
    cos_h, sin_h = np.cos(-heading), np.sin(-heading)
    np_x = cos_h * unit[0] - sin_h * unit[1]
    np_y = sin_h * unit[0] + cos_h * unit[1]

    # --- torch side: flag-on episode-factory precompute (inset 0.2, cursor 0) ---
    wps = shortest_path(navmesh, pos, goal, JUPEDSIM_ROUTER_INSET)
    assert wps is not None and len(wps) >= 3, "need a bent path for a meaningful test"
    stored = [np.asarray(w) for w in wps[1:]]
    n = len(stored)
    max_wp = 8
    wp_arr = np.zeros((max_wp, 2), dtype=np.float64)
    wp_len = np.zeros(max_wp, dtype=np.float64)
    for k in range(n):
        wp_arr[k] = stored[k]
    for k in range(n - 1, -1, -1):
        wp_len[k] = (
            0.0 if k == n - 1 else wp_len[k + 1] + np.linalg.norm(stored[k + 1] - stored[k])
        )

    waypoints = torch.tensor(wp_arr, dtype=torch.float32).view(1, 1, max_wp, 2)
    n_waypoints = torch.tensor([[n]], dtype=torch.int32)
    cursor = torch.zeros((1, 1), dtype=torch.int32)
    path_lengths = torch.tensor(wp_len, dtype=torch.float32).view(1, 1, max_wp)
    positions = torch.tensor(pos, dtype=torch.float32).view(1, 1, 2)
    goals = torch.tensor(goal, dtype=torch.float32).view(1, 1, 2)
    cos_t = torch.cos(torch.tensor(-heading, dtype=torch.float32)).view(1, 1)
    sin_t = torch.sin(torch.tensor(-heading, dtype=torch.float32)).view(1, 1)
    config = EnvConfig(max_waypoints=max_wp, use_navmesh=True, use_jupedsim_style_routing=True)

    nav = compute_navmesh_signals(
        positions, cos_t, sin_t, waypoints, n_waypoints, cursor, path_lengths, goals, config
    )
    tx, ty, tdev = nav[0, 0].tolist()

    assert tdev == 0.0, f"path_dev must be pinned to 0.0 under the flag, got {tdev}"
    assert np.allclose([tx, ty], [np_x, np_y], atol=1e-3), (
        f"dir mismatch {(tx, ty)} vs {(np_x, np_y)}"
    )
