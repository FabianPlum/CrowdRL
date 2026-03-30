"""Observation builder in PyTorch — assembles full obs vector from state arrays.

Port of ``crowdrl_core.observation.build_observations_batch``.
Produces (E, MAX_AGENTS, obs_dim) tensor; inactive agents get zeros.

Shapes carry a leading (E,) environment batch dimension throughout.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from crowdrl_torch.sensing import cast_rays, knn_social
from crowdrl_torch.types import EnvConfig


def compute_navmesh_signals(
    positions: Tensor,
    cos_h: Tensor,
    sin_h: Tensor,
    waypoints: Tensor,
    n_waypoints: Tensor,
    waypoint_cursor: Tensor,
    waypoint_path_lengths: Tensor,
    goal_positions: Tensor,
    config: EnvConfig,
) -> Tensor:
    """Compute blended waypoint direction + path deviation from pre-computed waypoints.

    All operations are pure tensor ops — no CPU round-trip.

    Parameters
    ----------
    positions : (E, N, 2)
    cos_h, sin_h : (E, N) — pre-computed cos/sin of -torso_orientation
    waypoints : (E, N, MAX_WP, 2)
    n_waypoints : (E, N) int32
    waypoint_cursor : (E, N) int32
    waypoint_path_lengths : (E, N, MAX_WP)
    goal_positions : (E, N, 2)
    config : EnvConfig

    Returns
    -------
    nav_signals : (E, N, 3) — [dir_ego_x, dir_ego_y, path_deviation]
    """
    E, N = positions.shape[:2]
    MAX_WP = config.max_waypoints
    dtype = positions.dtype

    cursor = waypoint_cursor.long()  # (E, N)
    n_wp = n_waypoints.long()  # (E, N)

    # Clamp cursor to valid range
    max_idx = (n_wp - 1).clamp(min=0)  # (E, N)
    cursor_a = cursor.clamp(min=0, max=MAX_WP - 1).clamp(max=max_idx)
    cursor_b = (cursor_a + 1).clamp(max=max_idx)  # next waypoint (or same if last)

    # Gather waypoints at cursor_a and cursor_b: (E, N, 2)
    idx_a = cursor_a.unsqueeze(-1).unsqueeze(-1).expand(E, N, 1, 2)  # (E, N, 1, 2)
    idx_b = cursor_b.unsqueeze(-1).unsqueeze(-1).expand(E, N, 1, 2)  # (E, N, 1, 2)
    wp_a = waypoints.gather(2, idx_a).squeeze(2)  # (E, N, 2)
    wp_b = waypoints.gather(2, idx_b).squeeze(2)  # (E, N, 2)

    # Distances to each waypoint
    diff_a = wp_a - positions  # (E, N, 2)
    diff_b = wp_b - positions  # (E, N, 2)
    d_a = (diff_a**2).sum(dim=-1).sqrt()  # (E, N)
    d_b = (diff_b**2).sum(dim=-1).sqrt()  # (E, N)

    # Blending: closer waypoint gets LESS influence.
    # When only one waypoint remains (cursor_a == cursor_b), both point to the
    # same location and the blend doesn't matter — weight is effectively 1.0.
    eps = 1e-8
    total = d_a + d_b + eps
    weight_a = d_a / total  # small when close to wp_a → low influence
    weight_b = d_b / total

    blended = weight_a.unsqueeze(-1) * wp_a + weight_b.unsqueeze(-1) * wp_b  # (E, N, 2)

    # Direction from agent to blended target (world frame)
    direction = blended - positions  # (E, N, 2)
    dir_norm = (direction**2).sum(dim=-1).sqrt().clamp(min=eps)  # (E, N)
    direction = direction / dir_norm.unsqueeze(-1)  # unit vector

    # Rotate to ego frame
    dir_ego_x = cos_h * direction[..., 0] - sin_h * direction[..., 1]
    dir_ego_y = sin_h * direction[..., 0] + cos_h * direction[..., 1]

    # Path deviation: (remaining_path / euclidean_to_goal) - 1
    # remaining_path = distance_to_current_wp + pre-computed cumulative from current wp
    idx_pl = cursor_a.unsqueeze(-1)  # (E, N, 1)
    remaining_from_wp = waypoint_path_lengths.gather(2, idx_pl).squeeze(2)  # (E, N)
    remaining_path = d_a + remaining_from_wp  # (E, N)

    euclidean = ((goal_positions - positions) ** 2).sum(dim=-1).sqrt().clamp(min=eps)
    path_dev = (remaining_path / euclidean) - 1.0
    path_dev = path_dev.clamp(min=0.0)  # can't be negative

    # Zero out agents with no waypoints
    has_wp = (n_wp > 0).to(dtype)  # (E, N)
    dir_ego_x = dir_ego_x * has_wp
    dir_ego_y = dir_ego_y * has_wp
    path_dev = path_dev * has_wp

    return torch.stack([dir_ego_x, dir_ego_y, path_dev], dim=-1)  # (E, N, 3)


def build_observations(
    positions: Tensor,
    velocities: Tensor,
    torso_orientations: Tensor,
    head_orientations: Tensor,
    shoulder_widths: Tensor,
    chest_depths: Tensor,
    goal_positions: Tensor,
    active_mask: Tensor,
    n_agents: Tensor,
    wall_segments: Tensor,
    n_segments: Tensor,
    config: EnvConfig,
    waypoints: Tensor | None = None,
    n_waypoints: Tensor | None = None,
    waypoint_cursor: Tensor | None = None,
    waypoint_path_lengths: Tensor | None = None,
) -> Tensor:
    """Build observations for all agents.

    Parameters
    ----------
    positions : (E, N, 2)
    waypoints : (E, N, MAX_WP, 2) — pre-computed funnel waypoints (optional)
    n_waypoints : (E, N) int32 — waypoint count per agent (optional)
    waypoint_cursor : (E, N) int32 — current progress index (optional)
    waypoint_path_lengths : (E, N, MAX_WP) — cumulative remaining distance (optional)

    Returns
    -------
    obs : (E, N, obs_dim) — zeros for inactive/padding agents
    """
    E, N = positions.shape[:2]

    # --- Ego state (E, N, 7) ---
    cos_h = torch.cos(-torso_orientations)
    sin_h = torch.sin(-torso_orientations)

    # Goal direction in ego frame
    goal_diff = goal_positions - positions  # (E, N, 2)
    goal_dist = (goal_diff**2).sum(dim=-1).sqrt()  # (E, N)
    safe_dist = torch.where(goal_dist > 1e-10, goal_dist, torch.ones_like(goal_dist))
    goal_unit = goal_diff / safe_dist.unsqueeze(-1)
    goal_unit = torch.where(
        (goal_dist > 1e-10).unsqueeze(-1), goal_unit, torch.zeros_like(goal_unit)
    )

    goal_dir_x = cos_h * goal_unit[..., 0] - sin_h * goal_unit[..., 1]
    goal_dir_y = sin_h * goal_unit[..., 0] + cos_h * goal_unit[..., 1]

    # Velocity in ego frame
    vel_ego_x = cos_h * velocities[..., 0] - sin_h * velocities[..., 1]
    vel_ego_y = sin_h * velocities[..., 0] + cos_h * velocities[..., 1]

    # Speed
    speed = (velocities**2).sum(dim=-1).sqrt()

    # Torso angle in ego frame is 0 by construction
    torso_angle = torch.zeros(E, N, dtype=positions.dtype, device=positions.device)

    # Head angle relative to torso
    head_rel_torso = (head_orientations - torso_orientations + math.pi) % (2 * math.pi) - math.pi

    ego_state = torch.stack(
        [goal_dir_x, goal_dir_y, vel_ego_x, vel_ego_y, speed, torso_angle, head_rel_torso],
        dim=-1,
    )  # (E, N, 7)

    # --- Social sensing (E, N, K*7) ---
    social = knn_social(
        positions,
        velocities,
        torso_orientations,
        shoulder_widths,
        chest_depths,
        active_mask,
        n_agents,
        config,
    )  # (E, N, K, 7)
    social_flat = social.reshape(E, N, -1)  # (E, N, K*7)

    # --- Raycasting (E, N, R) ---
    rays = cast_rays(
        positions,
        head_orientations,
        torso_orientations,
        shoulder_widths,
        chest_depths,
        active_mask,
        n_agents,
        wall_segments,
        n_segments,
        config,
    )  # (E, N, R)

    # --- Navmesh signals (E, N, 3) ---
    if config.use_navmesh:
        nav = compute_navmesh_signals(
            positions,
            cos_h,
            sin_h,
            waypoints,
            n_waypoints,
            waypoint_cursor,
            waypoint_path_lengths,
            goal_positions,
            config,
        )
        obs = torch.cat([ego_state, social_flat, rays, nav], dim=-1)
    else:
        obs = torch.cat([ego_state, social_flat, rays], dim=-1)  # (E, N, obs_dim)

    # Zero out inactive agents
    obs = torch.where(active_mask.unsqueeze(-1), obs, torch.zeros_like(obs))

    return obs
