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

_TEMPORAL_EPS = 1e-6


def compute_temporal_features(
    positions: Tensor,
    spawn_positions: Tensor,
    initial_goal_distances: Tensor,
    cumulative_path_length: Tensor,
    pos_history: Tensor,
    gdist_history: Tensor,
    goal_positions: Tensor,
    preferred_speeds: Tensor,
    step_count: Tensor,
    config: EnvConfig,
) -> Tensor:
    """Compute the 6D temporal-memory feature block.

    All operations are pure tensor ops — torch.compile friendly, no host sync.

    Parameters
    ----------
    positions : (E, N, 2) — current positions
    spawn_positions : (E, N, 2) — positions at episode start
    initial_goal_distances : (E, N) — ||goal - spawn|| at episode start
    cumulative_path_length : (E, N) — running path length
    pos_history : (E, N, W+1, 2) — ring buffer of positions
    gdist_history : (E, N, W+1) — ring buffer of goal distances
    goal_positions : (E, N, 2) — current goal
    preferred_speeds : (E, N) — per-agent preferred speed
    step_count : (E,) int — current episode step per env
    config : EnvConfig

    Returns
    -------
    features : (E, N, 6) float — the 6 temporal features per agent
    """
    E, N = positions.shape[:2]
    W = config.temporal_memory_window
    buf_size = W + 1
    eps = _TEMPORAL_EPS

    # Current goal distance
    gdist_now = ((goal_positions - positions) ** 2).sum(dim=-1).sqrt()  # (E, N)

    # Read "W steps ago" slot from the ring buffer.
    # Writer stores pos_{t+1} at index (t % buf_size) where t is the pre-step
    # step_count. After the step, step_count becomes t+1. The oldest valid
    # entry in a buf_size=W+1 ring is at index (t+1) % buf_size = new_step_count
    # % buf_size, which contains pos_{t+1-W} (or spawn_pos if t<W, since
    # all slots are initialised to spawn_pos on reset).
    read_idx = (step_count % buf_size).long()  # (E,)

    # Gather: pos_history[e, n, read_idx[e], :] -> (E, N, 2)
    read_idx_pos = read_idx.view(E, 1, 1, 1).expand(E, N, 1, 2)  # (E, N, 1, 2)
    pos_window = pos_history.gather(dim=2, index=read_idx_pos).squeeze(2)  # (E, N, 2)

    read_idx_gd = read_idx.view(E, 1, 1).expand(E, N, 1)  # (E, N, 1)
    gdist_window = gdist_history.gather(dim=2, index=read_idx_gd).squeeze(2)  # (E, N)

    # 1. Displacement from spawn / initial goal distance
    disp_spawn = ((positions - spawn_positions) ** 2).sum(dim=-1).sqrt()
    safe_init = initial_goal_distances.clamp(min=eps)
    disp_spawn_norm = disp_spawn / safe_init

    # 2. Cumulative path length / initial goal distance
    cum_path_norm = cumulative_path_length / safe_init

    # 3. Path efficiency: displacement / cumulative path
    safe_cum = cumulative_path_length.clamp(min=eps)
    path_eff = (disp_spawn / safe_cum).clamp(min=0.0, max=1.0)

    # 4. Elapsed fraction (broadcast step_count per env)
    max_steps = float(max(config.max_steps, 1))
    elapsed = (step_count.to(positions.dtype).view(E, 1) / max_steps).expand(E, N)

    # 5. Displacement over window, normalised by (v_pref * W * dt)
    disp_window = ((positions - pos_window) ** 2).sum(dim=-1).sqrt()
    expected_window = (preferred_speeds * (W * config.dt)).clamp(min=eps)
    disp_window_norm = disp_window / expected_window

    # 6. Goal progress over window (positive = approaching goal)
    goal_progress_window = gdist_window - gdist_now
    goal_progress_window_norm = goal_progress_window / expected_window

    return torch.stack(
        [
            disp_spawn_norm,
            cum_path_norm,
            path_eff,
            elapsed,
            disp_window_norm,
            goal_progress_window_norm,
        ],
        dim=-1,
    )  # (E, N, 6)


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


def compute_neighbor_vel_history_features(
    neighbor_ids: Tensor,
    neighbor_vel_history: Tensor,
    cos_h: Tensor,
    sin_h: Tensor,
    step_count: Tensor,
    config: EnvConfig,
) -> Tensor:
    """Compute the (E, N, K*2) neighbor velocity-history feature block.

    For each of the K persistent slots:
        diff_global = vel_at_step_now - vel_at_W_steps_ago
        diff_ego    = rot(-ego_heading) @ diff_global
    Empty slots (neighbor_ids == -1) are zeroed out.

    The newest ring-buffer slot corresponds to the last write index, and
    the oldest to the NEXT write index (which currently holds W_n+1
    steps ago or zeros if the buffer hasn't filled yet). See the numpy
    reference in crowdrl_core.observation for the same derivation.
    """
    E, N, buf_size, K, _ = neighbor_vel_history.shape
    W_n = config.neighbor_vel_history_window
    assert buf_size == W_n + 1, "neighbor_vel_history buffer size must equal W_n+1"

    # Indices are per-env scalars: (E,). Cast to long for gather.
    newest = ((step_count - 1) % buf_size).long()  # (E,)
    oldest = (step_count % buf_size).long()  # (E,)

    # Gather vel_now at newest index: output (E, N, K, 2)
    newest_idx = newest.view(E, 1, 1, 1, 1).expand(E, N, 1, K, 2)
    vel_now = neighbor_vel_history.gather(dim=2, index=newest_idx).squeeze(2)

    oldest_idx = oldest.view(E, 1, 1, 1, 1).expand(E, N, 1, K, 2)
    vel_old = neighbor_vel_history.gather(dim=2, index=oldest_idx).squeeze(2)

    diff_global = vel_now - vel_old  # (E, N, K, 2)

    # Rotate each diff into the current ego frame. cos_h / sin_h are (E, N).
    cos_exp = cos_h.unsqueeze(-1)  # (E, N, 1)
    sin_exp = sin_h.unsqueeze(-1)  # (E, N, 1)
    diff_ex = cos_exp * diff_global[..., 0] - sin_exp * diff_global[..., 1]
    diff_ey = sin_exp * diff_global[..., 0] + cos_exp * diff_global[..., 1]
    diff_ego = torch.stack([diff_ex, diff_ey], dim=-1)  # (E, N, K, 2)

    # Mask empty slots
    valid = (neighbor_ids >= 0).unsqueeze(-1)  # (E, N, K, 1)
    diff_ego = torch.where(valid, diff_ego, torch.zeros_like(diff_ego))

    # Flatten slot x feature to K*2
    return diff_ego.reshape(E, N, K * 2)


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
    spawn_positions: Tensor | None = None,
    initial_goal_distances: Tensor | None = None,
    cumulative_path_length: Tensor | None = None,
    pos_history: Tensor | None = None,
    gdist_history: Tensor | None = None,
    preferred_speeds: Tensor | None = None,
    step_count: Tensor | None = None,
    neighbor_ids: Tensor | None = None,
    neighbor_vel_history: Tensor | None = None,
) -> Tensor:
    """Build observations for all agents.

    Parameters
    ----------
    positions : (E, N, 2)
    waypoints : (E, N, MAX_WP, 2) — pre-computed funnel waypoints (optional)
    n_waypoints : (E, N) int32 — waypoint count per agent (optional)
    waypoint_cursor : (E, N) int32 — current progress index (optional)
    waypoint_path_lengths : (E, N, MAX_WP) — cumulative remaining distance (optional)
    spawn_positions, initial_goal_distances, cumulative_path_length, pos_history,
    gdist_history, preferred_speeds, step_count : temporal-memory state; required
        when ``config.use_temporal_memory`` is True, otherwise ignored.

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
    parts = [ego_state, social_flat, rays]
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
        parts.append(nav)

    # --- Temporal memory (E, N, 6) ---
    if config.use_temporal_memory:
        memory = compute_temporal_features(
            positions,
            spawn_positions,
            initial_goal_distances,
            cumulative_path_length,
            pos_history,
            gdist_history,
            goal_positions,
            preferred_speeds,
            step_count,
            config,
        )
        parts.append(memory)

    # --- Neighbor velocity history (E, N, K*2) ---
    if config.use_neighbor_memory and config.use_neighbor_vel_history:
        nb_vel_features = compute_neighbor_vel_history_features(
            neighbor_ids,
            neighbor_vel_history,
            cos_h,
            sin_h,
            step_count,
            config,
        )
        parts.append(nb_vel_features)

    obs = torch.cat(parts, dim=-1)  # (E, N, obs_dim)

    # Zero out inactive agents
    obs = torch.where(active_mask.unsqueeze(-1), obs, torch.zeros_like(obs))

    return obs
