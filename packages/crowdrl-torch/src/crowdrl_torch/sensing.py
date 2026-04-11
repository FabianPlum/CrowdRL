"""Raycasting and KNN social sensing in PyTorch.

Port of ``crowdrl_core.sensing.cast_rays_batch`` and
``knn_social_batch``. All operations are pure PyTorch tensor math.

Shapes use MAX_AGENTS and MAX_SEGMENTS padding throughout, with
a leading (E,) environment batch dimension.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from crowdrl_torch.types import EnvConfig


def cast_rays(
    positions: Tensor,
    head_orientations: Tensor,
    torso_orientations: Tensor,
    shoulder_widths: Tensor,
    chest_depths: Tensor,
    active_mask: Tensor,
    n_agents: Tensor,
    wall_segments: Tensor,
    n_segments: Tensor,
    config: EnvConfig,
) -> Tensor:
    """Cast rays for all agents.

    Parameters
    ----------
    positions : (E, N, 2)
    head_orientations : (E, N)
    wall_segments : (E, S, 2, 2)
    n_segments : (E,)

    Returns
    -------
    readings : (E, N, n_rays) — normalised distances [0, 1]
    """
    E, N = positions.shape[:2]
    R = config.n_rays
    max_range = config.max_range
    fov_rad = math.radians(config.fov_deg)

    # Ray angles: (E, N, R)
    start_angles = head_orientations - fov_rad / 2.0
    t_vals = torch.linspace(0.0, 1.0, R, device=positions.device, dtype=positions.dtype)
    ray_angles = start_angles.unsqueeze(-1) + fov_rad * t_vals

    # Ray directions: (E, N, R, 2)
    ray_dirs = torch.stack([torch.cos(ray_angles), torch.sin(ray_angles)], dim=-1)

    # Origins: (E, N, 1, 2) for broadcasting
    origins = positions.unsqueeze(2)

    # Initialize best distances
    best_t = torch.full((E, N, R), max_range, dtype=positions.dtype, device=positions.device)

    # --- Ray-wall segment intersections ---
    best_t = _ray_wall_intersections(
        origins, ray_dirs, best_t, wall_segments, n_segments, max_range
    )

    # --- Ray-agent ellipse intersections ---
    best_t = _ray_agent_intersections(
        origins,
        ray_dirs,
        best_t,
        positions,
        torso_orientations,
        shoulder_widths,
        chest_depths,
        active_mask,
        n_agents,
        max_range,
    )

    # Normalize to [0, 1]
    readings = torch.clamp(best_t / max_range, max=1.0)

    # Fill inactive agents with 1.0
    readings = torch.where(active_mask.unsqueeze(-1), readings, torch.ones_like(readings))

    return readings


def _ray_wall_intersections(
    origins: Tensor,
    ray_dirs: Tensor,
    best_t: Tensor,
    wall_segments: Tensor,
    n_segments: Tensor,
    max_range: float,
) -> Tensor:
    """Test all rays against all wall segments.

    origins: (E, N, 1, 2), ray_dirs: (E, N, R, 2), wall_segments: (E, S, 2, 2)
    """
    seg_starts = wall_segments[:, :, 0, :]  # (E, S, 2)
    seg_ends = wall_segments[:, :, 1, :]  # (E, S, 2)
    seg_d = seg_ends - seg_starts  # (E, S, 2)

    # Broadcast to (E, N, R, S, 2)
    o = origins.unsqueeze(3)  # (E, N, 1, 1, 2)
    rd = ray_dirs.unsqueeze(3)  # (E, N, R, 1, 2)
    ss = seg_starts[:, None, None, :, :]  # (E, 1, 1, S, 2)
    sd = seg_d[:, None, None, :, :]  # (E, 1, 1, S, 2)

    # denom = ray_dir.x * seg_d.y - ray_dir.y * seg_d.x
    denom = rd[..., 0] * sd[..., 1] - rd[..., 1] * sd[..., 0]  # (E, N, R, S)

    # diff = seg_start - origin
    diff = ss - o  # (E, N, R, S, 2)

    t_num = diff[..., 0] * sd[..., 1] - diff[..., 1] * sd[..., 0]
    u_num = diff[..., 0] * rd[..., 1] - diff[..., 1] * rd[..., 0]

    safe_denom = torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
    t_param = t_num / safe_denom
    u_param = u_num / safe_denom

    # Valid hits
    valid = (denom.abs() >= 1e-12) & (t_param > 1e-8) & (u_param >= 0.0) & (u_param <= 1.0)

    # Mask padding segments: (E, 1, 1, S)
    S = wall_segments.shape[1]
    seg_idx = torch.arange(S, device=wall_segments.device)
    seg_mask = seg_idx < n_segments[:, None]  # (E, S)
    valid = valid & seg_mask[:, None, None, :]

    t_param = torch.where(valid, t_param, torch.full_like(t_param, max_range + 1.0))

    # Best wall hit per ray
    wall_best_t = t_param.amin(dim=-1)  # (E, N, R)
    best_t = torch.minimum(best_t, wall_best_t)

    return best_t


def _ray_agent_intersections(
    origins: Tensor,
    ray_dirs: Tensor,
    best_t: Tensor,
    positions: Tensor,
    torso_orientations: Tensor,
    shoulder_widths: Tensor,
    chest_depths: Tensor,
    active_mask: Tensor,
    n_agents: Tensor,
    max_range: float,
) -> Tensor:
    """Test all rays against all agent ellipses.

    origins: (E, N, 1, 2), ray_dirs: (E, N, R, 2)
    """
    E, N = positions.shape[:2]

    cos_t = torch.cos(-torso_orientations)  # (E, N)
    sin_t = torch.sin(-torso_orientations)

    # Origin relative to each target ellipse: (E, N_sensor, N_target, 2)
    # rel_origin[e, i, j] = positions[e, i] - positions[e, j] (sensor - target)
    rel_origin = positions.unsqueeze(2) - positions.unsqueeze(1)  # (E, N, N, 2)

    # Rotate into ellipse-local frame: (E, N, N)
    rot_ox = cos_t.unsqueeze(1) * rel_origin[..., 0] - sin_t.unsqueeze(1) * rel_origin[..., 1]
    rot_oy = sin_t.unsqueeze(1) * rel_origin[..., 0] + cos_t.unsqueeze(1) * rel_origin[..., 1]

    # Scale to unit circle
    origin_sx = rot_ox / chest_depths.unsqueeze(1)  # (E, N, N)
    origin_sy = rot_oy / shoulder_widths.unsqueeze(1)

    # Rotate ray directions: (E, N, R, N_target)
    rd_x = ray_dirs[..., 0]  # (E, N, R)
    rd_y = ray_dirs[..., 1]

    rot_dx = cos_t[:, None, None, :] * rd_x.unsqueeze(3) - sin_t[
        :, None, None, :
    ] * rd_y.unsqueeze(3)  # (E, N, R, N)
    rot_dy = sin_t[:, None, None, :] * rd_x.unsqueeze(3) + cos_t[
        :, None, None, :
    ] * rd_y.unsqueeze(3)

    dir_sx = rot_dx / chest_depths[:, None, None, :]
    dir_sy = rot_dy / shoulder_widths[:, None, None, :]

    # Quadratic coefficients: (E, N, R, N)
    o_sx = origin_sx.unsqueeze(2)  # (E, N, 1, N)
    o_sy = origin_sy.unsqueeze(2)

    qa = dir_sx**2 + dir_sy**2
    qb = 2.0 * (o_sx * dir_sx + o_sy * dir_sy)
    qc = o_sx**2 + o_sy**2 - 1.0

    discriminant = qb**2 - 4.0 * qa * qc
    sqrt_disc = torch.sqrt(torch.clamp(discriminant, min=0.0))
    safe_qa = torch.where(qa.abs() < 1e-15, torch.ones_like(qa), qa)
    t1 = (-qb - sqrt_disc) / (2.0 * safe_qa)
    t2 = (-qb + sqrt_disc) / (2.0 * safe_qa)

    eps = 1e-8
    t1_valid = (discriminant >= 0) & (t1 > eps)
    t2_valid = (discriminant >= 0) & (t2 > eps)
    agent_t = torch.where(
        t1_valid, t1, torch.where(t2_valid, t2, torch.full_like(t1, max_range + 1.0))
    )

    # Self-exclusion mask: (N, N)
    i_idx = torch.arange(N, device=positions.device)
    self_exclude = i_idx.unsqueeze(0) != i_idx.unsqueeze(1)  # (N, N)

    # Active + agent count mask
    target_active = active_mask & (i_idx.unsqueeze(0) < n_agents.unsqueeze(1))  # (E, N)
    test_mask = self_exclude.unsqueeze(0) & target_active.unsqueeze(1)  # (E, N, N)

    # Broadcast to (E, N, R, N)
    test_mask_4d = test_mask.unsqueeze(2)
    agent_t = torch.where(test_mask_4d, agent_t, torch.full_like(agent_t, max_range + 1.0))

    # Best agent hit per ray
    agent_best_t = agent_t.amin(dim=-1)  # (E, N, R)
    best_t = torch.minimum(best_t, agent_best_t)

    return best_t


def knn_social(
    positions: Tensor,
    velocities: Tensor,
    torso_orientations: Tensor,
    shoulder_widths: Tensor,
    chest_depths: Tensor,
    active_mask: Tensor,
    n_agents: Tensor,
    config: EnvConfig,
) -> Tensor:
    """K-nearest-neighbour social sensing for all agents.

    Parameters
    ----------
    positions : (E, N, 2)

    Returns
    -------
    social : (E, N, K, 7) — per-neighbour features in ego frame
    """
    E, N = positions.shape[:2]
    K = config.k_neighbours

    # Pairwise squared distances: (E, N, N)
    dist_sq = ((positions.unsqueeze(1) - positions.unsqueeze(2)) ** 2).sum(dim=-1)

    # Mask self and inactive/padding agents
    i_idx = torch.arange(N, device=positions.device)
    self_mask = i_idx.unsqueeze(0) == i_idx.unsqueeze(1)  # (N, N)
    agent_valid = active_mask & (i_idx.unsqueeze(0) < n_agents.unsqueeze(1))  # (E, N)
    invalid_target = ~agent_valid.unsqueeze(1)  # (E, 1, N) — broadcast over sensor dim
    dist_sq = torch.where(
        self_mask.unsqueeze(0) | invalid_target,
        torch.tensor(float("inf"), device=positions.device),
        dist_sq,
    )

    # Find K nearest using topk (negate for bottom-K)
    # topk returns largest; we want smallest, so negate
    neg_dist_sq = -dist_sq
    _, knn_idx = neg_dist_sq.topk(K, dim=2)  # (E, N, K) — indices of K nearest

    # Check which neighbors are valid (finite distance)
    knn_dists = torch.gather(dist_sq, 2, knn_idx)  # (E, N, K)
    valid_mask = torch.isfinite(knn_dists)

    # Gather neighbor data using advanced indexing
    # Expand knn_idx for 2D fields: (E, N, K, 2)
    knn_idx_2d = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 2)
    nb_pos = torch.gather(positions.unsqueeze(1).expand(-1, N, -1, -1), 2, knn_idx_2d)
    nb_vel = torch.gather(velocities.unsqueeze(1).expand(-1, N, -1, -1), 2, knn_idx_2d)

    # For 1D fields: (E, N, K)
    nb_orient = torch.gather(torso_orientations.unsqueeze(1).expand(-1, N, -1), 2, knn_idx)
    nb_widths = torch.gather(shoulder_widths.unsqueeze(1).expand(-1, N, -1), 2, knn_idx)
    nb_depths = torch.gather(chest_depths.unsqueeze(1).expand(-1, N, -1), 2, knn_idx)

    # Ego frame rotation
    ego_heading = torso_orientations  # (E, N)
    cos_h = torch.cos(-ego_heading)
    sin_h = torch.sin(-ego_heading)

    # Relative positions in global frame: (E, N, K, 2)
    rel_pos_global = nb_pos - positions.unsqueeze(2)
    rel_vel_global = nb_vel - velocities.unsqueeze(2)

    # Rotate to ego frame
    rel_pos_x = (
        cos_h.unsqueeze(2) * rel_pos_global[..., 0] - sin_h.unsqueeze(2) * rel_pos_global[..., 1]
    )
    rel_pos_y = (
        sin_h.unsqueeze(2) * rel_pos_global[..., 0] + cos_h.unsqueeze(2) * rel_pos_global[..., 1]
    )
    rel_vel_x = (
        cos_h.unsqueeze(2) * rel_vel_global[..., 0] - sin_h.unsqueeze(2) * rel_vel_global[..., 1]
    )
    rel_vel_y = (
        sin_h.unsqueeze(2) * rel_vel_global[..., 0] + cos_h.unsqueeze(2) * rel_vel_global[..., 1]
    )

    # Relative orientation
    rel_orient = (nb_orient - ego_heading.unsqueeze(2) + math.pi) % (2 * math.pi) - math.pi

    # Assemble: (E, N, K, 7)
    zero = torch.zeros_like(rel_pos_x)
    social = torch.stack(
        [
            torch.where(valid_mask, rel_pos_x, zero),
            torch.where(valid_mask, rel_pos_y, zero),
            torch.where(valid_mask, rel_vel_x, zero),
            torch.where(valid_mask, rel_vel_y, zero),
            torch.where(valid_mask, rel_orient, zero),
            torch.where(valid_mask, nb_widths, zero),
            torch.where(valid_mask, nb_depths, zero),
        ],
        dim=-1,
    )

    return social


def match_persistent_neighbors(
    positions: Tensor,
    prev_slots: Tensor,
    active_mask: Tensor,
    n_agents: Tensor,
    sensing_radius: float,
    config: EnvConfig,
) -> Tensor:
    """GPU persistent-neighbor matching (batched equivalent of the numpy
    ``crowdrl_core.sensing.match_persistent_neighbors``).

    For each ego agent in each parallel env, update the K-slot neighbor-ID
    table so that previously-assigned neighbors stay pinned while they are
    in range and active, and empty slots get filled greedily with the
    nearest unassigned in-range active agent.

    Parameters
    ----------
    positions : (E, N, 2) float32 -- current agent positions.
    prev_slots : (E, N, K) int32 -- previous step's neighbor-ID table,
        -1 for empty slots.
    active_mask : (E, N) bool -- True for currently active agents.
    n_agents : (E,) int32 -- real agent count per env (N is padded).
    sensing_radius : float -- metres, eviction/fill threshold.
    config : EnvConfig

    Returns
    -------
    new_slots : (E, N, K) int32 -- updated assignment table. Inactive or
        padding ego agents always have all -1 slots.
    """
    E, N = positions.shape[:2]
    K = config.k_neighbours
    device = positions.device
    dtype_int = prev_slots.dtype
    inf_t = torch.tensor(float("inf"), device=device, dtype=positions.dtype)

    # Pairwise squared distances: (E, N, N)
    diff = positions.unsqueeze(1) - positions.unsqueeze(2)
    dist_sq = (diff * diff).sum(dim=-1)

    # Mask self and inactive/padding agents as candidates.
    i_range = torch.arange(N, device=device)
    self_mask = i_range.unsqueeze(0) == i_range.unsqueeze(1)  # (N, N)
    agent_valid = active_mask & (i_range.unsqueeze(0) < n_agents.unsqueeze(1))  # (E, N)
    invalid_target = ~agent_valid.unsqueeze(1)  # (E, 1, N) broadcast on ego dim
    dist_sq = torch.where(self_mask.unsqueeze(0) | invalid_target, inf_t, dist_sq)

    # Also mask by sensing radius -- anything beyond is treated as unreachable.
    radius_sq = sensing_radius * sensing_radius
    dist_sq = torch.where(dist_sq > radius_sq, inf_t, dist_sq)

    # --- Step 1: retain eligible prev slots ---------------------------------
    # Clamp -1 placeholders to 0 so ``torch.gather`` never indexes out of
    # bounds; we mask the gather result with ``prev_slots >= 0`` afterwards.
    prev_safe = prev_slots.clamp(min=0).long()  # (E, N, K)
    prev_dist_sq = torch.gather(dist_sq, dim=2, index=prev_safe)
    prev_eligible = (prev_slots >= 0) & torch.isfinite(prev_dist_sq)
    retained = torch.where(
        prev_eligible,
        prev_slots,
        torch.full_like(prev_slots, -1),
    )  # (E, N, K) -- each row has the retained prev_ids or -1

    # --- Step 2: fill empty slots in one vectorized pass -------------------
    # Key optimisation over the obvious per-slot loop: we build the
    # "retained neighbors" exclusion mask exactly ONCE, then call topk(K)
    # once to get the K smallest available distances per ego. Finally we
    # use a cumulative-count trick to map the j-th empty slot to the j-th
    # topk candidate, fully vectorised (no K-iteration loop).
    retained_safe = retained.clamp(min=0).long()  # (E, N, K)
    retained_mask = retained >= 0  # (E, N, K) bool
    retained_one_hot = torch.nn.functional.one_hot(
        retained_safe, num_classes=N
    )  # (E, N, K, N) int64
    retained_one_hot = retained_one_hot * retained_mask.unsqueeze(-1).to(retained_one_hot.dtype)
    exclude_static = retained_one_hot.any(dim=2)  # (E, N, N) bool

    dist_avail = torch.where(exclude_static, inf_t, dist_sq)

    # K smallest available distances per ego -- topk on the negated distance
    # gives us the smallest values.
    neg_topk_vals, candidate_idx = torch.topk(-dist_avail, K, dim=2)
    candidate_dist_sq = -neg_topk_vals  # (E, N, K)
    candidate_valid = torch.isfinite(candidate_dist_sq)  # (E, N, K)

    # For each slot k, determine its "rank among empty slots" so we can
    # pick the right topk candidate. Example: retained_mask = [T, F, T, F, F]
    # -> empty_mask = [F, T, F, T, T] -> empty_rank (cumsum-1) = [-1, 0, 0, 1, 2].
    # Empty slots read ranks 0, 1, 2 in order, which gives them the first,
    # second, and third candidate from topk.
    empty_mask = ~retained_mask
    empty_rank = empty_mask.to(torch.int64).cumsum(dim=2) - 1  # (E, N, K)
    empty_rank_safe = empty_rank.clamp(min=0)  # clamp for retained slots
    fill_values = torch.gather(candidate_idx, dim=2, index=empty_rank_safe)
    rank_valid = torch.gather(candidate_valid, dim=2, index=empty_rank_safe)
    fill_mask = empty_mask & rank_valid

    # Final assignment: retained slots keep prev_ids, filled slots get
    # topk candidates, otherwise -1.
    new_slots = torch.where(
        retained_mask,
        retained,
        torch.where(
            fill_mask,
            fill_values.to(dtype_int),
            torch.full_like(retained, -1),
        ),
    )

    # Rows whose ego agent is inactive/padding should be all -1 regardless
    # of what the gather/fill returned.
    ego_invalid = ~agent_valid  # (E, N)
    new_slots = torch.where(
        ego_invalid.unsqueeze(-1),
        torch.full_like(new_slots, -1),
        new_slots,
    )

    return new_slots
