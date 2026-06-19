"""Sensing: raycast engine and K-nearest-neighbour social query.

Raycasts are head-anchored (FOV follows head orientation, not torso).
Social sensing returns K nearest neighbours' relative state.

Both operate on WorldState — agnostic to whether state comes from
the training environment or from JuPedSim.

Provides both per-agent functions (for deployment) and batch functions
(for training) that process all agents in vectorized numpy operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from crowdrl_core.collision import ray_ellipse_intersection
from crowdrl_core.world_state import WorldState

# Hit type codes for 2-channel raycasts
HIT_NONE: float = 1.0  # no hit (ray returns max range)
HIT_WALL: float = 0.0
HIT_OBSTACLE: float = 0.5
HIT_AGENT: float = 1.0


@dataclass(frozen=True)
class RaycastConfig:
    """Configuration for the raycast sensor."""

    n_rays: int = 16
    """Number of rays."""

    fov_deg: float = 200.0
    """Field of view in degrees, centred on head forward direction."""

    max_range: float = 5.0
    """Maximum sensing range in metres."""

    two_channel: bool = False
    """If True, each ray returns (distance, hit_type) instead of just distance."""


def _ray_segment_intersection(
    origin: NDArray, direction: NDArray, p1: NDArray, p2: NDArray
) -> float | None:
    """Find the distance along a ray to its intersection with a line segment.

    Returns t > 0 if the ray hits the segment, None otherwise.
    """
    d = p2 - p1
    denom = direction[0] * d[1] - direction[1] * d[0]
    if abs(denom) < 1e-12:
        return None  # Parallel

    t_num = (p1[0] - origin[0]) * d[1] - (p1[1] - origin[1]) * d[0]
    u_num = (p1[0] - origin[0]) * direction[1] - (p1[1] - origin[1]) * direction[0]

    t = t_num / denom
    u = u_num / denom

    if t > 1e-8 and 0.0 <= u <= 1.0:
        return float(t)
    return None


def cast_rays(
    world: WorldState,
    agent_idx: int,
    config: RaycastConfig,
) -> NDArray[np.float64]:
    """Cast N rays from agent's position along head-anchored FOV.

    Parameters
    ----------
    world : WorldState
    agent_idx : int
        Which agent is sensing.
    config : RaycastConfig

    Returns
    -------
    readings : (n_rays,) or (n_rays, 2) array
        If single-channel: normalised distances [0, 1] (1.0 = no hit / max range).
        If two-channel: [[dist, hit_type], ...] for each ray.
    """
    origin = world.positions[agent_idx]
    head_angle = world.head_orientations[agent_idx]

    fov_rad = np.radians(config.fov_deg)
    start_angle = head_angle - fov_rad / 2.0

    if config.n_rays == 1:
        ray_angles = np.array([head_angle])
    else:
        ray_angles = np.linspace(start_angle, start_angle + fov_rad, config.n_rays)

    if config.two_channel:
        readings = np.ones((config.n_rays, 2), dtype=np.float64)
        readings[:, 1] = HIT_NONE
    else:
        readings = np.ones(config.n_rays, dtype=np.float64)

    for r, angle in enumerate(ray_angles):
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        best_t = config.max_range
        best_hit_type = HIT_NONE

        # Test against wall segments
        if world.wall_segments is not None:
            for seg in world.wall_segments:
                t = _ray_segment_intersection(origin, direction, seg[0], seg[1])
                if t is not None and t < best_t:
                    best_t = t
                    # Determine if this is exterior wall or obstacle hole
                    # For simplicity, all wall segments are treated as walls
                    best_hit_type = HIT_WALL

        # Test against other agents' elliptical boundaries
        for j in range(world.n_agents):
            if j == agent_idx:
                continue
            if world.active_mask is not None and not world.active_mask[j]:
                continue

            t = ray_ellipse_intersection(
                origin,
                direction,
                world.positions[j],
                world.shoulder_widths[j],
                world.chest_depths[j],
                world.torso_orientations[j],
            )
            if t is not None and t < best_t:
                best_t = t
                best_hit_type = HIT_AGENT

        # Normalise to [0, 1]
        normalised = min(best_t / config.max_range, 1.0)

        if config.two_channel:
            readings[r, 0] = normalised
            readings[r, 1] = best_hit_type if normalised < 1.0 else HIT_NONE
        else:
            readings[r] = normalised

    return readings


def cast_rays_batch(
    world: WorldState,
    agent_indices: NDArray[np.intp],
    config: RaycastConfig,
) -> NDArray[np.float64]:
    """Cast rays for multiple agents in a single vectorized operation.

    Parameters
    ----------
    world : WorldState
    agent_indices : (M,) array of agent indices to sense for
    config : RaycastConfig

    Returns
    -------
    readings : (M, n_rays) or (M, n_rays, 2) array
    """
    M = len(agent_indices)
    R = config.n_rays
    max_range = config.max_range
    fov_rad = np.radians(config.fov_deg)

    # Compute ray angles for all agents: (M, R)
    head_angles = world.head_orientations[agent_indices]  # (M,)
    start_angles = head_angles - fov_rad / 2.0

    if R == 1:
        ray_angles = head_angles[:, np.newaxis]  # (M, 1)
    else:
        # (M, R): each row is linspace from start to start+fov
        t_vals = np.linspace(0.0, 1.0, R)
        ray_angles = start_angles[:, np.newaxis] + fov_rad * t_vals[np.newaxis, :]

    # Ray directions: (M, R, 2)
    ray_dirs = np.stack([np.cos(ray_angles), np.sin(ray_angles)], axis=-1)

    # Origins: (M, 1, 2) for broadcasting
    origins = world.positions[agent_indices][:, np.newaxis, :]  # (M, 1, 2)

    # Initialize best distances to max_range
    best_t = np.full((M, R), max_range, dtype=np.float64)

    if config.two_channel:
        best_hit_type = np.full((M, R), HIT_NONE, dtype=np.float64)

    # --- Ray-wall segment intersections (vectorized) ---
    if world.wall_segments is not None and len(world.wall_segments) > 0:
        seg_starts = np.array([s[0] for s in world.wall_segments])  # (W, 2)
        seg_ends = np.array([s[1] for s in world.wall_segments])  # (W, 2)
        seg_d = seg_ends - seg_starts  # (W, 2)

        # Broadcast: origins (M, 1, 1, 2), seg_starts (1, 1, W, 2)
        # ray_dirs (M, R, 1, 2), seg_d (1, 1, W, 2)
        o = origins[:, :, np.newaxis, :]  # (M, 1, 1, 2) → broadcast to (M, R, W, 2)
        rd = ray_dirs[:, :, np.newaxis, :]  # (M, R, 1, 2)
        ss = seg_starts[np.newaxis, np.newaxis, :, :]  # (1, 1, W, 2)
        sd = seg_d[np.newaxis, np.newaxis, :, :]  # (1, 1, W, 2)

        # denom = ray_dir.x * seg_d.y - ray_dir.y * seg_d.x
        denom = rd[..., 0] * sd[..., 1] - rd[..., 1] * sd[..., 0]  # (M, R, W)

        # diff = seg_start - origin
        diff = ss - o  # (M, R, W, 2) via broadcast from (M, 1, 1, 2)

        t_num = diff[..., 0] * sd[..., 1] - diff[..., 1] * sd[..., 0]
        u_num = diff[..., 0] * rd[..., 1] - diff[..., 1] * rd[..., 0]

        # Avoid division by zero
        safe_denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        t_param = t_num / safe_denom
        u_param = u_num / safe_denom

        # Valid hits: t > eps, 0 <= u <= 1, and not parallel
        valid = (np.abs(denom) >= 1e-12) & (t_param > 1e-8) & (u_param >= 0.0) & (u_param <= 1.0)

        # Set invalid to max_range so they don't win the min
        t_param = np.where(valid, t_param, max_range + 1.0)

        # Best wall hit per ray: (M, R)
        wall_best_t = t_param.min(axis=-1)

        hit_wall = wall_best_t < best_t
        best_t = np.minimum(best_t, wall_best_t)

        if config.two_channel:
            best_hit_type = np.where(hit_wall, HIT_WALL, best_hit_type)

    # --- Ray-agent ellipse intersections (vectorized) ---
    n_agents = world.n_agents
    if n_agents > 1:
        # Determine which agents to test as targets
        if world.active_mask is not None:
            target_mask = world.active_mask.copy()
        else:
            target_mask = np.ones(n_agents, dtype=np.bool_)

        target_idx = np.where(target_mask)[0]
        T = len(target_idx)

        if T > 0:
            target_pos = world.positions[target_idx]  # (T, 2)
            target_widths = world.shoulder_widths[target_idx]  # (T,)
            target_depths = world.chest_depths[target_idx]  # (T,)
            target_angles = world.torso_orientations[target_idx]  # (T,)

            # For each sensing agent, exclude self from targets
            # Build self-exclusion mask: (M, T) — True means "test this target"
            self_exclude = agent_indices[:, np.newaxis] != target_idx[np.newaxis, :]

            # Distance pre-filter: only test agents within max_range
            agent_origins = world.positions[agent_indices]  # (M, 2)
            dist_to_target = np.sqrt(
                np.sum(
                    (agent_origins[:, np.newaxis, :] - target_pos[np.newaxis, :, :]) ** 2, axis=-1
                )
            )  # (M, T)
            in_range = dist_to_target < (
                max_range + np.maximum(target_widths, target_depths)[np.newaxis, :]
            )
            test_mask = self_exclude & in_range  # (M, T)

            # Transform all ray origins into each target ellipse's local frame
            # and solve the quadratic for all (agent, ray, target) combinations

            cos_t = np.cos(-target_angles)  # (T,)
            sin_t = np.sin(-target_angles)  # (T,)

            # Origin relative to ellipse centre: (M, T, 2)
            rel_origin = agent_origins[:, np.newaxis, :] - target_pos[np.newaxis, :, :]

            # Rotate into ellipse-local frame: (M, T, 2)
            rot_ox = (
                cos_t[np.newaxis, :] * rel_origin[..., 0]
                - sin_t[np.newaxis, :] * rel_origin[..., 1]
            )
            rot_oy = (
                sin_t[np.newaxis, :] * rel_origin[..., 0]
                + cos_t[np.newaxis, :] * rel_origin[..., 1]
            )

            # Scale to unit circle: (M, T, 2)
            origin_sx = rot_ox / target_depths[np.newaxis, :]
            origin_sy = rot_oy / target_widths[np.newaxis, :]

            # Rotate ray directions into ellipse-local frame: (M, R, T, 2)
            # ray_dirs is (M, R, 2), target rotation params are (T,)
            rd_x = ray_dirs[..., 0]  # (M, R)
            rd_y = ray_dirs[..., 1]  # (M, R)

            rot_dx = (
                cos_t[np.newaxis, np.newaxis, :] * rd_x[:, :, np.newaxis]
                - sin_t[np.newaxis, np.newaxis, :] * rd_y[:, :, np.newaxis]
            )  # (M, R, T)
            rot_dy = (
                sin_t[np.newaxis, np.newaxis, :] * rd_x[:, :, np.newaxis]
                + cos_t[np.newaxis, np.newaxis, :] * rd_y[:, :, np.newaxis]
            )  # (M, R, T)

            # Scale ray directions: (M, R, T)
            dir_sx = rot_dx / target_depths[np.newaxis, np.newaxis, :]
            dir_sy = rot_dy / target_widths[np.newaxis, np.newaxis, :]

            # Quadratic coefficients: (M, R, T)
            # origin_s is (M, T) — broadcast to (M, R, T) by adding ray axis
            o_sx = origin_sx[:, np.newaxis, :]  # (M, 1, T)
            o_sy = origin_sy[:, np.newaxis, :]  # (M, 1, T)

            qa = dir_sx**2 + dir_sy**2
            qb = 2.0 * (o_sx * dir_sx + o_sy * dir_sy)
            qc = o_sx**2 + o_sy**2 - 1.0

            discriminant = qb**2 - 4.0 * qa * qc

            # Solve for both roots
            sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))
            safe_qa = np.where(np.abs(qa) < 1e-15, 1.0, qa)
            t1 = (-qb - sqrt_disc) / (2.0 * safe_qa)
            t2 = (-qb + sqrt_disc) / (2.0 * safe_qa)

            eps = 1e-8
            # Pick nearest positive root
            t1_valid = (discriminant >= 0) & (t1 > eps)
            t2_valid = (discriminant >= 0) & (t2 > eps)

            # Start with t1 where valid, else t2, else large value
            agent_t = np.where(t1_valid, t1, np.where(t2_valid, t2, max_range + 1.0))

            # Apply self-exclusion and range masks: (M, R, T)
            test_mask_3d = test_mask[:, np.newaxis, :]  # (M, 1, T) → broadcast to (M, R, T)
            agent_t = np.where(test_mask_3d, agent_t, max_range + 1.0)

            # Best agent hit per ray: (M, R)
            agent_best_t = agent_t.min(axis=-1)

            hit_agent = agent_best_t < best_t
            best_t = np.minimum(best_t, agent_best_t)

            if config.two_channel:
                best_hit_type = np.where(hit_agent, HIT_AGENT, best_hit_type)

    # Normalize
    normalised = np.minimum(best_t / max_range, 1.0)

    if config.two_channel:
        readings = np.stack([normalised, best_hit_type], axis=-1)  # (M, R, 2)
        # No hit → HIT_NONE
        readings[..., 1] = np.where(normalised < 1.0, readings[..., 1], HIT_NONE)
        return readings
    else:
        return normalised


def knn_social(
    world: WorldState,
    agent_idx: int,
    k: int = 8,
) -> NDArray[np.float64]:
    """K-nearest-neighbour social sensing in the ego agent's reference frame.

    For each of the K nearest neighbours, returns:
    - Relative position (2D) — in ego frame
    - Relative velocity (2D) — in ego frame
    - Body orientation (1D) — relative to ego heading
    - Body dimensions (2D) — shoulder_width, chest_depth

    Total: K × 7 features. If fewer than K neighbours exist, the remaining
    slots are zero-padded.

    Parameters
    ----------
    world : WorldState
    agent_idx : int
    k : int

    Returns
    -------
    social : (K, 7) array
    """
    ego_pos = world.positions[agent_idx]
    ego_vel = world.velocities[agent_idx]
    ego_heading = world.torso_orientations[agent_idx]

    # Rotation matrix from global to ego frame
    cos_h, sin_h = np.cos(-ego_heading), np.sin(-ego_heading)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float64)

    # Compute distances to all other active agents
    candidates = []
    for j in range(world.n_agents):
        if j == agent_idx:
            continue
        if world.active_mask is not None and not world.active_mask[j]:
            continue
        dist = float(np.linalg.norm(world.positions[j] - ego_pos))
        candidates.append((dist, j))

    # Sort by distance, take K nearest
    candidates.sort(key=lambda x: x[0])
    nearest = candidates[:k]

    social = np.zeros((k, 7), dtype=np.float64)
    for i, (_, j) in enumerate(nearest):
        # Relative position in ego frame
        rel_pos = rot @ (world.positions[j] - ego_pos)
        # Relative velocity in ego frame
        rel_vel = rot @ (world.velocities[j] - ego_vel)
        # Relative body orientation
        rel_orient = world.torso_orientations[j] - ego_heading
        # Normalise to [-pi, pi]
        rel_orient = (rel_orient + np.pi) % (2 * np.pi) - np.pi

        social[i, 0:2] = rel_pos
        social[i, 2:4] = rel_vel
        social[i, 4] = rel_orient
        social[i, 5] = world.shoulder_widths[j]
        social[i, 6] = world.chest_depths[j]

    return social


def knn_social_batch(
    world: WorldState,
    agent_indices: NDArray[np.intp],
    k: int = 8,
) -> NDArray[np.float64]:
    """Batch K-nearest-neighbour social sensing for multiple agents.

    Parameters
    ----------
    world : WorldState
    agent_indices : (M,) array of agent indices
    k : int

    Returns
    -------
    social : (M, K, 7) array
    """
    M = len(agent_indices)
    n = world.n_agents

    if n < 2 or M == 0:
        return np.zeros((M, k, 7), dtype=np.float64)

    # Determine active agents for neighbor consideration
    if world.active_mask is not None:
        active_mask = world.active_mask
    else:
        active_mask = np.ones(n, dtype=np.bool_)

    ego_pos = world.positions[agent_indices]  # (M, 2)
    ego_vel = world.velocities[agent_indices]  # (M, 2)
    ego_heading = world.torso_orientations[agent_indices]  # (M,)

    # Pairwise distances from each ego to all agents: (M, N)
    all_pos = world.positions  # (N, 2)
    dist_sq = np.sum((ego_pos[:, np.newaxis, :] - all_pos[np.newaxis, :, :]) ** 2, axis=-1)

    # Mask out self and inactive agents
    self_mask = agent_indices[:, np.newaxis] == np.arange(n)[np.newaxis, :]
    inactive_mask = ~active_mask[np.newaxis, :]
    dist_sq = np.where(self_mask | inactive_mask, np.inf, dist_sq)

    # Find K nearest using argpartition (O(N) per row vs O(N log N) for sort)
    actual_k = min(k, n - 1)
    if actual_k <= 0:
        return np.zeros((M, k, 7), dtype=np.float64)

    # argpartition: get indices of K smallest distances
    if actual_k < n:
        knn_idx = np.argpartition(dist_sq, actual_k, axis=1)[:, :actual_k]  # (M, K)
        # Sort the K nearest by distance for consistent ordering
        knn_dists = np.take_along_axis(dist_sq, knn_idx, axis=1)
        sort_order = np.argsort(knn_dists, axis=1)
        knn_idx = np.take_along_axis(knn_idx, sort_order, axis=1)
    else:
        knn_idx = np.argsort(dist_sq, axis=1)[:, :actual_k]

    # Build rotation matrices for ego frames: (M, 2, 2)
    cos_h = np.cos(-ego_heading)
    sin_h = np.sin(-ego_heading)

    # Gather neighbor data using fancy indexing: (M, K, ...)
    # Pad knn_idx to k columns if actual_k < k
    if actual_k < k:
        pad_idx = np.zeros((M, k - actual_k), dtype=np.intp)
        knn_idx_full = np.concatenate([knn_idx, pad_idx], axis=1)
        valid_mask = np.zeros((M, k), dtype=np.bool_)
        valid_mask[:, :actual_k] = True
        # Also check that distance is not inf
        knn_dists_full = np.take_along_axis(dist_sq, knn_idx_full, axis=1)
        valid_mask &= np.isfinite(knn_dists_full)
    else:
        knn_idx_full = knn_idx
        knn_dists_full = np.take_along_axis(dist_sq, knn_idx_full, axis=1)
        valid_mask = np.isfinite(knn_dists_full)

    # Gather neighbor positions, velocities, orientations, body dims
    nb_pos = world.positions[knn_idx_full]  # (M, K, 2)
    nb_vel = world.velocities[knn_idx_full]  # (M, K, 2)
    nb_orient = world.torso_orientations[knn_idx_full]  # (M, K)
    nb_widths = world.shoulder_widths[knn_idx_full]  # (M, K)
    nb_depths = world.chest_depths[knn_idx_full]  # (M, K)

    # Relative positions in global frame: (M, K, 2)
    rel_pos_global = nb_pos - ego_pos[:, np.newaxis, :]
    rel_vel_global = nb_vel - ego_vel[:, np.newaxis, :]

    # Rotate to ego frame: (M, K, 2)
    rel_pos_x = (
        cos_h[:, np.newaxis] * rel_pos_global[..., 0]
        - sin_h[:, np.newaxis] * rel_pos_global[..., 1]
    )
    rel_pos_y = (
        sin_h[:, np.newaxis] * rel_pos_global[..., 0]
        + cos_h[:, np.newaxis] * rel_pos_global[..., 1]
    )

    rel_vel_x = (
        cos_h[:, np.newaxis] * rel_vel_global[..., 0]
        - sin_h[:, np.newaxis] * rel_vel_global[..., 1]
    )
    rel_vel_y = (
        sin_h[:, np.newaxis] * rel_vel_global[..., 0]
        + cos_h[:, np.newaxis] * rel_vel_global[..., 1]
    )

    # Relative orientation normalized to [-pi, pi]
    rel_orient = (nb_orient - ego_heading[:, np.newaxis] + np.pi) % (2 * np.pi) - np.pi

    # Assemble: (M, K, 7)
    social = np.zeros((M, k, 7), dtype=np.float64)
    social[..., 0] = np.where(valid_mask, rel_pos_x, 0.0)
    social[..., 1] = np.where(valid_mask, rel_pos_y, 0.0)
    social[..., 2] = np.where(valid_mask, rel_vel_x, 0.0)
    social[..., 3] = np.where(valid_mask, rel_vel_y, 0.0)
    social[..., 4] = np.where(valid_mask, rel_orient, 0.0)
    social[..., 5] = np.where(valid_mask, nb_widths, 0.0)
    social[..., 6] = np.where(valid_mask, nb_depths, 0.0)

    return social


def match_persistent_neighbors(
    positions: NDArray[np.float64],
    prev_slots: NDArray[np.int32],
    active_mask: NDArray[np.bool_],
    sensing_radius: float,
    k: int,
) -> NDArray[np.int32]:
    """Greedy persistent-neighbor matching.

    For each ego agent, update the K-slot neighbor-ID table so that

    1. A previously-assigned neighbor keeps its slot if and only if it is
       still active and still within ``sensing_radius`` of the ego agent.
    2. Slots that became empty (either because the prior assignee moved
       out of range / deactivated, or because they were empty at the start)
       are filled with the nearest currently-in-range active agent that
       is not already in another slot of the same ego.

    This preserves neighbor identity across timesteps so long as the
    neighbor stays visible, which is the prerequisite for any per-neighbor
    temporal memory feature (velocity history, trajectory features, etc.).

    Parameters
    ----------
    positions : (N, 2) float64 -- current agent positions.
    prev_slots : (N, K) int32 -- previous step's neighbor-ID table, or
        an all-``-1`` table on the first step.
    active_mask : (N,) bool -- True for agents still active in the episode.
    sensing_radius : float -- metres. Previously-assigned neighbors beyond
        this distance are evicted; newly filled slots only consider
        candidates within this range.
    k : int -- number of slots per ego agent.

    Returns
    -------
    new_slots : (N, K) int32 -- updated assignment table.
    """
    n = positions.shape[0]
    new_slots = np.full((n, k), -1, dtype=np.int32)

    # Pairwise squared distances: (N, N). Diagonal set to +inf so an agent
    # never becomes its own neighbor.
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)

    # Mask out inactive agents as candidates
    dist_sq[:, ~active_mask] = np.inf

    # Also mask rows where ego itself is inactive (those will return all -1)
    radius_sq = sensing_radius * sensing_radius

    for i in range(n):
        if not active_mask[i]:
            continue  # leave row as all -1

        # Step 1: retain eligible prev slots
        for s in range(k):
            prev = int(prev_slots[i, s])
            if prev < 0 or prev >= n:
                continue
            if not active_mask[prev]:
                continue
            if dist_sq[i, prev] <= radius_sq:
                new_slots[i, s] = prev

        # Step 2: fill empty slots with nearest unassigned in-range candidate
        assigned: set[int] = {int(x) for x in new_slots[i] if x >= 0}

        for s in range(k):
            if new_slots[i, s] >= 0:
                continue

            best_j = -1
            best_d = np.inf
            for j in range(n):
                if j == i or not active_mask[j] or j in assigned:
                    continue
                d = float(dist_sq[i, j])
                if d <= radius_sq and d < best_d:
                    best_d = d
                    best_j = j

            if best_j >= 0:
                new_slots[i, s] = best_j
                assigned.add(best_j)

    return new_slots
