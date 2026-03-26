"""Sensing: raycast engine and K-nearest-neighbour social query.

Raycasts are head-anchored (FOV follows head orientation, not torso).
Social sensing returns K nearest neighbours' relative state.

Both operate on WorldState — agnostic to whether state comes from
the training environment or from JuPedSim.
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
                origin, direction,
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
