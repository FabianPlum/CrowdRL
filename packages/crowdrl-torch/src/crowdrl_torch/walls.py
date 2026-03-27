"""Wall boundary enforcement in PyTorch — no Shapely dependency.

Replaces ``crowdrl_core.collision.enforce_wall_boundaries`` which uses
Shapely's ``polygon.contains()`` and ``nearest_points()``.

Uses:
- Ray-crossing (even-odd) point-in-polygon test against segment arrays
- Vectorised nearest-point-on-segment projection

Shapes carry a leading (E,) environment batch dimension throughout.
"""

from __future__ import annotations

import torch
from torch import Tensor

from crowdrl_torch.types import EnvConfig


def points_to_segments_nearest(
    points: Tensor,
    seg_starts: Tensor,
    seg_ends: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Batch nearest-point-on-segment for all points x all segments.

    Parameters
    ----------
    points : (E, N, 2)
    seg_starts : (E, S, 2)
    seg_ends : (E, S, 2)

    Returns
    -------
    nearest : (E, N, S, 2)
    distances : (E, N, S)
    normals : (E, N, S, 2) — unit vectors from nearest point toward agent
    """
    edges = seg_ends - seg_starts  # (E, S, 2)
    edge_len_sq = (edges**2).sum(dim=-1)  # (E, S)

    # diff from each point to each segment start: (E, N, S, 2)
    diff = points.unsqueeze(2) - seg_starts.unsqueeze(1)

    # Project: t = dot(diff, edge) / edge_len_sq, clamped to [0, 1]
    dot_prod = (diff * edges.unsqueeze(1)).sum(dim=-1)  # (E, N, S)
    safe_len_sq = torch.where(edge_len_sq < 1e-12, torch.ones_like(edge_len_sq), edge_len_sq)
    t = torch.clamp(dot_prod / safe_len_sq.unsqueeze(1), 0.0, 1.0)
    t = torch.where(edge_len_sq.unsqueeze(1) < 1e-12, torch.zeros_like(t), t)

    # Nearest points: (E, N, S, 2)
    nearest = seg_starts.unsqueeze(1) + t.unsqueeze(-1) * edges.unsqueeze(1)

    # Vectors from nearest to point: (E, N, S, 2)
    to_point = points.unsqueeze(2) - nearest
    distances = (to_point**2).sum(dim=-1).sqrt()  # (E, N, S)

    # Normals: from nearest toward point
    safe_dist = torch.where(distances < 1e-10, torch.ones_like(distances), distances)
    normals = to_point / safe_dist.unsqueeze(-1)

    # For near-zero distances, use edge perpendicular
    edge_normals = torch.stack([-edges[..., 1], edges[..., 0]], dim=-1)  # (E, S, 2)
    edge_norm_len = (edge_normals**2).sum(dim=-1, keepdim=True).sqrt()
    safe_edge_len = torch.where(
        edge_norm_len < 1e-12, torch.ones_like(edge_norm_len), edge_norm_len
    )
    edge_normals = edge_normals / safe_edge_len

    close_mask = distances < 1e-10  # (E, N, S)
    normals = torch.where(
        close_mask.unsqueeze(-1),
        edge_normals.unsqueeze(1),
        normals,
    )

    return nearest, distances, normals


def point_in_polygon(
    points: Tensor,
    wall_segments: Tensor,
    n_segments: Tensor,
) -> Tensor:
    """Ray-crossing (even-odd) point-in-polygon test.

    Parameters
    ----------
    points : (E, N, 2)
    wall_segments : (E, S, 2, 2) — [[x1,y1],[x2,y2]]
    n_segments : (E,) int

    Returns
    -------
    inside : (E, N) bool
    """
    S = wall_segments.shape[1]

    px = points[..., 0]  # (E, N)
    py = points[..., 1]  # (E, N)

    # Segment endpoints: (E, S)
    x1 = wall_segments[:, :, 0, 0]
    y1 = wall_segments[:, :, 0, 1]
    x2 = wall_segments[:, :, 1, 0]
    y2 = wall_segments[:, :, 1, 1]

    # Broadcast to (E, N, S)
    px_ = px.unsqueeze(2)
    py_ = py.unsqueeze(2)
    x1_ = x1.unsqueeze(1)
    y1_ = y1.unsqueeze(1)
    x2_ = x2.unsqueeze(1)
    y2_ = y2.unsqueeze(1)

    # Ray-crossing test
    cond_a = (y1_ <= py_) & (y2_ > py_)  # upward crossing
    cond_b = (y2_ <= py_) & (y1_ > py_)  # downward crossing
    straddles = cond_a | cond_b

    # Compute x-coordinate of intersection
    dy = y2_ - y1_
    safe_dy = torch.where(dy.abs() < 1e-12, torch.ones_like(dy), dy)
    t = (py_ - y1_) / safe_dy
    x_intersect = x1_ + t * (x2_ - x1_)

    # Crossing: segment straddles AND intersection is to the right
    crossing = straddles & (x_intersect > px_)

    # Mask out padding segments: (E, 1, S) broadcast
    seg_idx = torch.arange(S, device=wall_segments.device)
    seg_mask = seg_idx.unsqueeze(0) < n_segments.unsqueeze(1)  # (E, S)
    crossing = crossing & seg_mask.unsqueeze(1)

    # Odd number of crossings = inside
    n_crossings = crossing.sum(dim=2)  # (E, N)
    inside = (n_crossings % 2) == 1

    return inside


def enforce_wall_boundaries(
    positions: Tensor,
    velocities: Tensor,
    shoulder_widths: Tensor,
    chest_depths: Tensor,
    active_mask: Tensor,
    wall_segments: Tensor,
    n_segments: Tensor,
    config: EnvConfig,
) -> tuple[Tensor, Tensor]:
    """Project agents to stay inside the walkable polygon.

    Parameters
    ----------
    positions : (E, N, 2)
    velocities : (E, N, 2)

    Returns
    -------
    new_positions : (E, N, 2)
    new_velocities : (E, N, 2)
    """
    S = wall_segments.shape[1]
    seg_starts = wall_segments[:, :, 0, :]  # (E, S, 2)
    seg_ends = wall_segments[:, :, 1, :]  # (E, S, 2)

    # Check which agents are inside the polygon
    inside = point_in_polygon(positions, wall_segments, n_segments)  # (E, N)

    # Nearest point on boundary for each agent
    nearest, distances, normals_to_agent = points_to_segments_nearest(
        positions, seg_starts, seg_ends
    )  # (E, N, S, 2), (E, N, S), (E, N, S, 2)

    # Mask padding segments with large distance
    seg_mask = torch.arange(S, device=wall_segments.device).unsqueeze(0) < n_segments.unsqueeze(1)
    distances = torch.where(
        seg_mask.unsqueeze(1), distances, torch.tensor(1e10, device=distances.device)
    )

    # Find closest segment per agent
    closest_seg = distances.argmin(dim=2)  # (E, N)
    E, N = positions.shape[:2]
    e_idx = torch.arange(E, device=positions.device).unsqueeze(1).expand(E, N)
    n_idx = torch.arange(N, device=positions.device).unsqueeze(0).expand(E, N)
    min_dist = distances[e_idx, n_idx, closest_seg]  # (E, N)
    closest_nearest = nearest[e_idx, n_idx, closest_seg]  # (E, N, 2)
    closest_normal = normals_to_agent[e_idx, n_idx, closest_seg]  # (E, N, 2)

    # Agent radii
    radii = torch.maximum(shoulder_widths, chest_depths)  # (E, N)

    # Agents that need correction: outside polygon OR too close to wall
    needs_fix = active_mask & (~inside | (min_dist < radii))

    # Inward normal (toward polygon interior)
    inward = torch.where(inside.unsqueeze(-1), closest_normal, -closest_normal)

    # Project to: nearest_point + inward * radius
    corrected_pos = closest_nearest + inward * radii.unsqueeze(-1)

    new_positions = torch.where(needs_fix.unsqueeze(-1), corrected_pos, positions)

    # Cancel velocity component into wall
    vel_into_wall = (velocities * (-inward)).sum(dim=-1)  # (E, N)
    vel_correction = torch.clamp(vel_into_wall, min=0.0).unsqueeze(-1) * inward
    corrected_vel = velocities + vel_correction
    new_velocities = torch.where(needs_fix.unsqueeze(-1), corrected_vel, velocities)

    return new_positions, new_velocities
