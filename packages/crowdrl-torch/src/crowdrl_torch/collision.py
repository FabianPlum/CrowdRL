"""Elliptical collision detection and contact forces in PyTorch.

Uses dense (E, N, N) pairwise computation instead of sparse pair extraction.
With MAX_AGENTS=64 the (E, 64, 64, 2) tensors are ~32KB per env — negligible on GPU.

Shapes carry a leading (E,) environment batch dimension throughout.
"""

from __future__ import annotations

import torch
from torch import Tensor

from crowdrl_torch.types import EnvConfig


def detect_collisions_pairwise(
    positions: Tensor,
    torso_orientations: Tensor,
    shoulder_widths: Tensor,
    chest_depths: Tensor,
    active_mask: Tensor,
    n_agents: Tensor,
) -> tuple[Tensor, Tensor]:
    """Detect all pairwise agent-agent collisions.

    Parameters
    ----------
    positions : (E, N, 2)
    torso_orientations : (E, N)
    shoulder_widths : (E, N)
    chest_depths : (E, N)
    active_mask : (E, N)
    n_agents : (E,)

    Returns
    -------
    overlap_matrix : (E, N, N) — approximate overlap, 0 if no collision
    collision_mask : (E, N) bool — True if agent is in any collision
    """
    N = positions.shape[1]

    # Pairwise displacements: diff[...,i,j] = pos[j] - pos[i], shape (E, N, N, 2)
    diff = positions.unsqueeze(1) - positions.unsqueeze(2)

    # --- Check B centre in A's ellipse frame ---
    cos_a = torch.cos(-torso_orientations)  # (E, N)
    sin_a = torch.sin(-torso_orientations)

    # Rotate diff into A's frame: (E, N, N)
    dx_a = cos_a.unsqueeze(2) * diff[..., 0] - sin_a.unsqueeze(2) * diff[..., 1]
    dy_a = sin_a.unsqueeze(2) * diff[..., 0] + cos_a.unsqueeze(2) * diff[..., 1]

    # Algebraic distance in A's ellipse: (E, N, N)
    dist_a = (dx_a / chest_depths.unsqueeze(2)) ** 2 + (dy_a / shoulder_widths.unsqueeze(2)) ** 2

    # --- Check A centre in B's ellipse frame ---
    neg_diff = -diff
    cos_b = cos_a  # same agents, same orientations
    sin_b = sin_a

    dx_b = cos_b.unsqueeze(1) * neg_diff[..., 0] - sin_b.unsqueeze(1) * neg_diff[..., 1]
    dy_b = sin_b.unsqueeze(1) * neg_diff[..., 0] + cos_b.unsqueeze(1) * neg_diff[..., 1]

    dist_b = (dx_b / chest_depths.unsqueeze(1)) ** 2 + (dy_b / shoulder_widths.unsqueeze(1)) ** 2

    min_dist = torch.minimum(dist_a, dist_b)
    overlap_matrix = torch.where(
        min_dist < 1.0, 1.0 - torch.sqrt(min_dist), torch.zeros_like(min_dist)
    )

    # Mask: no self-collision, only active agents
    i_idx = torch.arange(N, device=positions.device)
    self_mask = i_idx.unsqueeze(0) == i_idx.unsqueeze(1)  # (N, N)
    active_pair = active_mask.unsqueeze(2) & active_mask.unsqueeze(1)  # (E, N, N)
    overlap_matrix = torch.where(
        self_mask.unsqueeze(0) | ~active_pair, torch.zeros_like(overlap_matrix), overlap_matrix
    )

    # Quick distance pre-filter
    pair_dist_sq = (diff**2).sum(dim=-1)  # (E, N, N)
    max_radii = torch.maximum(shoulder_widths, chest_depths)  # (E, N)
    pair_max_range = max_radii.unsqueeze(2) + max_radii.unsqueeze(1)  # (E, N, N)
    too_far = pair_dist_sq > (pair_max_range**2) * 4.0
    overlap_matrix = torch.where(too_far, torch.zeros_like(overlap_matrix), overlap_matrix)

    # Per-agent collision mask
    collision_mask = (overlap_matrix > 0).any(dim=2)  # (E, N)

    return overlap_matrix, collision_mask


def compute_contact_forces(
    positions: Tensor,
    velocities: Tensor,
    shoulder_widths: Tensor,
    chest_depths: Tensor,
    active_mask: Tensor,
    overlap_matrix: Tensor,
    wall_segments: Tensor,
    n_segments: Tensor,
    config: EnvConfig,
) -> Tensor:
    """Compute agent-agent contact forces + wall repulsion.

    Uses dense (E, N, N, 2) pairwise force tensor — no scatter needed.

    Returns
    -------
    forces : (E, N, 2) — net force per agent
    """
    # --- Agent-agent spring-damper forces ---
    # Pairwise displacement: (E, N, N, 2)
    diff = positions.unsqueeze(1) - positions.unsqueeze(2)  # j - i
    dist = (diff**2).sum(dim=-1).sqrt()  # (E, N, N)

    # Normal vectors (fallback to [1,0] for coincident agents)
    safe_dist = torch.where(dist < 1e-10, torch.ones_like(dist), dist)
    normals = diff / safe_dist.unsqueeze(-1)  # (E, N, N, 2)
    fallback = torch.tensor([1.0, 0.0], device=positions.device)
    normals = torch.where(
        (dist < 1e-10).unsqueeze(-1),
        fallback,
        normals,
    )

    # Relative velocity: (E, N, N, 2) — vi - vj
    rel_vel = velocities.unsqueeze(2) - velocities.unsqueeze(1)
    rel_vel_normal = (rel_vel * normals).sum(dim=-1)  # (E, N, N)

    # Force magnitude per pair
    has_overlap = overlap_matrix > 0
    force_mag = config.contact_stiffness * overlap_matrix + torch.where(
        has_overlap,
        config.contact_damping * torch.clamp(rel_vel_normal, min=0.0),
        torch.zeros_like(rel_vel_normal),
    )

    # Pairwise force vectors: (E, N, N, 2)
    force_vectors = force_mag.unsqueeze(-1) * normals

    # Net force on agent i: sum of forces from all j
    agent_forces = -force_vectors.sum(dim=2)  # (E, N, 2)

    # --- Wall repulsion forces ---
    wall_forces = _compute_wall_repulsion(
        positions,
        shoulder_widths,
        chest_depths,
        active_mask,
        wall_segments,
        n_segments,
        config,
    )

    forces = agent_forces + wall_forces

    # Zero forces for inactive agents
    forces = torch.where(active_mask.unsqueeze(-1), forces, torch.zeros_like(forces))

    return forces


def _compute_wall_repulsion(
    positions: Tensor,
    shoulder_widths: Tensor,
    chest_depths: Tensor,
    active_mask: Tensor,
    wall_segments: Tensor,
    n_segments: Tensor,
    config: EnvConfig,
) -> Tensor:
    """Smooth exponential wall repulsion force.

    Returns
    -------
    forces : (E, N, 2)
    """
    from crowdrl_torch.walls import points_to_segments_nearest

    S = wall_segments.shape[1]
    seg_starts = wall_segments[:, :, 0, :]  # (E, S, 2)
    seg_ends = wall_segments[:, :, 1, :]  # (E, S, 2)

    _, distances, normals = points_to_segments_nearest(positions, seg_starts, seg_ends)
    # distances: (E, N, S), normals: (E, N, S, 2)

    # Mask out padding segments
    seg_mask = torch.arange(S, device=wall_segments.device).unsqueeze(0) < n_segments.unsqueeze(1)

    # Agent radii
    radii = torch.maximum(shoulder_widths, chest_depths)  # (E, N)

    # Exponential repulsion: (E, N, S)
    f_mag = config.wall_strength * torch.exp((radii.unsqueeze(2) - distances) / config.wall_range)

    # Mask padding segments
    f_mag = torch.where(seg_mask.unsqueeze(1), f_mag, torch.zeros_like(f_mag))

    # Sum over segments: (E, N, 2)
    forces = (f_mag.unsqueeze(-1) * normals).sum(dim=2)

    return forces
