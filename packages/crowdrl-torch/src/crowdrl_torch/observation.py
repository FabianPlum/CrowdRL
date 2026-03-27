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
) -> Tensor:
    """Build observations for all agents.

    Parameters
    ----------
    positions : (E, N, 2)
    ...

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

    # --- Concatenate ---
    obs = torch.cat([ego_state, social_flat, rays], dim=-1)  # (E, N, obs_dim)

    # Zero out inactive agents
    obs = torch.where(active_mask.unsqueeze(-1), obs, torch.zeros_like(obs))

    return obs
