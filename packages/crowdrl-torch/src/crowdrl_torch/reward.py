"""Reward computation in PyTorch.

Port of ``crowdrl_env.reward.compute_rewards``.
Temporal state (prev_velocities etc.) is part of TorchWorldState,
not a separate mutable object.

Shapes carry a leading (E,) environment batch dimension throughout.
"""

from __future__ import annotations

import torch
from torch import Tensor

from crowdrl_torch.types import EnvConfig


def compute_rewards(
    positions: Tensor,
    velocities: Tensor,
    goal_positions: Tensor,
    active_mask: Tensor,
    collision_mask: Tensor,
    prev_goal_distances: Tensor,
    config: EnvConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute per-agent rewards for one timestep.

    Parameters
    ----------
    positions : (E, N, 2)
    velocities : (E, N, 2)
    goal_positions : (E, N, 2)
    active_mask : (E, N) bool
    collision_mask : (E, N) bool
    prev_goal_distances : (E, N)

    Returns
    -------
    rewards : (E, N)
    reached_goal : (E, N) bool
    new_goal_distances : (E, N) — for next step's progress reward
    """
    # Goal distances
    goal_diffs = goal_positions - positions
    goal_distances = (goal_diffs**2).sum(dim=-1).sqrt()  # (E, N)

    rewards = torch.zeros_like(goal_distances)

    # Goal reaching
    reached_goal = (goal_distances < config.goal_radius) & active_mask
    rewards = torch.where(reached_goal, rewards + config.goal_bonus, rewards)

    # Collision penalty
    rewards = torch.where(
        collision_mask & active_mask,
        rewards + config.collision_penalty,
        rewards,
    )

    # Progress reward (potential-based shaping)
    progress = prev_goal_distances - goal_distances
    rewards = torch.where(active_mask, rewards + config.progress_weight * progress, rewards)

    # Zero rewards for inactive agents
    rewards = torch.where(active_mask, rewards, torch.zeros_like(rewards))

    return rewards, reached_goal, goal_distances
