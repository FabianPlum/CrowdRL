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
    *,
    wall_distances: Tensor | None = None,
    agent_radii: Tensor | None = None,
    actions: Tensor | None = None,
    prev_actions: Tensor | None = None,
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
    config : EnvConfig
    wall_distances : (E, N) optional — min distance to nearest wall per agent
    agent_radii : (E, N) optional — agent body radii
    actions : (E, N, 4) optional — raw policy output this step
    prev_actions : (E, N, 4) optional — raw policy output previous step

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

    # Wall proximity penalty (smooth, distance-based)
    if (
        config.wall_proximity_penalty != 0.0
        and wall_distances is not None
        and agent_radii is not None
    ):
        threshold = agent_radii * config.wall_proximity_threshold
        wall_proximity = (wall_distances < threshold) & active_mask
        rewards = torch.where(
            wall_proximity,
            rewards + config.wall_proximity_penalty,
            rewards,
        )

    # Action rate penalty (change in raw policy output between steps)
    if config.action_rate_weight != 0.0 and actions is not None and prev_actions is not None:
        action_change = ((actions - prev_actions) ** 2).sum(dim=-1).sqrt()  # (E, N)
        rewards = torch.where(
            active_mask,
            rewards + config.action_rate_weight * action_change,
            rewards,
        )

    # Existence penalty: every step alive costs you
    if config.existence_penalty != 0.0:
        rewards = torch.where(
            active_mask,
            rewards + config.existence_penalty,
            rewards,
        )

    # Progress reward (potential-based shaping)
    progress = prev_goal_distances - goal_distances
    rewards = torch.where(active_mask, rewards + config.progress_weight * progress, rewards)

    # Zero rewards for inactive agents
    rewards = torch.where(active_mask, rewards, torch.zeros_like(rewards))

    return rewards, reached_goal, goal_distances
