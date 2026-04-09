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
    agent_distances: Tensor | None = None,
    agent_radii: Tensor | None = None,
    actions: Tensor | None = None,
    prev_actions: Tensor | None = None,
    headings: Tensor | None = None,
    preferred_speeds: Tensor | None = None,
    prev_velocities: Tensor | None = None,
    prev_accelerations: Tensor | None = None,
    prev_headings: Tensor | None = None,
    prev_heading_changes: Tensor | None = None,
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
    wall_distances : (E, N) optional -- min distance to nearest wall per agent
    agent_radii : (E, N) optional -- agent body radii
    actions : (E, N, 4) optional -- raw policy output this step
    prev_actions : (E, N, 4) optional -- raw policy output previous step
    headings : (E, N) optional -- current torso orientations (for angular accel)
    preferred_speeds : (E, N) optional -- preferred walking speeds
    prev_velocities : (E, N, 2) optional -- velocities from previous step
    prev_accelerations : (E, N, 2) optional -- accelerations from previous step
    prev_headings : (E, N) optional -- headings from previous step
    prev_heading_changes : (E, N) optional -- heading changes from previous step

    Returns
    -------
    rewards : (E, N)
    reached_goal : (E, N) bool
    new_goal_distances : (E, N) -- for next step's progress reward
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

    # Agent proximity penalty (learned collision avoidance signal)
    if (
        config.agent_proximity_penalty != 0.0
        and agent_distances is not None
        and agent_radii is not None
    ):
        threshold = agent_radii * config.agent_proximity_threshold
        too_close = (agent_distances < threshold) & active_mask
        rewards = torch.where(
            too_close,
            rewards + config.agent_proximity_penalty,
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

    # --- Tier 2: Smoothness ---
    if config.use_smoothness and prev_velocities is not None:
        dt = config.dt
        accelerations = (velocities - prev_velocities) / dt  # (E, N, 2)

        # Jerk penalty (change in acceleration)
        if config.jerk_penalty_weight != 0.0 and prev_accelerations is not None:
            jerk = (accelerations - prev_accelerations) / dt  # (E, N, 2)
            jerk_mag = (jerk**2).sum(dim=-1).sqrt()  # (E, N)
            rewards = torch.where(
                active_mask,
                rewards + config.jerk_penalty_weight * jerk_mag,
                rewards,
            )

        # Angular acceleration penalty
        if (
            config.angular_accel_penalty_weight != 0.0
            and headings is not None
            and prev_headings is not None
            and prev_heading_changes is not None
        ):
            heading_change = headings - prev_headings
            # Normalise to [-pi, pi]
            heading_change = (heading_change + torch.pi) % (2 * torch.pi) - torch.pi
            angular_vel = heading_change / dt
            prev_angular_vel = prev_heading_changes / dt
            angular_accel = (angular_vel - prev_angular_vel).abs()
            rewards = torch.where(
                active_mask,
                rewards + config.angular_accel_penalty_weight * angular_accel,
                rewards,
            )

        # Preferred speed deviation
        if config.speed_deviation_weight != 0.0 and preferred_speeds is not None:
            speeds = (velocities**2).sum(dim=-1).sqrt()  # (E, N)
            speed_dev = (speeds - preferred_speeds).abs()
            rewards = torch.where(
                active_mask,
                rewards + config.speed_deviation_weight * speed_dev,
                rewards,
            )

    # Zero rewards for inactive agents
    rewards = torch.where(active_mask, rewards, torch.zeros_like(rewards))

    return rewards, reached_goal, goal_distances
