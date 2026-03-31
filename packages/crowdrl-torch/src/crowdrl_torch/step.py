"""Batched environment step in PyTorch.

The ``batched_step`` function processes all N_ENVS environments in a single
call using tensors with shape (E, N, ...). No ``torch.vmap`` — the batch
dimension is explicit.

Use ``@torch.compile(mode="reduce-overhead")`` for kernel fusion and
CUDA graph support once validated.
"""

from __future__ import annotations

import torch
from torch import Tensor

from crowdrl_torch.action import interpret_actions
from crowdrl_torch.collision import compute_contact_forces, detect_collisions_pairwise
from crowdrl_torch.observation import build_observations
from crowdrl_torch.reward import compute_rewards
from crowdrl_torch.types import EnvConfig, TorchWorldState
from crowdrl_torch.walls import compute_min_wall_distances, enforce_wall_boundaries


def batched_step(
    state: TorchWorldState,
    actions: Tensor,
    config: EnvConfig,
) -> tuple[TorchWorldState, Tensor, Tensor, Tensor, Tensor]:
    """Batched environment step for all envs.

    Parameters
    ----------
    state : TorchWorldState — (E, N, ...) tensors
    actions : (E, N, 4) raw policy output
    config : EnvConfig

    Returns
    -------
    new_state : TorchWorldState
    observations : (E, N, obs_dim)
    rewards : (E, N)
    terminated : (E, N) bool — agent reached goal
    truncated : (E, N) bool — episode time limit
    """
    E, N = state.positions.shape[:2]
    step_count = state.step_count + 1  # (E,)

    # --- 1. Interpret actions ---
    desired_velocities, new_headings, new_torsos, new_heads = interpret_actions(
        actions,
        state.torso_orientations,
        state.torso_orientations,
        state.head_orientations,
        config,
    )

    # --- 2. Velocity blending (damped) ---
    mask_2d = state.active_mask.unsqueeze(-1)
    new_velocities = torch.where(
        mask_2d,
        config.velocity_damping * desired_velocities
        + (1.0 - config.velocity_damping) * state.velocities,
        state.velocities,
    )
    new_torso_orientations = torch.where(state.active_mask, new_torsos, state.torso_orientations)
    new_head_orientations = torch.where(state.active_mask, new_heads, state.head_orientations)

    # --- 3. Collision detection ---
    overlap_matrix, collision_mask = detect_collisions_pairwise(
        state.positions,
        new_torso_orientations,
        state.shoulder_widths,
        state.chest_depths,
        state.active_mask,
        state.n_agents,
    )

    # --- 4. Contact forces ---
    forces = compute_contact_forces(
        state.positions,
        new_velocities,
        state.shoulder_widths,
        state.chest_depths,
        state.active_mask,
        overlap_matrix,
        state.wall_segments,
        state.n_segments,
        config,
    )

    # Apply contact accelerations (implicit unit mass) as velocity impulse
    new_velocities = torch.where(
        mask_2d,
        new_velocities + forces * config.dt,
        new_velocities,
    )

    # Clamp velocity magnitudes to prevent contact-force blow-up
    max_vel = config.max_speed_multiplier * config.max_speed
    speeds = (new_velocities**2).sum(dim=-1, keepdim=True).sqrt()
    scale = torch.where(
        speeds > max_vel, max_vel / torch.clamp(speeds, min=1e-10), torch.ones_like(speeds)
    )
    new_velocities = new_velocities * scale

    # --- 5. Position update ---
    new_positions = torch.where(
        mask_2d,
        state.positions + new_velocities * config.dt,
        state.positions,
    )

    # --- 6. Wall boundary enforcement ---
    new_positions, new_velocities = enforce_wall_boundaries(
        new_positions,
        new_velocities,
        state.shoulder_widths,
        state.chest_depths,
        state.active_mask,
        state.wall_segments,
        state.n_segments,
        config,
    )

    # --- 7. Rewards ---
    # Compute wall distances for proximity penalty
    wall_distances = compute_min_wall_distances(
        new_positions, state.wall_segments, state.n_segments
    )
    agent_radii = torch.maximum(state.shoulder_widths, state.chest_depths)

    rewards, reached_goal, new_goal_distances = compute_rewards(
        new_positions,
        new_velocities,
        state.goal_positions,
        state.active_mask,
        collision_mask,
        state.prev_goal_distances,
        config,
        wall_distances=wall_distances,
        agent_radii=agent_radii,
        actions=actions,
        prev_actions=state.prev_actions,
    )

    # --- 8. Update active mask ---
    newly_done = reached_goal & state.active_mask
    new_cumulative_terminated = state.cumulative_terminated | newly_done
    new_active_mask = state.active_mask & ~newly_done

    # Zero velocities for inactive agents
    new_velocities = torch.where(
        new_active_mask.unsqueeze(-1), new_velocities, torch.zeros_like(new_velocities)
    )

    # --- 9. Termination / truncation ---
    terminated = reached_goal
    truncated = torch.zeros(E, N, dtype=torch.bool, device=state.positions.device)

    # Check for timeout: (E,) -> broadcast to (E, N)
    is_timeout = (step_count >= config.max_steps).unsqueeze(1)  # (E, 1)
    truncated = torch.where(is_timeout & new_active_mask, torch.ones_like(truncated), truncated)
    rewards = torch.where(is_timeout & new_active_mask, rewards + config.timeout_penalty, rewards)
    new_active_mask = torch.where(
        is_timeout.expand_as(new_active_mask),
        torch.zeros_like(new_active_mask),
        new_active_mask,
    )

    # --- 10. Advance waypoint cursor ---
    # An agent's cursor advances when it gets close enough to the current waypoint.
    # This is a monotonic index — once passed, a waypoint is never reconsidered.
    if config.use_navmesh:
        wp_cursor = state.waypoint_cursor.long()
        wp_max_idx = (state.n_waypoints.long() - 1).clamp(min=0)
        cur_idx = wp_cursor.clamp(min=0, max=config.max_waypoints - 1).clamp(max=wp_max_idx)

        # Gather current waypoint position: (E, N, 2)
        gather_idx = cur_idx.unsqueeze(-1).unsqueeze(-1).expand(E, N, 1, 2)
        cur_wp = state.waypoints.gather(2, gather_idx).squeeze(2)

        # Distance to current waypoint
        dist_to_wp = ((new_positions - cur_wp) ** 2).sum(dim=-1).sqrt()

        # Advance cursor where agent is within crossing threshold
        advance = (dist_to_wp < config.waypoint_crossing_threshold) & new_active_mask
        new_wp_cursor = torch.where(advance, wp_cursor + 1, wp_cursor)
        new_wp_cursor = new_wp_cursor.clamp(max=wp_max_idx).to(state.waypoint_cursor.dtype)
    else:
        new_wp_cursor = state.waypoint_cursor

    # --- 11. Update reward temporal state ---
    has_prev = state.prev_velocities.any(dim=-1, keepdim=True)
    new_prev_accelerations = torch.where(
        has_prev,
        (new_velocities - state.prev_velocities) / config.dt,
        state.prev_accelerations,
    )

    # --- 12. Build new state ---
    new_state = TorchWorldState(
        positions=new_positions,
        velocities=new_velocities,
        torso_orientations=new_torso_orientations,
        head_orientations=new_head_orientations,
        shoulder_widths=state.shoulder_widths,
        chest_depths=state.chest_depths,
        goal_positions=state.goal_positions,
        preferred_speeds=state.preferred_speeds,
        active_mask=new_active_mask,
        cumulative_terminated=new_cumulative_terminated,
        wall_segments=state.wall_segments,
        n_segments=state.n_segments,
        prev_velocities=new_velocities,
        prev_goal_distances=new_goal_distances,
        prev_accelerations=new_prev_accelerations,
        prev_headings=new_torso_orientations,
        prev_heading_changes=state.prev_heading_changes,
        prev_actions=actions,
        waypoints=state.waypoints,
        n_waypoints=state.n_waypoints,
        waypoint_cursor=new_wp_cursor,
        waypoint_path_lengths=state.waypoint_path_lengths,
        n_agents=state.n_agents,
        step_count=step_count,
    )

    # --- 13. Build observations ---
    observations = build_observations(
        new_positions,
        new_velocities,
        new_torso_orientations,
        new_head_orientations,
        state.shoulder_widths,
        state.chest_depths,
        state.goal_positions,
        new_active_mask,
        state.n_agents,
        state.wall_segments,
        state.n_segments,
        config,
        waypoints=state.waypoints,
        n_waypoints=state.n_waypoints,
        waypoint_cursor=new_wp_cursor,
        waypoint_path_lengths=state.waypoint_path_lengths,
    )

    return new_state, observations, rewards, terminated, truncated
