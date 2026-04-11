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
from crowdrl_torch.sensing import match_persistent_neighbors
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

    # --- 4. Contact forces (F=ma, divided by per-agent mass) ---
    forces = compute_contact_forces(
        state.positions,
        new_velocities,
        state.shoulder_widths,
        state.chest_depths,
        state.masses,
        state.active_mask,
        overlap_matrix,
        state.wall_segments,
        state.n_segments,
        config,
    )

    # Apply accelerations as velocity impulse
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
    # Wall distances for wall-proximity penalty. Agent-agent pair distances
    # are computed inside compute_rewards so the graded proximity ramp can
    # use per-pair contact distances.
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
        headings=new_torso_orientations,
        preferred_speeds=state.preferred_speeds,
        prev_velocities=state.prev_velocities,
        prev_accelerations=state.prev_accelerations,
        prev_headings=state.prev_headings,
        prev_heading_changes=state.prev_heading_changes,
    )

    # --- 8. Update active mask ---
    newly_done = reached_goal & state.active_mask
    new_cumulative_terminated = state.cumulative_terminated | newly_done
    new_active_mask = state.active_mask & ~newly_done

    # Zero velocities for inactive agents
    new_velocities = torch.where(
        new_active_mask.unsqueeze(-1), new_velocities, torch.zeros_like(new_velocities)
    )

    # --- 9. Stuck-agent termination (rolling progress window) ---
    # After the active-mask update above, an agent that just reached its
    # goal is inactive and excluded from the stuck check. Window counters
    # accumulate only on active agents; inactive agents keep their state
    # frozen (will be reset by the reset path on next episode).
    truncated = torch.zeros(E, N, dtype=torch.bool, device=state.positions.device)
    stuck_mask = torch.zeros_like(new_active_mask)
    new_stuck_window_step = state.stuck_window_step
    new_stuck_window_start_dist = state.stuck_window_start_dist
    if config.stuck_termination_enabled:
        # Increment the window step for agents that were active coming into
        # this step AND are still active after the goal-reach update.
        inc_mask = new_active_mask
        new_stuck_window_step = torch.where(
            inc_mask,
            state.stuck_window_step + 1,
            state.stuck_window_step,
        )
        # At end of window: compute progress = start_dist - current_dist.
        # If progress < threshold, agent is stuck. Otherwise reset window.
        window_full = new_stuck_window_step >= config.stuck_window_steps
        progress = state.stuck_window_start_dist - new_goal_distances
        stuck_mask = window_full & inc_mask & (progress < config.stuck_progress_threshold)
        # For non-stuck window-full agents, restart the window: zero the
        # counter and capture the current distance as the new start.
        reset_mask = window_full & inc_mask & ~stuck_mask
        new_stuck_window_step = torch.where(
            window_full & inc_mask,
            torch.zeros_like(new_stuck_window_step),
            new_stuck_window_step,
        )
        new_stuck_window_start_dist = torch.where(
            reset_mask,
            new_goal_distances,
            state.stuck_window_start_dist,
        )

        # Apply timeout penalty to stuck agents and mark them truncated.
        rewards = torch.where(stuck_mask, rewards + config.timeout_penalty, rewards)
        truncated = torch.where(stuck_mask, torch.ones_like(truncated), truncated)
        new_active_mask = new_active_mask & ~stuck_mask
        new_velocities = torch.where(
            new_active_mask.unsqueeze(-1), new_velocities, torch.zeros_like(new_velocities)
        )

    # --- 10. Termination / truncation (episode timeout) ---
    terminated = reached_goal

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

    # Heading changes for angular acceleration penalty
    heading_change = new_torso_orientations - state.prev_headings
    heading_change = (heading_change + torch.pi) % (2 * torch.pi) - torch.pi
    new_prev_heading_changes = torch.where(
        state.prev_headings.abs() > 0,
        heading_change,
        state.prev_heading_changes,
    )

    # --- 11b. Update temporal memory state ---
    # Cumulative path length accumulates only while the agent is active going
    # into this step (excluding stuck/timeout termination which zeros the
    # active mask above). We use the pre-step position so the per-step
    # delta reflects this step's actual motion.
    step_delta = ((new_positions - state.positions) ** 2).sum(dim=-1).sqrt()  # (E, N)
    # Use state.active_mask (pre-update) so the final motion step of an agent
    # that just reached the goal or was just deactivated still counts.
    path_inc = torch.where(state.active_mask, step_delta, torch.zeros_like(step_delta))
    new_cumulative_path_length = state.cumulative_path_length + path_inc

    # Scatter-write the new position and goal distance into the ring buffer.
    # Writer index = pre-step step_count mod (W+1). After the first W steps
    # the oldest slot gets overwritten, and the read at (step_count+1) mod
    # (W+1) returns the entry from W steps back.
    W = config.temporal_memory_window
    buf_size = W + 1
    write_idx = (state.step_count % buf_size).long()  # (E,)
    write_idx_pos = write_idx.view(E, 1, 1, 1).expand(E, N, 1, 2)  # (E, N, 1, 2)
    new_pos_history = state.pos_history.scatter(
        dim=2, index=write_idx_pos, src=new_positions.unsqueeze(2)
    )
    write_idx_gd = write_idx.view(E, 1, 1).expand(E, N, 1)  # (E, N, 1)
    new_gdist_history = state.gdist_history.scatter(
        dim=2, index=write_idx_gd, src=new_goal_distances.unsqueeze(2)
    )

    # --- 11c. Update persistent neighbor slots + velocity history ---
    # Recompute neighbor slot assignments from the new positions. Uses the
    # post-update active mask so newly-deactivated agents (goal-reach or
    # stuck-termination) are immediately evicted from other agents' slots.
    # Guarded so we only pay the ~2% compute hit when neighbor memory is
    # actually consumed by the observation builder (commits 4/5).
    if config.use_neighbor_memory:
        new_neighbor_ids = match_persistent_neighbors(
            new_positions,
            state.neighbor_ids,
            new_active_mask,
            state.n_agents,
            sensing_radius=config.neighbor_sensing_radius,
            config=config,
        )

        # --- Zero-reset slot history on reassignment ---
        # When slot k at ego i gets a new neighbor ID, we wipe the full
        # history for that slot so the diff/mean features don't mix the
        # prior assignee's velocities with the new one. For slots that
        # kept their previous assignment, the history passes through
        # unchanged (the scatter below will overwrite only the write slot).
        K_local = config.k_neighbours
        slot_changed = new_neighbor_ids != state.neighbor_ids  # (E, N, K)
        # Broadcast to (E, N, W_n+1, K, 2) for torch.where
        preserve_mask = ~slot_changed.view(E, N, 1, K_local, 1)
        history_after_reset = torch.where(
            preserve_mask,
            state.neighbor_vel_history,
            torch.zeros_like(state.neighbor_vel_history),
        )

        # --- Gather new velocities for each slot's assigned neighbor ---
        # new_neighbor_ids[e, i, k] holds the global agent index of the k-th
        # neighbor of ego i; -1 means empty. Use advanced indexing to pull
        # new_velocities[e, nb_id] for each slot. Clamp -1 to 0 first so
        # the gather doesn't index out of bounds, then mask empty slots.
        nb_ids_safe = new_neighbor_ids.clamp(min=0).long()  # (E, N, K)
        env_idx = torch.arange(E, device=new_positions.device).view(E, 1, 1).expand(E, N, K_local)
        nb_vels = new_velocities[env_idx, nb_ids_safe]  # (E, N, K, 2)
        nb_valid = (new_neighbor_ids >= 0).unsqueeze(-1)  # (E, N, K, 1)
        nb_vels = torch.where(nb_valid, nb_vels, torch.zeros_like(nb_vels))

        # --- Scatter-write into the ring buffer at step_count % (W_n+1) ---
        W_n = config.neighbor_vel_history_window
        nb_buf = W_n + 1
        nb_write_idx = (state.step_count % nb_buf).long()  # (E,)
        # history_after_reset has shape (E, N, W_n+1, K, 2); we scatter on
        # dim=2 with index of shape (E, N, 1, K, 2) and src of the same shape.
        nb_write_idx_exp = nb_write_idx.view(E, 1, 1, 1, 1).expand(E, N, 1, K_local, 2)
        new_neighbor_vel_history = history_after_reset.scatter(
            dim=2,
            index=nb_write_idx_exp,
            src=nb_vels.unsqueeze(2),
        )
    else:
        new_neighbor_ids = state.neighbor_ids
        new_neighbor_vel_history = state.neighbor_vel_history

    # --- 12. Build new state ---
    new_state = TorchWorldState(
        positions=new_positions,
        velocities=new_velocities,
        torso_orientations=new_torso_orientations,
        head_orientations=new_head_orientations,
        shoulder_widths=state.shoulder_widths,
        chest_depths=state.chest_depths,
        masses=state.masses,
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
        prev_heading_changes=new_prev_heading_changes,
        prev_actions=actions,
        waypoints=state.waypoints,
        n_waypoints=state.n_waypoints,
        waypoint_cursor=new_wp_cursor,
        waypoint_path_lengths=state.waypoint_path_lengths,
        n_agents=state.n_agents,
        step_count=step_count,
        stuck_window_step=new_stuck_window_step,
        stuck_window_start_dist=new_stuck_window_start_dist,
        spawn_positions=state.spawn_positions,
        initial_goal_distances=state.initial_goal_distances,
        cumulative_path_length=new_cumulative_path_length,
        pos_history=new_pos_history,
        gdist_history=new_gdist_history,
        neighbor_ids=new_neighbor_ids,
        neighbor_vel_history=new_neighbor_vel_history,
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
        spawn_positions=new_state.spawn_positions,
        initial_goal_distances=new_state.initial_goal_distances,
        cumulative_path_length=new_state.cumulative_path_length,
        pos_history=new_state.pos_history,
        gdist_history=new_state.gdist_history,
        preferred_speeds=new_state.preferred_speeds,
        step_count=step_count,
    )

    return new_state, observations, rewards, terminated, truncated
