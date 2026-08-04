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

# Per-component reward breakdown channels, in stacking order. The components
# tensor returned by ``compute_rewards`` has shape (E, N, len(...)) and its
# channels sum exactly to the total per-agent reward (timeout is filled in by
# ``batched_step``, which is where the episode-end penalties are applied).
# Single source of truth: collector and training loop import this to label the
# per-episode reward decomposition used for collapse instrumentation.
REWARD_COMPONENT_NAMES: tuple[str, ...] = (
    "goal",
    "collision_agent",
    "wall_proximity",
    "wall_collision",
    "agent_proximity",
    "action_rate",
    "existence",
    "progress",
    "smoothness",
    "timeout",
)
TIMEOUT_COMPONENT_IDX = REWARD_COMPONENT_NAMES.index("timeout")


def compute_remaining_path(
    positions: Tensor,
    waypoints: Tensor,
    waypoint_cursor: Tensor,
    n_waypoints: Tensor,
    waypoint_path_lengths: Tensor,
) -> Tensor:
    """Remaining navmesh path length from each agent to its goal.

    ``dist(position, current_waypoint) + cumulative_remaining_from_that_waypoint``
    -- the same quantity the navmesh obs signal uses for ``path_deviation``
    (see ``crowdrl_torch.observation.compute_navmesh_signals``). Factored out so
    the progress reward and the stuck-termination check measure progress ALONG
    THE ROUTE rather than straight-line distance to the global goal: the two
    diverge whenever the path bends away from the goal, and grading on the
    bee-line both under-rewards and prematurely kills correct path-following.

    Returns 0 for agents with no waypoints; callers fall back to straight-line
    goal distance there (see :func:`path_distance_metric`).
    """
    max_wp = waypoints.shape[2]
    cursor = waypoint_cursor.long()
    n_wp = n_waypoints.long()
    max_idx = (n_wp - 1).clamp(min=0)
    cursor_a = cursor.clamp(min=0, max=max_wp - 1).clamp(max=max_idx)

    idx = cursor_a.unsqueeze(-1).unsqueeze(-1).expand(cursor_a.shape[0], cursor_a.shape[1], 1, 2)
    wp_a = waypoints.gather(2, idx).squeeze(2)  # (E, N, 2)
    d_a = ((wp_a - positions) ** 2).sum(dim=-1).sqrt()  # (E, N)

    remaining_from_wp = waypoint_path_lengths.gather(2, cursor_a.unsqueeze(-1)).squeeze(-1)
    return d_a + remaining_from_wp


def path_distance_metric(
    positions: Tensor,
    goal_positions: Tensor,
    waypoints: Tensor | None,
    waypoint_cursor: Tensor | None,
    n_waypoints: Tensor | None,
    waypoint_path_lengths: Tensor | None,
    use_navmesh: bool,
) -> Tensor:
    """Distance-to-goal metric for the progress reward and stuck check.

    Returns the remaining navmesh **path** length where a path is available,
    else straight-line distance to the global goal. The fallback covers
    ``use_navmesh=False`` and any agent with no computed waypoints; on Tier-0
    open fields the single waypoint IS the goal, so the two coincide.
    """
    goal_dist = ((goal_positions - positions) ** 2).sum(dim=-1).sqrt()
    if not use_navmesh or waypoints is None or n_waypoints is None:
        return goal_dist
    remaining = compute_remaining_path(
        positions, waypoints, waypoint_cursor, n_waypoints, waypoint_path_lengths
    )
    return torch.where(n_waypoints > 0, remaining, goal_dist)


def compute_rewards(
    positions: Tensor,
    velocities: Tensor,
    goal_positions: Tensor,
    active_mask: Tensor,
    collision_mask: Tensor,
    prev_distances: Tensor,
    config: EnvConfig,
    *,
    current_distances: Tensor | None = None,
    wall_distances: Tensor | None = None,
    wall_collision_mask: Tensor | None = None,
    wall_directions: Tensor | None = None,
    agent_radii: Tensor | None = None,
    collision_velocities: Tensor | None = None,
    actions: Tensor | None = None,
    prev_actions: Tensor | None = None,
    headings: Tensor | None = None,
    preferred_speeds: Tensor | None = None,
    prev_velocities: Tensor | None = None,
    prev_accelerations: Tensor | None = None,
    prev_headings: Tensor | None = None,
    prev_heading_changes: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute per-agent rewards for one timestep.

    Parameters
    ----------
    positions : (E, N, 2)
    velocities : (E, N, 2)
    goal_positions : (E, N, 2)
    active_mask : (E, N) bool
    collision_mask : (E, N) bool
    prev_distances : (E, N) -- previous-step distance metric for the progress
        reward. In production this is the navmesh remaining-PATH distance; with
        no ``current_distances`` passed it pairs with straight-line goal distance.
    config : EnvConfig
    current_distances : (E, N) optional -- current-step distance metric. When
        given, progress = ``prev_distances - current_distances`` (path-aware);
        otherwise it falls back to straight-line goal distance.
    wall_distances : (E, N) optional -- min distance to nearest wall per agent
    wall_collision_mask : (E, N) bool optional -- True where the boundary
        enforcement corrected the agent this step (hard wall-contact signal)
    wall_directions : (E, N, 2) optional -- unit vector agent -> nearest wall
        point (``compute_min_wall_distances_and_directions``). Needed by
        ``use_velocity_weighted_wall_proximity`` (closing speed toward the
        wall) and ``use_wall_normal_impact`` (into-wall impact component);
        both fall back to their legacy behaviour when this is None.
    agent_radii : (E, N) optional -- agent body radii (used for the graded
        agent-proximity penalty: per-pair contact distance = r_i + r_j)
    collision_velocities : (E, N, 2) optional -- pre-contact velocities for the
        impact-speed weighting (``use_velocity_weighted_collision``); falls back
        to ``velocities`` when not given.
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
    components : (E, N, C) -- per-component reward breakdown, channels ordered
        by ``REWARD_COMPONENT_NAMES``. Channels sum to ``rewards`` (the
        ``timeout`` channel is left zero here and filled by ``batched_step``).

    Notes
    -----
    Each component is accumulated into both ``rewards`` and its own channel via
    ``rewards = rewards + comp_k`` where ``comp_k = where(mask_k, X_k, 0)``. This
    is bit-identical to the prior ``where(mask_k, rewards + X_k, rewards)`` form
    (adding a masked-to-zero delta), so the total reward is unchanged.
    """
    # Goal distances
    goal_diffs = goal_positions - positions
    goal_distances = (goal_diffs**2).sum(dim=-1).sqrt()  # (E, N)

    rewards = torch.zeros_like(goal_distances)
    zero = torch.zeros_like(goal_distances)

    # Numerical safety for the velocity-weighted penalties (mirrors crowdrl_env):
    # sanitize the pre-contact velocity snapshot and cap the impact speed so a
    # degenerate high-density pileup / transient non-finite velocity cannot
    # inject NaN or an unbounded multiplier into the reward and poison training.
    max_impact_speed = 10.0  # m/s, a safety ceiling well above any physical closing
    if collision_velocities is not None:
        collision_velocities = torch.nan_to_num(
            collision_velocities, nan=0.0, posinf=0.0, neginf=0.0
        )

    # Goal reaching
    reached_goal = (goal_distances < config.goal_radius) & active_mask
    comp_goal = torch.where(reached_goal, torch.full_like(rewards, config.goal_bonus), zero)
    rewards = rewards + comp_goal

    # Collision penalty. Binary per-step by default; when
    # use_velocity_weighted_collision is set, scale by the CLOSING speed between
    # contacting agents so high-speed impacts cost more than gentle contact
    # (still gated by collision_mask). Mirrors crowdrl_env.reward.compute_rewards
    # exactly -- keep the two in lockstep (test_equivalence guards this).
    coll_active = collision_mask & active_mask
    if config.use_velocity_weighted_collision and agent_radii is not None:
        vel = collision_velocities if collision_velocities is not None else velocities
        sep = positions.unsqueeze(2) - positions.unsqueeze(1)  # (E,N,N,2) p_i - p_j
        sep_dist = (sep**2).sum(dim=-1).sqrt()  # (E,N,N)
        sep_unit = sep / sep_dist.clamp(min=1e-9).unsqueeze(-1)
        rel_vel = vel.unsqueeze(2) - vel.unsqueeze(1)  # (E,N,N,2) v_i - v_j
        closing = -(rel_vel * sep_unit).sum(dim=-1)  # (E,N,N) >0 when approaching
        contact_dist = 1.2 * (agent_radii.unsqueeze(2) + agent_radii.unsqueeze(1))
        n_c = positions.shape[1]
        eye_c = torch.eye(n_c, device=positions.device, dtype=torch.bool).unsqueeze(0)
        near = (
            (~eye_c)
            & (sep_dist <= contact_dist)
            & active_mask.unsqueeze(1)
            & active_mask.unsqueeze(2)
        )
        closing = torch.where(near, closing.clamp(min=0.0), torch.zeros_like(closing))
        impact_speed = closing.max(dim=2).values.clamp(max=max_impact_speed)  # (E,N) capped
        speed_scale = (
            config.collision_speed_floor + config.collision_speed_scale * impact_speed
        ).clamp(min=0.0)
        comp_collision = torch.where(coll_active, config.collision_penalty * speed_scale, zero)
    else:
        comp_collision = torch.where(
            coll_active,
            torch.full_like(rewards, config.collision_penalty),
            zero,
        )
    # Cap the per-step collision penalty at a floor: velocity weighting may
    # DISCOUNT slow contact but not AMPLIFY fast contact below the cap (mirrors
    # crowdrl_env.reward). cap=0.0 disables. Static branch -> torch.compile-safe.
    if config.collision_penalty_cap < 0.0:
        comp_collision = comp_collision.clamp(min=config.collision_penalty_cap)
    rewards = rewards + comp_collision

    # Wall proximity penalty. Legacy mode is a FLAT band -- a step, not a
    # gradient: constant penalty anywhere inside threshold*radius. Opt-in
    # graded mode mirrors the agent-proximity ramp (near at body contact ->
    # far at the threshold edge); opt-in closing-speed weighting taxes
    # APPROACHING the wall rather than being near it, so yielding beside a
    # wall can be free. Mirrors crowdrl_env.reward -- keep in lockstep
    # (test_equivalence guards this).
    comp_wall = zero
    if config.use_graded_wall_proximity:
        wall_prox_on = (
            config.wall_proximity_penalty_near != 0.0 or config.wall_proximity_penalty_far != 0.0
        )
    else:
        wall_prox_on = config.wall_proximity_penalty != 0.0
    if wall_prox_on and wall_distances is not None and agent_radii is not None:
        threshold = agent_radii * config.wall_proximity_threshold
        in_band = (wall_distances < threshold) & active_mask
        if config.use_graded_wall_proximity:
            # Linear ramp: ``near`` at wall_distance == radius, ``far`` at the
            # threshold. Below-radius distances clamp to ``near``.
            denom = torch.clamp(threshold - agent_radii, min=1e-6)
            t = torch.clamp((wall_distances - agent_radii) / denom, 0.0, 1.0)
            wall_pen = (1.0 - t) * config.wall_proximity_penalty_near + (
                t * config.wall_proximity_penalty_far
            )
        else:
            wall_pen = torch.full_like(rewards, config.wall_proximity_penalty)
        if config.use_velocity_weighted_wall_proximity and wall_directions is not None:
            vel = collision_velocities if collision_velocities is not None else velocities
            closing = (
                (vel * wall_directions).sum(dim=-1).clamp(max=max_impact_speed)
            )  # (E, N), >0 when approaching the nearest wall (capped)
            speed_w = (
                config.wall_proximity_speed_floor + config.wall_proximity_speed_scale * closing
            ).clamp(min=0.0)
            wall_pen = wall_pen * speed_w
        comp_wall = torch.where(in_band, wall_pen, zero)
        rewards = rewards + comp_wall

    # Wall contact penalty (hard, per-step while the boundary pushes the agent
    # back). Distinct from the proximity band -- deters using walls as a free
    # brake. Mirrors the agent collision penalty.
    comp_wall_collision = zero
    if config.wall_collision_penalty != 0.0 and wall_collision_mask is not None:
        wall_active = wall_collision_mask & active_mask
        if config.use_velocity_weighted_collision:
            vel = collision_velocities if collision_velocities is not None else velocities
            if config.use_wall_normal_impact and wall_directions is not None:
                # Impact speed = the (pre-contact) velocity component INTO the
                # wall, clamped >= 0, so sliding parallel to the wall is cheap
                # while slamming into it head-on is not.
                impact = (vel * wall_directions).sum(dim=-1).clamp(min=0.0, max=max_impact_speed)
            else:
                # No wall normal in use: weight by the agent's own FULL
                # (pre-contact) speed -- ramming at speed costs more than
                # drifting, but sliding parallel costs exactly as much as a
                # head-on at the same speed.
                impact = (vel**2).sum(dim=-1).sqrt().clamp(max=max_impact_speed)  # (E,N) capped
            wall_scale = (
                config.collision_speed_floor + config.collision_speed_scale * impact
            ).clamp(min=0.0)
            comp_wall_collision = torch.where(
                wall_active, config.wall_collision_penalty * wall_scale, zero
            )
        else:
            comp_wall_collision = torch.where(
                wall_active,
                torch.full_like(rewards, config.wall_collision_penalty),
                zero,
            )
        rewards = rewards + comp_wall_collision

    # Agent proximity penalty (graded linear ramp, min over neighbours).
    # Penalty interpolates between ``near`` (at contact, r_i + r_j) and
    # ``far`` (at personal_space_radius). Each agent pays the penalty of its
    # most-penalised neighbour inside the zone.
    comp_agent_prox = zero
    if (
        config.agent_proximity_penalty_near != 0.0 or config.agent_proximity_penalty_far != 0.0
    ) and agent_radii is not None:
        # Pairwise center-to-center distances (E, N, N)
        diff_p = positions.unsqueeze(2) - positions.unsqueeze(1)
        pair_dist = (diff_p**2).sum(dim=-1).sqrt()

        # Per-pair contact distance r_i + r_j (E, N, N)
        pair_contact = agent_radii.unsqueeze(2) + agent_radii.unsqueeze(1)

        # Linear interpolation factor t in [0, 1]: 0 at contact, 1 at boundary.
        denom = torch.clamp(config.personal_space_radius - pair_contact, min=1e-6)
        t = torch.clamp((pair_dist - pair_contact) / denom, 0.0, 1.0)
        pair_penalty = (1.0 - t) * config.agent_proximity_penalty_near + (
            t * config.agent_proximity_penalty_far
        )

        # Optionally weight by CLOSING speed (penalise approaching at speed, not
        # mere coexistence) -- mirrors crowdrl_env.reward. Here ``diff_p`` is
        # p_i - p_j, so closing (>0 when i, j approach) is -(v_i - v_j) . unit.
        if config.use_velocity_weighted_proximity:
            vel = collision_velocities if collision_velocities is not None else velocities
            diff_unit = diff_p / pair_dist.clamp(min=1e-9).unsqueeze(-1)
            rel_vel = vel.unsqueeze(2) - vel.unsqueeze(1)  # v_i - v_j
            closing = (-(rel_vel * diff_unit).sum(dim=-1)).clamp(max=max_impact_speed)  # capped
            speed_w = (
                config.proximity_speed_floor + config.proximity_speed_scale * closing
            ).clamp(min=0.0)
            pair_penalty = pair_penalty * speed_w

        # Mask: no self-pairs, only active-active pairs, only pairs inside zone.
        N_ = positions.shape[1]
        eye = torch.eye(N_, device=positions.device, dtype=torch.bool)
        valid_pair = (~eye.unsqueeze(0)) & active_mask.unsqueeze(1) & active_mask.unsqueeze(2)
        in_zone = pair_dist < config.personal_space_radius
        pair_penalty = torch.where(
            valid_pair & in_zone, pair_penalty, torch.zeros_like(pair_penalty)
        )

        # Per-agent: most-negative penalty from any neighbour.
        proximity = pair_penalty.min(dim=2).values  # (E, N)
        comp_agent_prox = torch.where(active_mask, proximity, zero)
        rewards = rewards + comp_agent_prox

    # Action rate penalty (change in raw policy output between steps)
    comp_action_rate = zero
    if config.action_rate_weight != 0.0 and actions is not None and prev_actions is not None:
        action_change = ((actions - prev_actions) ** 2).sum(dim=-1).sqrt()  # (E, N)
        comp_action_rate = torch.where(
            active_mask,
            config.action_rate_weight * action_change,
            zero,
        )
        rewards = rewards + comp_action_rate

    # Existence penalty: every step alive costs you
    comp_existence = zero
    if config.existence_penalty != 0.0:
        comp_existence = torch.where(
            active_mask,
            torch.full_like(rewards, config.existence_penalty),
            zero,
        )
        rewards = rewards + comp_existence

    # Progress reward (potential-based shaping). Uses ``current_distances`` (the
    # navmesh remaining-PATH metric, passed by batched_step) when available so
    # progress is measured along the route, not the straight-line bee-line to the
    # goal; falls back to straight-line goal distance otherwise.
    curr = current_distances if current_distances is not None else goal_distances
    progress = prev_distances - curr
    comp_progress = torch.where(active_mask, config.progress_weight * progress, zero)
    rewards = rewards + comp_progress

    # --- Tier 2: Smoothness (jerk + angular accel; speed deviation is separate) ---
    # NOTE: jerk/angular-accel need prev_velocities and are gated by use_smoothness.
    # Preferred-speed deviation is NOT: it depends only on the current velocity and
    # is a locomotion-style target (match your own preferred speed), not a motion-
    # smoothness regulariser. Nesting it here made speed_deviation_weight silently
    # inert whenever use_smoothness was false -- which is the baseline setting -- so
    # a run could ask for speed matching and train without it. Its contribution is
    # still reported on the ``smoothness`` channel to keep the component schema at
    # 10 entries.
    comp_smoothness = zero
    if config.use_smoothness and prev_velocities is not None:
        dt = config.dt
        accelerations = (velocities - prev_velocities) / dt  # (E, N, 2)

        # Jerk penalty (change in acceleration)
        if config.jerk_penalty_weight != 0.0 and prev_accelerations is not None:
            jerk = (accelerations - prev_accelerations) / dt  # (E, N, 2)
            jerk_mag = (jerk**2).sum(dim=-1).sqrt()  # (E, N)
            term = torch.where(active_mask, config.jerk_penalty_weight * jerk_mag, zero)
            comp_smoothness = comp_smoothness + term
            rewards = rewards + term

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
            term = torch.where(
                active_mask, config.angular_accel_penalty_weight * angular_accel, zero
            )
            comp_smoothness = comp_smoothness + term
            rewards = rewards + term

    # Preferred speed deviation -- independent of use_smoothness (see note above).
    if config.speed_deviation_weight != 0.0 and preferred_speeds is not None:
        speeds = (velocities**2).sum(dim=-1).sqrt()  # (E, N)
        speed_dev = (speeds - preferred_speeds).abs()
        term = torch.where(active_mask, config.speed_deviation_weight * speed_dev, zero)
        comp_smoothness = comp_smoothness + term
        rewards = rewards + term

    # Zero rewards for inactive agents (no-op for the sum: every component is
    # already zero where inactive, but kept for parity with the prior code).
    rewards = torch.where(active_mask, rewards, zero)

    # Per-component breakdown. ``timeout`` is the last channel and stays zero
    # here; batched_step adds the episode-end timeout/stuck penalty into it.
    comp_timeout = zero
    components = torch.stack(
        [
            comp_goal,
            comp_collision,
            comp_wall,
            comp_wall_collision,
            comp_agent_prox,
            comp_action_rate,
            comp_existence,
            comp_progress,
            comp_smoothness,
            comp_timeout,
        ],
        dim=-1,
    )

    return rewards, reached_goal, goal_distances, components
