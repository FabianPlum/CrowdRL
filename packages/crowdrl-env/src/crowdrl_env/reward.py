"""Reward computation for the CrowdRL training environment.

Tier 1 — Sparse task rewards:
  Goal-reaching bonus, collision penalty, timeout penalty.

Tier 2 — Smoothness priors:
  Jerk penalty, angular acceleration penalty, preferred-speed deviation.

Tier 3 — Distributional style matching (future, requires PeTrack data).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RewardConfig:
    """Configuration for reward computation."""

    # Tier 1: sparse
    goal_bonus: float = 10.0
    """Reward for reaching the goal."""

    collision_penalty: float = -1.0
    """Penalty per timestep while in collision."""

    timeout_penalty: float = -5.0
    """Penalty for not reaching goal within episode."""

    goal_radius: float = 0.5
    """Distance threshold (metres) for goal reached."""

    # Wall penalties: a soft proximity band plus a hard contact penalty.
    wall_proximity_penalty: float = -0.1
    """Penalty for agents too close to walls (smooth, distance-based)."""

    wall_proximity_threshold: float = 1.5
    """Wall proximity threshold as a multiple of agent radius."""

    wall_collision_penalty: float = -1.0
    """Penalty per step while in CONTACT with a wall, i.e. the boundary
    enforcement had to push the agent back / cancel its into-wall velocity.
    Distinct from the proximity band: proximity is fine, contact is not.
    Deters using walls as a free brake (instant deceleration at no cost).
    Mirrors the agent ``collision_penalty``. Default -1.0; set 0.0 to disable."""

    # Impact-speed weighting for the collision / wall-contact penalties
    # (optional, default OFF -> binary per-step penalty). A binary penalty has
    # ZERO marginal cost for hitting at speed, so once contact is unavoidable
    # the policy has no reason to slow down ("bulldozing"). Weighting the
    # penalty by impact speed -- CLOSING speed for agent-agent contact, the
    # agent's own speed for wall contact -- restores that gradient: a head-on at
    # 3 m/s hurts far more than a gentle brush, while slow contact in a dense
    # crowd stays cheap. The penalty is still GATED by the env's collision /
    # wall-contact masks; this only reshapes its magnitude:
    #   multiplier = max(0, collision_speed_floor + collision_speed_scale * v)
    use_velocity_weighted_collision: bool = False
    """Enable impact-speed weighting of the collision/wall-contact penalties.
    False reproduces the binary per-step penalty exactly (the default)."""

    collision_speed_floor: float = 0.5
    """Penalty multiplier at zero impact speed -- a base deterrent so resting in
    contact is not free. With the defaults, a ~1 m/s contact reproduces the
    unweighted magnitude (floor + scale * 1.0 = 1.0)."""

    collision_speed_scale: float = 0.5
    """Extra penalty multiplier per m/s of impact speed (closing speed for
    agent-agent contact, own speed for wall contact)."""

    collision_penalty_cap: float = 0.0
    """Per-step floor (NEGATIVE, e.g. -2.0; 0.0 disables) on the agent collision
    penalty. Caps how negative the per-step term can get so the velocity
    weighting can DISCOUNT slow contact but never AMPLIFY fast contact below this
    floor. The slow60 experiment showed amplification-by-closing-speed makes
    dense pileups WORSE (coll_ag -> -1086, gridlock); capping at the base
    collision_penalty turns the weighting into discount-only."""

    # Velocity weighting for the agent-PROXIMITY penalty (optional, default OFF).
    # The distance-only proximity ramp penalises an agent for merely BEING near a
    # neighbour, so in a crowd the cheapest policy is to stop at the edge and not
    # enter -- a major driver of the freezing failure mode. Weighting the penalty
    # by CLOSING speed makes slow coexistence / threading cheap and only taxes
    # approaching a neighbour at speed, so agents can move THROUGH a crowd instead
    # of freezing outside it. Same closing-speed philosophy as P1 (collision).
    use_velocity_weighted_proximity: bool = False
    """Enable closing-speed weighting of the agent-proximity penalty.
    False reproduces the distance-only penalty exactly (the default)."""

    proximity_speed_floor: float = 0.25
    """Proximity-penalty multiplier at zero closing speed (a mild residual cost
    so coexistence keeps some anticipation pressure; 0.0 = free when not closing)."""

    proximity_speed_scale: float = 0.5
    """Extra proximity-penalty multiplier per m/s of closing speed."""

    # Agent proximity penalty (graded linear ramp, learned collision avoidance)
    # The penalty per step is linearly interpolated on the center-to-center
    # distance between an agent and its nearest active neighbour:
    #   - at contact distance (r_i + r_j): ``agent_proximity_penalty_near``
    #   - at ``personal_space_radius``:    ``agent_proximity_penalty_far``
    #   - beyond ``personal_space_radius``: no penalty
    # This provides a continuous gradient for the policy to maintain personal
    # space, while the binary ``collision_penalty`` handles the hard "you
    # touched someone" signal on top. See the project plan, Section 3.2.
    agent_proximity_penalty_near: float = -0.005
    """Strongest proximity penalty magnitude, applied when agents are at
    contact distance (sum of body radii, center-to-center)."""

    agent_proximity_penalty_far: float = -0.0001
    """Weakest proximity penalty magnitude, applied when agents are right at
    the ``personal_space_radius`` boundary."""

    personal_space_radius: float = 1.0
    """Absolute center-to-center distance (metres) at which the proximity
    penalty first kicks in. Decoupled from body dimensions so the ramp has
    a meaningful approach zone regardless of agent size."""

    # Action rate penalty
    action_rate_weight: float = -0.01
    """Weight for penalising large changes in raw policy output between steps.
    0.0 = disabled. Layer 1 of plan/agent_dynamics_refactor.md
    (2026-05-25) enabled this at -0.01 so the penalty can compete with
    the goal bonus."""

    # Tier 2: smoothness
    use_smoothness: bool = True
    """Whether to apply Tier 2 smoothness penalties."""

    jerk_penalty_weight: float = -1e-5
    """Weight for acceleration change (jerk) penalty. Layer 1 v2
    (notebook 09 diagnostic showed Layer 1 v1's -1e-4 made deceleration
    expensive enough to flip the brake-vs-collide trade-off the wrong
    way). Dropped 10x to -1e-5. Jerk scales as 1/dt^2 so raw magnitudes
    are large; this weight just makes the signal informative without
    dominating the brake decision."""

    angular_accel_penalty_weight: float = -0.01
    """Weight for angular acceleration penalty. Layer 1 of
    plan/agent_dynamics_refactor.md (2026-05-25) raised this 100x
    (from -1e-4 to -1e-2)."""

    speed_deviation_weight: float = -0.005
    """Weight for |actual_speed - preferred_speed| penalty (m/s units).

    Layer 1 v2 (notebook 09 surfaced the ice-skating pathology: at -0.1,
    braking for 1 s cost ~-13 reward in speed_deviation alone, more than
    the cost of plowing through a wall; the policy correctly converged
    to never brake). Dropped 20x to -0.005 so braking is cheap enough
    that collision-avoidance becomes the dominant signal, but still
    biases the policy toward the per-agent preferred speed under
    unconstrained motion. The policy can now observe its preferred
    speed directly (ego state index 5) so this is a true tracking
    signal, not a hidden target.

    Future tunable: scale this weight down further in dense-proximity
    contexts so queuing behaviour is not punished. Likely emerges
    implicitly from the agent_proximity penalties; revisit if it
    does not."""

    # Existence penalty (per-step cost for being alive)
    existence_penalty: float = -0.01
    """Small negative reward every step an agent is active.
    Pressures agents to reach their goal quickly. 0.0 = disabled."""

    # Progress reward (shaped)
    progress_weight: float = 1.0
    """Reward for getting closer to goal (potential-based shaping)."""

    # Inverse distance to goal (continuous proximity signal)
    inverse_distance_weight: float = 0.0
    """Per-step reward proportional to 1/(distance_to_goal + 1).
    Captures intermediate progress — closer is better. 0.0 = disabled."""


@dataclass
class RewardState:
    """Mutable state needed for temporal reward computation.

    Tracks previous-step quantities to compute derivatives (jerk, angular accel).
    """

    prev_velocities: NDArray[np.float64] | None = None
    """(n_agents, 2) — velocities from the previous step."""

    prev_accelerations: NDArray[np.float64] | None = None
    """(n_agents, 2) — accelerations from the previous step (for jerk)."""

    prev_headings: NDArray[np.float64] | None = None
    """(n_agents,) — headings from the previous step."""

    prev_heading_changes: NDArray[np.float64] | None = None
    """(n_agents,) — heading changes from the previous step (for angular accel)."""

    prev_nav_distances: NDArray[np.float64] | None = None
    """(n_agents,) — path-distance metric (navmesh remaining path, or straight-
    line goal distance as fallback) from the previous step, for the progress
    reward. Measures progress along the route, not the bee-line to the goal."""

    prev_actions: NDArray[np.float64] | None = None
    """(n_agents, action_dim) — raw actions from the previous step (for action rate)."""

    def reset(self, n_agents: int, distances: NDArray[np.float64]) -> None:
        """Reset reward state for a new episode.

        ``distances`` is the initial path-distance metric (see
        ``prev_nav_distances``); the caller passes the navmesh remaining-path
        distance (straight-line fallback) so the first progress delta is
        route-aware.
        """
        self.prev_velocities = None
        self.prev_accelerations = None
        self.prev_headings = None
        self.prev_heading_changes = None
        self.prev_nav_distances = distances.copy()
        self.prev_actions = None


def compute_rewards(
    positions: NDArray[np.float64],
    velocities: NDArray[np.float64],
    headings: NDArray[np.float64],
    goal_positions: NDArray[np.float64],
    preferred_speeds: NDArray[np.float64],
    active_mask: NDArray[np.bool_],
    collision_mask: NDArray[np.bool_],
    state: RewardState,
    config: RewardConfig,
    dt: float,
    *,
    current_distances: NDArray[np.float64] | None = None,
    wall_distances: NDArray[np.float64] | None = None,
    wall_collision_mask: NDArray[np.bool_] | None = None,
    agent_radii: NDArray[np.float64] | None = None,
    actions: NDArray[np.float64] | None = None,
    collision_velocities: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Compute per-agent rewards for one timestep.

    Parameters
    ----------
    positions : (n_agents, 2)
    velocities : (n_agents, 2)
    headings : (n_agents,)
    goal_positions : (n_agents, 2)
    preferred_speeds : (n_agents,)
    active_mask : (n_agents,) — True if agent is still active
    collision_mask : (n_agents,) — True if agent is currently in collision
    state : RewardState — mutable, updated in-place
    config : RewardConfig
    dt : float — timestep duration
    wall_distances : (n_agents,) optional — min distance to nearest wall per agent
    wall_collision_mask : (n_agents,) bool optional — True where the boundary
        enforcement corrected the agent this step (hard wall-contact signal)
    agent_radii : (n_agents,) optional — agent body radii (used for the
        graded agent-proximity penalty: contact distance = r_i + r_j)
    actions : (n_agents, action_dim) optional — raw policy output this step
    collision_velocities : (n_agents, 2) optional — pre-contact velocities (the
        policy's chosen approach velocity, before contact-force impulses). Used
        only by ``use_velocity_weighted_collision`` so the closing-speed weight
        reflects the approach speed the policy controls, not the post-bounce
        velocity. Falls back to ``velocities`` when not provided.

    Returns
    -------
    rewards : (n_agents,)
    reached_goal : (n_agents,) bool — True for agents that reached their goal this step
    """
    n_agents = len(positions)
    rewards = np.zeros(n_agents, dtype=np.float64)

    # Numerical safety for the velocity-weighted penalties below: sanitize the
    # pre-contact velocity snapshot -- a degenerate high-density pileup or a
    # transient non-finite policy output could otherwise inject NaN/Inf into the
    # reward and poison training -- and cap the impact speed that scales them so
    # a blowup can never produce an unbounded multiplier.
    max_impact_speed = 10.0  # m/s, a safety ceiling well above any physical closing
    if collision_velocities is not None:
        collision_velocities = np.nan_to_num(collision_velocities, nan=0.0, posinf=0.0, neginf=0.0)

    # Goal distances
    goal_diffs = goal_positions - positions
    goal_distances = np.linalg.norm(goal_diffs, axis=1)

    # --- Tier 1: Sparse ---

    # Goal reaching
    reached_goal = (goal_distances < config.goal_radius) & active_mask
    rewards[reached_goal] += config.goal_bonus

    # Collision penalty. Binary per-step by default; when
    # ``use_velocity_weighted_collision`` is set, scale by the closing speed
    # between contacting agents so high-speed impacts cost more than gentle
    # contact (still gated by ``collision_mask``).
    coll_active = collision_mask & active_mask
    if (
        config.use_velocity_weighted_collision
        and agent_radii is not None
        and n_agents >= 2
        and bool(coll_active.any())
    ):
        vel = collision_velocities if collision_velocities is not None else velocities
        sep = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # p_i - p_j
        sep_dist = np.sqrt(np.sum(sep**2, axis=-1))
        sep_unit = sep / np.maximum(sep_dist[..., np.newaxis], 1e-9)
        rel_vel = vel[:, np.newaxis, :] - vel[np.newaxis, :, :]  # v_i - v_j
        closing = -np.sum(rel_vel * sep_unit, axis=-1)  # >0 when i, j approach
        # Only count active neighbours within (slightly enlarged) contact range,
        # so the weight reflects the actual colliding partner; the 1.2x absorbs
        # the ellipse-vs-circle gap with the env's detect_collisions.
        contact_dist = 1.2 * (agent_radii[:, np.newaxis] + agent_radii[np.newaxis, :])
        eye = np.eye(n_agents, dtype=np.bool_)
        near = (
            (~eye)
            & (sep_dist <= contact_dist)
            & active_mask[:, np.newaxis]
            & active_mask[np.newaxis, :]
        )
        closing = np.where(near, np.maximum(closing, 0.0), 0.0)
        impact_speed = np.minimum(closing.max(axis=1), max_impact_speed)  # worst closing (capped)
        speed_scale = np.maximum(
            config.collision_speed_floor + config.collision_speed_scale * impact_speed, 0.0
        )
        coll_pen = config.collision_penalty * speed_scale[coll_active]
    else:
        coll_pen = config.collision_penalty
    # Cap the per-step collision penalty at a floor: velocity weighting may
    # DISCOUNT slow contact but not AMPLIFY fast contact below the cap (mirrors
    # crowdrl_torch.reward). cap=0.0 disables.
    if config.collision_penalty_cap < 0.0:
        coll_pen = np.maximum(coll_pen, config.collision_penalty_cap)
    rewards[coll_active] += coll_pen

    # Wall proximity penalty (smooth, distance-based)
    if (
        config.wall_proximity_penalty != 0.0
        and wall_distances is not None
        and agent_radii is not None
    ):
        threshold = agent_radii * config.wall_proximity_threshold
        wall_proximity = (wall_distances < threshold) & active_mask
        rewards[wall_proximity] += config.wall_proximity_penalty

    # Wall contact penalty (hard, per step while the boundary pushed the agent
    # back). Distinct from the proximity band; mirrors the agent collision
    # penalty -- deters using walls as a free brake.
    if config.wall_collision_penalty != 0.0 and wall_collision_mask is not None:
        wall_active = wall_collision_mask & active_mask
        if config.use_velocity_weighted_collision and bool(wall_active.any()):
            # No wall normal is available here, so weight by the agent's own
            # (pre-contact) speed: ramming a wall at speed costs more than
            # drifting into it. wall_collision_mask already implies into-wall
            # motion the boundary had to cancel.
            vel = collision_velocities if collision_velocities is not None else velocities
            own_speed = np.minimum(np.linalg.norm(vel, axis=1), max_impact_speed)
            wall_scale = np.maximum(
                config.collision_speed_floor + config.collision_speed_scale * own_speed, 0.0
            )
            rewards[wall_active] += config.wall_collision_penalty * wall_scale[wall_active]
        else:
            rewards[wall_active] += config.wall_collision_penalty

    # Agent proximity penalty (graded linear ramp, min over neighbours).
    # Penalty interpolates between ``near`` (at contact, r_i + r_j) and
    # ``far`` (at personal_space_radius). Each agent pays the penalty of its
    # most-penalised neighbour inside the zone.
    if (
        (config.agent_proximity_penalty_near != 0.0 or config.agent_proximity_penalty_far != 0.0)
        and agent_radii is not None
        and n_agents >= 2
    ):
        # Pairwise center-to-center distances (n, n)
        diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        pair_dist = np.sqrt(np.sum(diff**2, axis=-1))

        # Per-pair contact distance r_i + r_j (n, n)
        pair_contact = agent_radii[:, np.newaxis] + agent_radii[np.newaxis, :]

        # Linear interpolation factor t in [0, 1]: 0 at contact, 1 at boundary.
        denom = np.maximum(config.personal_space_radius - pair_contact, 1e-6)
        t = np.clip((pair_dist - pair_contact) / denom, 0.0, 1.0)
        pair_penalty = (1.0 - t) * config.agent_proximity_penalty_near + (
            t * config.agent_proximity_penalty_far
        )

        # Optionally weight by CLOSING speed (penalise approaching at speed, not
        # mere coexistence). ``diff`` here is p_j - p_i, so the closing speed
        # (>0 when i, j approach) is +(v_i - v_j) . unit(diff). Uses the
        # pre-contact (policy-chosen) velocities when supplied.
        if config.use_velocity_weighted_proximity:
            vel = collision_velocities if collision_velocities is not None else velocities
            diff_unit = diff / np.maximum(pair_dist[..., np.newaxis], 1e-9)
            rel_vel = vel[:, np.newaxis, :] - vel[np.newaxis, :, :]  # v_i - v_j
            closing = np.minimum(
                np.sum(rel_vel * diff_unit, axis=-1), max_impact_speed
            )  # (n, n), >0 approaching (capped)
            speed_w = np.maximum(
                config.proximity_speed_floor + config.proximity_speed_scale * closing, 0.0
            )
            pair_penalty = pair_penalty * speed_w

        # Mask: no self-pairs, only active-active pairs, only pairs inside zone.
        eye = np.eye(n_agents, dtype=np.bool_)
        valid_pair = (~eye) & active_mask[:, np.newaxis] & active_mask[np.newaxis, :]
        in_zone = pair_dist < config.personal_space_radius
        pair_penalty = np.where(valid_pair & in_zone, pair_penalty, 0.0)

        # Per-agent: most-negative penalty from any neighbour.
        proximity = pair_penalty.min(axis=1)
        rewards[active_mask] += proximity[active_mask]

    # Action rate penalty (change in raw policy output between steps)
    if config.action_rate_weight != 0.0 and actions is not None:
        if state.prev_actions is not None:
            action_change = np.linalg.norm(actions - state.prev_actions, axis=1)
            rewards[active_mask] += config.action_rate_weight * action_change[active_mask]

    # Existence penalty: every step alive costs you
    if config.existence_penalty != 0.0:
        rewards[active_mask] += config.existence_penalty

    # Progress reward (potential-based shaping): r = prev_dist - curr_dist.
    # Uses ``current_distances`` (navmesh remaining-PATH metric) when provided so
    # progress is measured along the route, not the straight-line bee-line to the
    # goal; falls back to straight-line goal distance otherwise.
    if state.prev_nav_distances is not None:
        curr = current_distances if current_distances is not None else goal_distances
        progress = state.prev_nav_distances - curr
        rewards[active_mask] += config.progress_weight * progress[active_mask]

    # Inverse distance to goal: 1 / (d + 1) — closer is better
    if config.inverse_distance_weight != 0.0:
        inv_dist = 1.0 / (goal_distances + 1.0)
        rewards[active_mask] += config.inverse_distance_weight * inv_dist[active_mask]

    # --- Tier 2: Smoothness ---
    if config.use_smoothness and state.prev_velocities is not None:
        # Current acceleration
        accelerations = (velocities - state.prev_velocities) / dt

        # Jerk penalty (change in acceleration)
        if state.prev_accelerations is not None:
            jerk = (accelerations - state.prev_accelerations) / dt
            jerk_magnitude = np.linalg.norm(jerk, axis=1)
            rewards[active_mask] += config.jerk_penalty_weight * jerk_magnitude[active_mask]

        # Angular acceleration penalty
        if state.prev_headings is not None:
            heading_change = headings - state.prev_headings
            # Normalise to [-pi, pi]
            heading_change = (heading_change + np.pi) % (2 * np.pi) - np.pi
            angular_vel = heading_change / dt

            if state.prev_heading_changes is not None:
                prev_angular_vel = state.prev_heading_changes / dt
                angular_accel = np.abs(angular_vel - prev_angular_vel)
                rewards[active_mask] += (
                    config.angular_accel_penalty_weight * angular_accel[active_mask]
                )

            state.prev_heading_changes = heading_change.copy()

        # Preferred speed deviation
        speeds = np.linalg.norm(velocities, axis=1)
        speed_dev = np.abs(speeds - preferred_speeds)
        rewards[active_mask] += config.speed_deviation_weight * speed_dev[active_mask]

        state.prev_accelerations = accelerations.copy()
    elif state.prev_velocities is not None:
        # Even without smoothness, update acceleration state
        state.prev_accelerations = (velocities - state.prev_velocities) / dt

    # Update state for next step
    state.prev_velocities = velocities.copy()
    state.prev_headings = headings.copy()
    state.prev_nav_distances = (
        current_distances if current_distances is not None else goal_distances
    ).copy()
    if actions is not None:
        state.prev_actions = actions.copy()

    return rewards, reached_goal
