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

    # Wall collision penalty
    wall_proximity_penalty: float = -0.1
    """Penalty for agents too close to walls (smooth, distance-based)."""

    wall_proximity_threshold: float = 1.5
    """Wall proximity threshold as a multiple of agent radius."""

    # Agent proximity penalty (graded linear ramp, learned collision avoidance)
    # The penalty per step is linearly interpolated on the center-to-center
    # distance between an agent and its nearest active neighbour:
    #   - at contact distance (r_i + r_j): ``agent_proximity_penalty_near``
    #   - at ``personal_space_radius``:    ``agent_proximity_penalty_far``
    #   - beyond ``personal_space_radius``: no penalty
    # This provides a continuous gradient for the policy to maintain personal
    # space, while the binary ``collision_penalty`` handles the hard "you
    # touched someone" signal on top. See Project Plan v6, Section 3.2.
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
    action_rate_weight: float = 0.0
    """Weight for penalising large changes in raw policy output between steps.
    Negative value (e.g. -0.05). 0.0 = disabled."""

    # Tier 2: smoothness
    use_smoothness: bool = True
    """Whether to apply Tier 2 smoothness penalties."""

    jerk_penalty_weight: float = -0.000001
    """Weight for acceleration change (jerk) penalty.  Kept very small
    because jerk scales as 1/dt^2; with dt=0.01 even a 0.1 m/s
    velocity glitch produces jerk ~1000 m/s^3."""

    angular_accel_penalty_weight: float = -0.0001
    """Weight for angular acceleration penalty."""

    speed_deviation_weight: float = -0.001
    """Weight for deviation from preferred speed. Kept low to avoid
    dominating the reward budget in congested scenarios where agents
    must slow down."""

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

    prev_goal_distances: NDArray[np.float64] | None = None
    """(n_agents,) — distances to goal from the previous step (for progress)."""

    prev_actions: NDArray[np.float64] | None = None
    """(n_agents, action_dim) — raw actions from the previous step (for action rate)."""

    def reset(self, n_agents: int, goal_distances: NDArray[np.float64]) -> None:
        """Reset reward state for a new episode."""
        self.prev_velocities = None
        self.prev_accelerations = None
        self.prev_headings = None
        self.prev_heading_changes = None
        self.prev_goal_distances = goal_distances.copy()
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
    wall_distances: NDArray[np.float64] | None = None,
    agent_radii: NDArray[np.float64] | None = None,
    actions: NDArray[np.float64] | None = None,
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
    agent_radii : (n_agents,) optional — agent body radii (used for the
        graded agent-proximity penalty: contact distance = r_i + r_j)
    actions : (n_agents, action_dim) optional — raw policy output this step

    Returns
    -------
    rewards : (n_agents,)
    reached_goal : (n_agents,) bool — True for agents that reached their goal this step
    """
    n_agents = len(positions)
    rewards = np.zeros(n_agents, dtype=np.float64)

    # Goal distances
    goal_diffs = goal_positions - positions
    goal_distances = np.linalg.norm(goal_diffs, axis=1)

    # --- Tier 1: Sparse ---

    # Goal reaching
    reached_goal = (goal_distances < config.goal_radius) & active_mask
    rewards[reached_goal] += config.goal_bonus

    # Collision penalty
    rewards[collision_mask & active_mask] += config.collision_penalty

    # Wall proximity penalty (smooth, distance-based)
    if (
        config.wall_proximity_penalty != 0.0
        and wall_distances is not None
        and agent_radii is not None
    ):
        threshold = agent_radii * config.wall_proximity_threshold
        wall_proximity = (wall_distances < threshold) & active_mask
        rewards[wall_proximity] += config.wall_proximity_penalty

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

    # Progress reward (potential-based shaping): r = prev_dist - curr_dist
    if state.prev_goal_distances is not None:
        progress = state.prev_goal_distances - goal_distances
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
    state.prev_goal_distances = goal_distances.copy()
    if actions is not None:
        state.prev_actions = actions.copy()

    return rewards, reached_goal
