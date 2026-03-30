"""Observation builder: assembles the full observation vector from WorldState.

**This is THE single function that must be identical between training and deployment.**
Any discrepancy here means the policy sees a different world and produces wrong actions.

Observation layout (all in egocentric frame):
  Ego state (7D):
    goal_dir (2), velocity (2), heading (1), torso_angle (1), head_angle_rel_torso (1)
  Social (K*7 = 56D):
    per-neighbour: rel_pos (2), rel_vel (2), body_orient (1), body_dims (2)
  Raycasts (N or N*2):
    single-channel: N normalised distances
    two-channel: N * (distance, hit_type)
  Navmesh (optional, 3D):
    next_waypoint_dir (2), path_deviation (1)

Provides both per-agent ``build_observation()`` (deployment) and vectorized
``build_observations_batch()`` (training) that produce numerically identical results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from crowdrl_core.navmesh import next_waypoint_direction, path_deviation
from crowdrl_core.sensing import (
    RaycastConfig,
    cast_rays,
    cast_rays_batch,
    knn_social,
    knn_social_batch,
)
from crowdrl_core.world_state import WorldState


@dataclass(frozen=True)
class ObsConfig:
    """Configuration for the observation builder."""

    k_neighbours: int = 8
    """Number of nearest neighbours for social sensing."""

    raycast: RaycastConfig = RaycastConfig()
    """Raycast sensor configuration."""

    use_navmesh: bool = False
    """Whether to include navmesh signals (next-waypoint direction + path deviation)."""

    navmesh_max_waypoints: int = 16
    """Maximum number of pre-computed waypoints per agent for GPU waypoint lookup."""

    @property
    def obs_dim(self) -> int:
        """Total observation dimensionality."""
        ego = 7
        social = self.k_neighbours * 7
        rays = self.raycast.n_rays * (2 if self.raycast.two_channel else 1)
        nav = 3 if self.use_navmesh else 0
        return ego + social + rays + nav


def build_observation(
    world: WorldState,
    agent_idx: int,
    config: ObsConfig,
) -> NDArray[np.float64]:
    """Build the full observation vector for a single agent.

    This is the single function used by both training (crowdrl-env) and
    deployment (crowdrl-jupedsim). Never duplicate this logic.

    Parameters
    ----------
    world : WorldState
    agent_idx : int
    config : ObsConfig

    Returns
    -------
    obs : (obs_dim,) array
    """
    ego_pos = world.positions[agent_idx]
    ego_vel = world.velocities[agent_idx]
    ego_heading = world.torso_orientations[agent_idx]
    ego_head = world.head_orientations[agent_idx]
    goal = world.goal_positions[agent_idx]

    # Rotation matrix: global -> ego frame
    cos_h, sin_h = np.cos(-ego_heading), np.sin(-ego_heading)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float64)

    # --- Ego state (7D) ---
    # Goal direction in ego frame (unit vector)
    goal_diff = goal - ego_pos
    goal_dist = np.linalg.norm(goal_diff)
    if goal_dist > 1e-10:
        goal_dir_ego = rot @ (goal_diff / goal_dist)
    else:
        goal_dir_ego = np.zeros(2, dtype=np.float64)

    # Velocity in ego frame
    vel_ego = rot @ ego_vel

    # Heading is 0 in ego frame by construction (it's the reference direction)
    # But we include the speed as a proxy
    speed = np.linalg.norm(ego_vel)

    # Torso angle (relative to velocity direction, or 0 if standing still)
    torso_angle = 0.0  # In ego frame, torso IS the reference

    # Head angle relative to torso
    head_rel_torso = ego_head - ego_heading
    head_rel_torso = (head_rel_torso + np.pi) % (2 * np.pi) - np.pi

    ego_state = np.array(
        [
            goal_dir_ego[0],
            goal_dir_ego[1],
            vel_ego[0],
            vel_ego[1],
            speed,
            torso_angle,
            head_rel_torso,
        ],
        dtype=np.float64,
    )

    # --- Social sensing (K * 7D) ---
    social = knn_social(world, agent_idx, k=config.k_neighbours)
    social_flat = social.flatten()

    # --- Raycasts (N or N*2) ---
    rays = cast_rays(world, agent_idx, config.raycast)
    rays_flat = rays.flatten()

    # --- Navmesh signals (optional, 3D) ---
    parts = [ego_state, social_flat, rays_flat]

    if config.use_navmesh and world.navmesh is not None:
        # Use the larger body half-dimension as the clearance radius so the
        # path stays clear of wall corners.
        agent_radius = float(
            max(
                world.shoulder_widths[agent_idx],
                world.chest_depths[agent_idx],
            )
        )
        wp_dir = next_waypoint_direction(world.navmesh, ego_pos, goal, agent_radius)
        p_dev = path_deviation(world.navmesh, ego_pos, goal, agent_radius)

        if wp_dir is not None and p_dev is not None:
            # Transform waypoint direction to ego frame
            wp_dir_ego = rot @ wp_dir
            nav_signal = np.array([wp_dir_ego[0], wp_dir_ego[1], p_dev], dtype=np.float64)
        else:
            nav_signal = np.zeros(3, dtype=np.float64)

        parts.append(nav_signal)

    return np.concatenate(parts)


def build_observations_batch(
    world: WorldState,
    config: ObsConfig,
) -> NDArray[np.float64]:
    """Build observations for all agents using vectorized batch operations.

    Produces numerically identical results to calling ``build_observation()``
    per agent, but uses batch KNN and raycasting for much higher throughput.

    Returns
    -------
    obs : (n_agents, obs_dim) array — zero for inactive agents
    """
    n = world.n_agents
    obs_dim = config.obs_dim

    if n == 0:
        return np.empty((0, obs_dim), dtype=np.float64)

    # Determine which agents to build observations for
    if world.active_mask is not None:
        active_idx = np.where(world.active_mask)[0].astype(np.intp)
    else:
        active_idx = np.arange(n, dtype=np.intp)

    # Output array (zero for inactive agents)
    obs = np.zeros((n, obs_dim), dtype=np.float64)

    if len(active_idx) == 0:
        return obs

    M = len(active_idx)

    # --- Vectorized ego state (M, 7) ---
    ego_pos = world.positions[active_idx]  # (M, 2)
    ego_vel = world.velocities[active_idx]  # (M, 2)
    ego_heading = world.torso_orientations[active_idx]  # (M,)
    ego_head = world.head_orientations[active_idx]  # (M,)
    goals = world.goal_positions[active_idx]  # (M, 2)

    cos_h = np.cos(-ego_heading)  # (M,)
    sin_h = np.sin(-ego_heading)  # (M,)

    # Goal direction in ego frame
    goal_diff = goals - ego_pos  # (M, 2)
    goal_dist = np.sqrt(np.sum(goal_diff**2, axis=-1))  # (M,)
    safe_dist = np.where(goal_dist > 1e-10, goal_dist, 1.0)
    goal_unit = goal_diff / safe_dist[:, np.newaxis]
    goal_unit = np.where((goal_dist > 1e-10)[:, np.newaxis], goal_unit, 0.0)

    goal_dir_ego_x = cos_h * goal_unit[:, 0] - sin_h * goal_unit[:, 1]
    goal_dir_ego_y = sin_h * goal_unit[:, 0] + cos_h * goal_unit[:, 1]

    # Velocity in ego frame
    vel_ego_x = cos_h * ego_vel[:, 0] - sin_h * ego_vel[:, 1]
    vel_ego_y = sin_h * ego_vel[:, 0] + cos_h * ego_vel[:, 1]

    # Speed
    speed = np.sqrt(np.sum(ego_vel**2, axis=-1))

    # Torso angle in ego frame is 0 by construction
    torso_angle = np.zeros(M, dtype=np.float64)

    # Head angle relative to torso
    head_rel_torso = (ego_head - ego_heading + np.pi) % (2 * np.pi) - np.pi

    ego_state = np.column_stack(
        [
            goal_dir_ego_x,
            goal_dir_ego_y,
            vel_ego_x,
            vel_ego_y,
            speed,
            torso_angle,
            head_rel_torso,
        ]
    )  # (M, 7)

    # --- Batch social sensing (M, K*7) ---
    social = knn_social_batch(world, active_idx, k=config.k_neighbours)  # (M, K, 7)
    social_flat = social.reshape(M, -1)  # (M, K*7)

    # --- Batch raycasting (M, R) or (M, R*2) ---
    rays = cast_rays_batch(world, active_idx, config.raycast)  # (M, R) or (M, R, 2)
    rays_flat = rays.reshape(M, -1)

    # --- Assemble ---
    offset = 0
    obs[active_idx, offset : offset + 7] = ego_state
    offset += 7

    social_dim = config.k_neighbours * 7
    obs[active_idx, offset : offset + social_dim] = social_flat
    offset += social_dim

    ray_dim = config.raycast.n_rays * (2 if config.raycast.two_channel else 1)
    obs[active_idx, offset : offset + ray_dim] = rays_flat
    offset += ray_dim

    # --- Navmesh signals (still per-agent for now as navmesh queries are complex) ---
    if config.use_navmesh and world.navmesh is not None:
        for idx_pos, i in enumerate(active_idx):
            agent_radius = float(max(world.shoulder_widths[i], world.chest_depths[i]))
            wp_dir = next_waypoint_direction(
                world.navmesh, world.positions[i], world.goal_positions[i], agent_radius
            )
            p_dev = path_deviation(
                world.navmesh, world.positions[i], world.goal_positions[i], agent_radius
            )
            if wp_dir is not None and p_dev is not None:
                wp_ego_x = cos_h[idx_pos] * wp_dir[0] - sin_h[idx_pos] * wp_dir[1]
                wp_ego_y = sin_h[idx_pos] * wp_dir[0] + cos_h[idx_pos] * wp_dir[1]
                obs[i, offset : offset + 3] = [wp_ego_x, wp_ego_y, p_dev]

    return obs
