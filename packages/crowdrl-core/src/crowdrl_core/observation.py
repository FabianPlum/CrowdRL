"""Observation builder: assembles the full observation vector from WorldState.

**This is THE single function that must be identical between training and deployment.**
Any discrepancy here means the policy sees a different world and produces wrong actions.

Observation layout (all in egocentric frame):
  Ego state (7D):
    goal_dir (2), velocity (2), heading (1), torso_angle (1), head_angle_rel_torso (1)
  Social (K×7 = 56D):
    per-neighbour: rel_pos (2), rel_vel (2), body_orient (1), body_dims (2)
  Raycasts (N or N×2):
    single-channel: N normalised distances
    two-channel: N × (distance, hit_type)
  Navmesh (optional, 3D):
    next_waypoint_dir (2), path_deviation (1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from crowdrl_core.navmesh import next_waypoint_direction, path_deviation
from crowdrl_core.sensing import RaycastConfig, cast_rays, knn_social
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

    # Rotation matrix: global → ego frame
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

    ego_state = np.array([
        goal_dir_ego[0], goal_dir_ego[1],
        vel_ego[0], vel_ego[1],
        speed,
        torso_angle,
        head_rel_torso,
    ], dtype=np.float64)

    # --- Social sensing (K × 7D) ---
    social = knn_social(world, agent_idx, k=config.k_neighbours)
    social_flat = social.flatten()

    # --- Raycasts (N or N×2) ---
    rays = cast_rays(world, agent_idx, config.raycast)
    rays_flat = rays.flatten()

    # --- Navmesh signals (optional, 3D) ---
    parts = [ego_state, social_flat, rays_flat]

    if config.use_navmesh and world.navmesh is not None:
        wp_dir = next_waypoint_direction(world.navmesh, ego_pos, goal)
        p_dev = path_deviation(world.navmesh, ego_pos, goal)

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
    """Build observations for all active agents in the world.

    Returns
    -------
    obs : (n_active, obs_dim) array
    """
    obs_list = []
    for i in range(world.n_agents):
        if world.active_mask is not None and not world.active_mask[i]:
            continue
        obs_list.append(build_observation(world, i, config))

    if not obs_list:
        return np.empty((0, config.obs_dim), dtype=np.float64)

    return np.stack(obs_list)
