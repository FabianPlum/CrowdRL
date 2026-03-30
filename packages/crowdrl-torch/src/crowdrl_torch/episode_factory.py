"""CPU-side episode generation for the PyTorch batched environment.

Wraps the existing ``crowdrl-env`` geometry generator and spawner to
produce the dict format expected by ``BatchedTorchEnv.make_episode_fn``.

This runs on CPU in a thread pool — Shapely and rejection sampling
stay on CPU, only the resulting arrays go to GPU.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from crowdrl_core.geometry import build_navmesh, extract_wall_segments
from crowdrl_core.navmesh import shortest_path
from crowdrl_env.crowd_env import CrowdEnvConfig
from crowdrl_env.geometry_generator import GeometryConfig, generate_geometry
from crowdrl_env.solvability import verify_solvability
from crowdrl_env.spawner import spawn_agents


def make_episode_factory(
    env_config: CrowdEnvConfig,
) -> callable:
    """Create a ``make_episode_fn(seed) -> dict`` for BatchedTorchEnv.

    The returned function generates one episode on CPU: random geometry,
    agent spawning, solvability check. Returns a flat dict of NumPy arrays.

    Parameters
    ----------
    env_config : CrowdEnvConfig
        Environment configuration (geometry, spawn, solvability settings).

    Returns
    -------
    make_episode_fn : callable
        ``(seed: int) -> dict[str, NDArray]``
    """

    def make_episode(seed: int) -> dict[str, NDArray]:
        rng = np.random.default_rng(seed)

        for _attempt in range(env_config.max_regeneration_attempts):
            # Pick tier
            if env_config.geometry_tiers is not None:
                tier = rng.choice(env_config.geometry_tiers)
                geom_config = GeometryConfig(
                    tier=tier,
                    min_side=env_config.geometry.min_side,
                    max_side=env_config.geometry.max_side,
                    corridor_width_range=env_config.geometry.corridor_width_range,
                    corridor_length_range=env_config.geometry.corridor_length_range,
                    bottleneck_aperture_range=env_config.geometry.bottleneck_aperture_range,
                    bottleneck_depth_range=env_config.geometry.bottleneck_depth_range,
                    branch_width_range=env_config.geometry.branch_width_range,
                    branch_length_range=env_config.geometry.branch_length_range,
                )
            else:
                geom_config = env_config.geometry

            geom = generate_geometry(rng, geom_config)
            navmesh = build_navmesh(geom.polygon)
            wall_segments = extract_wall_segments(geom.polygon)

            # Spawn agents
            spawn_result = spawn_agents(
                rng,
                geom.spawn_regions,
                geom.goal_regions,
                env_config.spawn,
            )

            if spawn_result.n_agents == 0:
                continue

            # Solvability check
            agent_radii = np.maximum(
                spawn_result.shoulder_widths,
                spawn_result.chest_depths,
            )
            solvable_mask = verify_solvability(
                navmesh,
                spawn_result.positions,
                spawn_result.goal_positions,
                agent_radii,
                env_config.solvability_mode,
                env_config.max_unsolvable_fraction,
            )

            if solvable_mask is None:
                continue

            # Filter to solvable agents
            if not np.all(solvable_mask):
                idx = np.where(solvable_mask)[0]
                if len(idx) == 0:
                    continue
                positions = spawn_result.positions[idx]
                velocities = spawn_result.velocities[idx]
                torso_orientations = spawn_result.torso_orientations[idx]
                head_orientations = spawn_result.head_orientations[idx]
                shoulder_widths = spawn_result.shoulder_widths[idx]
                chest_depths = spawn_result.chest_depths[idx]
                goal_positions = spawn_result.goal_positions[idx]
                preferred_speeds = spawn_result.preferred_speeds[idx]
            else:
                positions = spawn_result.positions
                velocities = spawn_result.velocities
                torso_orientations = spawn_result.torso_orientations
                head_orientations = spawn_result.head_orientations
                shoulder_widths = spawn_result.shoulder_widths
                chest_depths = spawn_result.chest_depths
                goal_positions = spawn_result.goal_positions
                preferred_speeds = spawn_result.preferred_speeds

            # Pre-compute funnel waypoints per agent (CPU, amortised over episode)
            n_agents = len(positions)
            max_wp = env_config.obs.navmesh_max_waypoints
            wp_array = np.zeros((n_agents, max_wp, 2), dtype=np.float64)
            wp_counts = np.zeros(n_agents, dtype=np.int32)
            wp_path_lengths = np.zeros((n_agents, max_wp), dtype=np.float64)

            if env_config.obs.use_navmesh and navmesh is not None:
                for i in range(n_agents):
                    radius = float(max(shoulder_widths[i], chest_depths[i]))
                    path = shortest_path(navmesh, positions[i], goal_positions[i], radius)
                    if path is not None and len(path) >= 2:
                        # Drop start position, keep intermediate + goal
                        wps = path[1:]
                        n_wp = min(len(wps), max_wp)
                        for k in range(n_wp):
                            wp_array[i, k] = wps[k]
                        wp_counts[i] = n_wp

                        # Cumulative remaining path length from each waypoint
                        # to the goal (last waypoint has distance 0)
                        for k in range(n_wp - 1, -1, -1):
                            if k == n_wp - 1:
                                wp_path_lengths[i, k] = 0.0
                            else:
                                seg = float(np.linalg.norm(wps[k + 1] - wps[k]))
                                wp_path_lengths[i, k] = wp_path_lengths[i, k + 1] + seg

            return {
                "positions": positions,
                "velocities": velocities,
                "torso_orientations": torso_orientations,
                "head_orientations": head_orientations,
                "shoulder_widths": shoulder_widths,
                "chest_depths": chest_depths,
                "goal_positions": goal_positions,
                "preferred_speeds": preferred_speeds,
                "wall_segments": wall_segments,
                "waypoints": wp_array,
                "n_waypoints": wp_counts,
                "waypoint_path_lengths": wp_path_lengths,
                "tier": geom.tier.name,
            }

        raise RuntimeError(
            f"Failed to generate solvable episode after "
            f"{env_config.max_regeneration_attempts} attempts"
        )

    return make_episode
