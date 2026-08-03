"""CPU-side episode generation for the PyTorch batched environment.

Wraps the existing ``crowdrl-env`` geometry generator and spawner to
produce the dict format expected by ``BatchedTorchEnv.make_episode_fn``.

This runs on CPU in a thread pool — Shapely and rejection sampling
stay on CPU, only the resulting arrays go to GPU.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from crowdrl_core.geometry import build_navmesh, extract_wall_segments
from crowdrl_core.navmesh import JUPEDSIM_ROUTER_INSET, first_waypoint_headings, shortest_path
from crowdrl_env.crowd_env import CrowdEnvConfig
from crowdrl_env.geometry_generator import GeometryConfig, generate_geometry
from crowdrl_env.solvability import verify_solvability
from crowdrl_env.spawner import SpawnShortfallError, spawn_agents

logger = logging.getLogger(__name__)


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

        for attempt in range(env_config.max_regeneration_attempts):
            is_last_attempt = attempt == env_config.max_regeneration_attempts - 1
            # Pick tier
            if env_config.geometry_tiers is not None:
                weights = env_config.tier_weights
                if weights is not None:
                    p = np.array(weights, dtype=np.float64)
                    p = p / p.sum()
                else:
                    p = None
                tier = rng.choice(env_config.geometry_tiers, p=p)
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
                    max_wall_segments=env_config.geometry.max_wall_segments,
                    min_passage_width=env_config.geometry.min_passage_width,
                )
            else:
                geom_config = env_config.geometry

            geom = generate_geometry(rng, geom_config)
            navmesh = build_navmesh(geom.polygon)
            wall_segments = extract_wall_segments(
                geom.polygon,
                max_segments=geom_config.max_wall_segments,
            )

            # Guard: if simplification still couldn't bring segments under
            # budget, discard this geometry and regenerate.
            if len(wall_segments) > geom_config.max_wall_segments:
                continue

            # Spawn agents
            # Must match CrowdEnv exactly (see its call site): walkable enables both
            # the body-radius wall clearance and entry-zone dilation.
            spawn_result = spawn_agents(
                rng,
                geom.spawn_regions,
                geom.goal_regions,
                env_config.spawn,
                walkable=geom.polygon,
            )

            if spawn_result.n_agents == 0:
                continue

            # Bail early on a spawn shortfall -- same contract as CrowdEnv, so the
            # two engines deliver the same crowd size for a given seed.
            if (
                spawn_result.is_short
                and not is_last_attempt
                and env_config.spawn.spawn_shortfall_policy == "regenerate"
            ):
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
                env_config.solvability_clearance_factor,
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

            # Final delivered count, after both drop paths (placement + solvability).
            requested_n = spawn_result.requested_n
            if len(positions) < requested_n:
                policy = env_config.spawn.spawn_shortfall_policy
                if policy == "raise":
                    raise SpawnShortfallError(
                        requested_n,
                        len(positions),
                        spawn_result.spawn_area_m2,
                        spawn_result.capacity,
                    )
                if not is_last_attempt and policy == "regenerate":
                    continue
                logger.warning(
                    "agent-count shortfall after %d attempt(s): %s after_solvability=%d tier=%s",
                    attempt + 1,
                    spawn_result.shortfall_summary,
                    len(positions),
                    geom.tier.name,
                )

            # Pre-compute funnel waypoints per agent (CPU, amortised over episode)
            n_agents = len(positions)
            max_wp = env_config.obs.navmesh_max_waypoints
            wp_array = np.zeros((n_agents, max_wp, 2), dtype=np.float64)
            wp_counts = np.zeros(n_agents, dtype=np.int32)
            wp_path_lengths = np.zeros((n_agents, max_wp), dtype=np.float64)

            if env_config.obs.use_navmesh and navmesh is not None:
                jps_routing = env_config.obs.use_jupedsim_style_routing
                path_over_cap = False
                for i in range(n_agents):
                    # Under the jupedsim-style contract the stored path uses the
                    # router's fixed 0.2 m portal inset, not the body radius --
                    # the waypoints the deployed policy will actually be served.
                    radius = (
                        JUPEDSIM_ROUTER_INSET
                        if jps_routing
                        else float(max(shoulder_widths[i], chest_depths[i]))
                    )
                    path = shortest_path(navmesh, positions[i], goal_positions[i], radius)
                    if path is not None and len(path) >= 2:
                        # Drop start position, keep intermediate + goal
                        wps = path[1:]
                        if len(wps) > max_wp:
                            # This agent's shortest path needs more waypoints than
                            # we can store. Truncating would strand it short of its
                            # goal, so discard the geometry and regenerate instead
                            # (same attempt budget as every other regen trigger).
                            # With max_wp=1024 this is a safety net that should
                            # never fire for 2D pedestrian geometry.
                            path_over_cap = True
                            break
                        n_wp = len(wps)
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

                if path_over_cap:
                    continue

                # Orient agents toward their first navmesh waypoint rather than
                # the global goal (which may sit behind a wall relative to that
                # waypoint). Falls back to the goal bearing per-agent when no
                # path exists. Single-sourced with the numpy env via
                # first_waypoint_headings so spawn orientation matches in both.
                radii = (
                    np.full(n_agents, JUPEDSIM_ROUTER_INSET)
                    if jps_routing
                    else np.maximum(shoulder_widths, chest_depths)
                )
                torso_orientations = first_waypoint_headings(
                    navmesh, positions, goal_positions, radii
                )
                head_orientations = torso_orientations.copy()

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
