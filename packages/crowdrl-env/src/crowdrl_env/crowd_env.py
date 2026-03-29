"""CrowdEnv: multi-agent Gymnasium environment for pedestrian navigation.

This is the Gymnasium wrapper that ties together geometry generation, agent
spawning, solvability verification, physics integration, and reward computation.

The environment manages all agents internally and exposes a batched interface:
- Observations: (n_agents, obs_dim)
- Actions: (n_agents, action_dim)
- Rewards: (n_agents,)

Designed for MAPPO with parameter sharing: one policy network, many agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from crowdrl_core.action import ActionConfig, interpret_actions_batch
from crowdrl_core.collision import (
    compute_contact_forces,
    compute_min_wall_distances,
    detect_collisions,
    enforce_wall_boundaries,
)
from crowdrl_core.geometry import build_navmesh, extract_wall_segments
from crowdrl_core.observation import ObsConfig, build_observations_batch
from crowdrl_core.world_state import WorldState

from crowdrl_env.geometry_generator import GeometryConfig, GeometryTier, generate_geometry
from crowdrl_env.reward import RewardConfig, RewardState, compute_rewards
from crowdrl_env.solvability import SolvabilityMode, filter_by_solvability, verify_solvability
from crowdrl_env.spawner import SpawnConfig, spawn_agents


@dataclass(frozen=True)
class CrowdEnvConfig:
    """Full configuration for the CrowdRL training environment."""

    # Geometry
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    geometry_tiers: list[GeometryTier] | None = None
    """If set, randomly pick from these tiers each episode (overrides geometry.tier)."""

    # Spawning
    spawn: SpawnConfig = field(default_factory=SpawnConfig)

    # Solvability
    solvability_mode: SolvabilityMode = SolvabilityMode.PRUNE
    max_unsolvable_fraction: float = 0.3
    max_regeneration_attempts: int = 10

    # Observation
    obs: ObsConfig = field(default_factory=ObsConfig)

    # Action
    action: ActionConfig = field(default_factory=ActionConfig)

    # Reward
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Physics
    dt: float = 0.01
    """Timestep duration (seconds)."""
    contact_stiffness: float = 2000.0
    contact_damping: float = 50.0
    velocity_damping: float = 0.8
    """Velocity damping factor: v_new = damping * v_desired + (1-damping) * v_old."""

    max_speed_multiplier: float = 2.0
    """Velocity magnitude clamp as a multiple of action.max_speed.

    After contact forces are applied, agent speeds are clamped to
    ``max_speed_multiplier * action.max_speed``.  This prevents contact
    forces from launching agents at unrealistic velocities while still
    allowing brief bursts above the desired-speed ceiling (e.g. being
    pushed by a crowd).
    """

    # Episode
    max_steps: int = 5000
    """Maximum timesteps per episode."""


class CrowdEnv(gym.Env):
    """Multi-agent pedestrian navigation environment.

    Manages N agents in a procedurally generated 2D polygon.
    All agents share the same observation/action spaces (MAPPO parameter sharing).

    The environment returns batched arrays for all agents. Agents that
    have reached their goal or been deactivated receive zero observations
    and zero rewards.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: CrowdEnvConfig | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.config = config or CrowdEnvConfig()
        self.render_mode = render_mode

        # Spaces are per-agent (MAPPO treats each agent identically)
        obs_dim = self.config.obs.obs_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.config.action.action_dim,), dtype=np.float64
        )

        self._rng = np.random.default_rng(seed)
        self._world: WorldState | None = None
        self._active_mask: NDArray[np.bool_] | None = None
        self._preferred_speeds: NDArray[np.float64] | None = None
        self._reward_state = RewardState()
        self._step_count = 0
        self._n_agents = 0

    @property
    def n_agents(self) -> int:
        """Current number of agents in the episode."""
        return self._n_agents

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[NDArray[np.float64], dict]:
        """Reset the environment: generate geometry, spawn agents.

        Returns
        -------
        observations : (n_agents, obs_dim) array
        info : dict with episode metadata
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Generate geometry (with optional tier randomisation)
        world, preferred_speeds, geom_metadata = self._generate_episode()

        self._world = world
        self._preferred_speeds = preferred_speeds
        self._n_agents = world.n_agents
        self._active_mask = np.ones(self._n_agents, dtype=np.bool_)
        self._world.active_mask = self._active_mask
        self._step_count = 0

        # Initialise reward state
        goal_distances = np.linalg.norm(world.goal_positions - world.positions, axis=1)
        self._reward_state.reset(self._n_agents, goal_distances)

        # Build initial observations
        obs = self._build_all_observations()

        info = {
            "n_agents": self._n_agents,
            "geometry_tier": geom_metadata.get("tier"),
            "geometry_shape": geom_metadata.get("shape"),
        }

        return obs, info

    def step(
        self,
        actions: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_], dict
    ]:
        """Execute one timestep.

        Parameters
        ----------
        actions : (n_agents, action_dim) array
            Raw policy output for each agent (values in [-1, 1]).

        Returns
        -------
        observations : (n_agents, obs_dim)
        rewards : (n_agents,)
        terminated : (n_agents,) — True if agent reached goal
        truncated : (n_agents,) — True if episode time limit reached
        info : dict
        """
        assert self._world is not None, "Call reset() before step()"
        assert actions.shape == (self._n_agents, self.config.action.action_dim), (
            f"Expected actions shape ({self._n_agents}, {self.config.action.action_dim}), "
            f"got {actions.shape}"
        )

        self._step_count += 1
        cfg = self.config

        # --- 1. Interpret actions → desired velocities and orientations ---
        batch_result = interpret_actions_batch(
            actions,
            self._world.torso_orientations,
            self._world.torso_orientations,
            self._world.head_orientations,
            cfg.action,
        )

        # --- 2. Apply velocity update (damped blending) — vectorized ---
        mask = self._active_mask
        self._world.velocities[mask] = (
            cfg.velocity_damping * batch_result.desired_velocities[mask]
            + (1.0 - cfg.velocity_damping) * self._world.velocities[mask]
        )
        self._world.torso_orientations[mask] = batch_result.new_torso_orientations[mask]
        self._world.head_orientations[mask] = batch_result.new_head_orientations[mask]

        # --- 3. Collision detection and contact forces ---
        # Detect collisions once, pass to both force computation and reward
        collisions = detect_collisions(self._world)
        contact_forces = compute_contact_forces(
            self._world,
            stiffness=cfg.contact_stiffness,
            damping=cfg.contact_damping,
            collisions=collisions,
        )

        # Collision mask for reward computation
        collision_mask = np.zeros(self._n_agents, dtype=np.bool_)
        if collisions:
            col_arr = np.asarray(collisions)
            col_i = col_arr[:, 0].astype(np.intp)
            col_j = col_arr[:, 1].astype(np.intp)
            collision_mask[col_i] = True
            collision_mask[col_j] = True

        # --- 4. Physics integration (semi-implicit Euler) ---
        # Apply contact forces as velocity impulse — vectorized
        self._world.velocities[mask] += contact_forces[mask] * cfg.dt

        # Clamp velocity magnitudes to prevent contact-force blow-up
        max_vel = cfg.max_speed_multiplier * cfg.action.max_speed
        speeds = np.linalg.norm(self._world.velocities[mask], axis=1)
        too_fast = speeds > max_vel
        if np.any(too_fast):
            scale = np.where(too_fast, max_vel / np.maximum(speeds, 1e-10), 1.0)
            self._world.velocities[mask] *= scale[:, np.newaxis]

        # Position update
        self._world.positions[self._active_mask] += (
            self._world.velocities[self._active_mask] * cfg.dt
        )

        # Wall boundary enforcement
        enforce_wall_boundaries(self._world)

        # --- 5. Compute rewards ---
        # Wall distances for proximity penalty
        wall_distances = compute_min_wall_distances(self._world)
        agent_radii = np.maximum(self._world.shoulder_widths, self._world.chest_depths)

        rewards, reached_goal = compute_rewards(
            positions=self._world.positions,
            velocities=self._world.velocities,
            headings=self._world.torso_orientations,
            goal_positions=self._world.goal_positions,
            preferred_speeds=self._preferred_speeds,
            active_mask=self._active_mask,
            collision_mask=collision_mask,
            state=self._reward_state,
            config=cfg.reward,
            dt=cfg.dt,
            wall_distances=wall_distances,
            agent_radii=agent_radii,
            actions=actions,
        )

        # --- 6. Update active mask ---
        # Deactivate agents that reached their goal
        newly_done = reached_goal & self._active_mask
        self._active_mask[newly_done] = False

        # Zero out velocities for inactive agents
        self._world.velocities[~self._active_mask] = 0.0

        # --- 7. Termination / truncation ---
        terminated = reached_goal.copy()
        truncated = np.zeros(self._n_agents, dtype=np.bool_)

        episode_over = False
        if self._step_count >= cfg.max_steps:
            # Timeout: all remaining active agents are truncated
            still_active = self._active_mask.copy()
            truncated[still_active] = True
            rewards[still_active] += cfg.reward.timeout_penalty
            self._active_mask[:] = False
            episode_over = True

        if not np.any(self._active_mask):
            episode_over = True

        # --- 8. Build observations ---
        obs = self._build_all_observations()

        info = {
            "step": self._step_count,
            "n_active": int(np.sum(self._active_mask)),
            "n_collisions": len(collisions),
            "episode_over": episode_over,
        }

        return obs, rewards, terminated, truncated, info

    def _generate_episode(self) -> tuple[WorldState, NDArray[np.float64], dict]:
        """Generate geometry, spawn agents, verify solvability.

        Returns (world, preferred_speeds, metadata).
        """
        cfg = self.config

        for _attempt in range(cfg.max_regeneration_attempts):
            # Pick tier
            if cfg.geometry_tiers is not None:
                tier = self._rng.choice(cfg.geometry_tiers)
                geom_config = GeometryConfig(
                    tier=tier,
                    min_side=cfg.geometry.min_side,
                    max_side=cfg.geometry.max_side,
                    corridor_width_range=cfg.geometry.corridor_width_range,
                    corridor_length_range=cfg.geometry.corridor_length_range,
                    bottleneck_aperture_range=cfg.geometry.bottleneck_aperture_range,
                    bottleneck_depth_range=cfg.geometry.bottleneck_depth_range,
                    branch_width_range=cfg.geometry.branch_width_range,
                    branch_length_range=cfg.geometry.branch_length_range,
                )
            else:
                geom_config = cfg.geometry

            geom = generate_geometry(self._rng, geom_config)

            # Build navmesh
            navmesh = build_navmesh(geom.polygon)
            wall_segments = extract_wall_segments(geom.polygon)

            # Spawn agents
            spawn_result = spawn_agents(
                self._rng,
                geom.spawn_regions,
                geom.goal_regions,
                cfg.spawn,
            )

            # Per-agent clearance radius: use the larger body half-dimension
            # (same convention as the observation builder's navmesh signals)
            agent_radii = np.maximum(spawn_result.shoulder_widths, spawn_result.chest_depths)

            # Verify solvability (A* + portal-width check per agent)
            solvable_mask = verify_solvability(
                navmesh,
                spawn_result.positions,
                spawn_result.goal_positions,
                agent_radii,
                cfg.solvability_mode,
                cfg.max_unsolvable_fraction,
            )

            if solvable_mask is None:
                # Regenerate
                continue

            # Filter to solvable agents
            if not np.all(solvable_mask):
                (
                    positions,
                    velocities,
                    torso_orientations,
                    head_orientations,
                    shoulder_widths,
                    chest_depths,
                    goal_positions,
                    preferred_speeds,
                ) = filter_by_solvability(
                    solvable_mask,
                    spawn_result.positions,
                    spawn_result.velocities,
                    spawn_result.torso_orientations,
                    spawn_result.head_orientations,
                    spawn_result.shoulder_widths,
                    spawn_result.chest_depths,
                    spawn_result.goal_positions,
                    spawn_result.preferred_speeds,
                )
            else:
                positions = spawn_result.positions
                velocities = spawn_result.velocities
                torso_orientations = spawn_result.torso_orientations
                head_orientations = spawn_result.head_orientations
                shoulder_widths = spawn_result.shoulder_widths
                chest_depths = spawn_result.chest_depths
                goal_positions = spawn_result.goal_positions
                preferred_speeds = spawn_result.preferred_speeds

            if len(positions) == 0:
                continue

            world = WorldState(
                positions=positions,
                velocities=velocities,
                torso_orientations=torso_orientations,
                head_orientations=head_orientations,
                shoulder_widths=shoulder_widths,
                chest_depths=chest_depths,
                goal_positions=goal_positions,
                walkable_polygon=geom.polygon,
                wall_segments=wall_segments,
                navmesh=navmesh,
            )
            world.validate()

            metadata = {
                "tier": geom.tier.name,
                "shape": geom.metadata.get("shape", "unknown"),
                **geom.metadata,
            }

            return world, preferred_speeds, metadata

        raise RuntimeError(
            f"Failed to generate a solvable episode after {cfg.max_regeneration_attempts} attempts"
        )

    def _build_all_observations(self) -> NDArray[np.float64]:
        """Build observations for all agents (zero for inactive)."""
        self._world.active_mask = self._active_mask
        return build_observations_batch(self._world, self.config.obs)
