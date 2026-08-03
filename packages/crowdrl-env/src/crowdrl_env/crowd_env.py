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

import logging
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
from crowdrl_core.navmesh import (
    JUPEDSIM_ROUTER_INSET,
    first_waypoint_headings,
    remaining_path_lengths,
)
from crowdrl_core.observation import ObsConfig, build_observations_batch
from crowdrl_core.world_state import WorldState

from crowdrl_env.geometry_generator import GeometryConfig, GeometryTier, generate_geometry
from crowdrl_env.reward import RewardConfig, RewardState, compute_rewards
from crowdrl_env.solvability import SolvabilityMode, filter_by_solvability, verify_solvability
from crowdrl_env.spawner import SpawnConfig, SpawnShortfallError, spawn_agents

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrowdEnvConfig:
    """Full configuration for the CrowdRL training environment."""

    # Geometry
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    geometry_tiers: list[GeometryTier] | None = None
    """If set, randomly pick from these tiers each episode (overrides geometry.tier)."""

    tier_weights: list[float] | None = None
    """Sampling weights for geometry_tiers. Same length as geometry_tiers.
    None = uniform sampling."""

    # Spawning
    spawn: SpawnConfig = field(default_factory=SpawnConfig)

    # Solvability
    solvability_mode: SolvabilityMode = SolvabilityMode.PRUNE
    max_unsolvable_fraction: float = 0.3
    max_regeneration_attempts: int = 10
    solvability_clearance_factor: float = 1.2
    """Safety margin multiplier for agent radius in solvability checks.
    1.2 = 20% margin, ensuring agents at their widest orientation can
    traverse the proposed path. Applied to both portal-width and
    geometric clearance checks."""

    # Observation
    obs: ObsConfig = field(default_factory=ObsConfig)

    # Action
    action: ActionConfig = field(default_factory=ActionConfig)

    # Reward
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Physics
    dt: float = 0.01
    """Timestep duration (seconds)."""
    contact_stiffness: float = 30000.0
    """Agent-agent spring constant (N per unit overlap). Calibrated against
    JuPedSim's Social Force Model (Helbing et al. 2000): body_force k =
    120,000 N/m. Since our overlap is normalised [0,1] and typical overlap
    of 0.1 corresponds to ~0.023m physical penetration, 30,000 N * 0.1 /
    80 kg ~ 37 m/s^2, matching JuPedSim's ~35 m/s^2 at equivalent overlap."""
    contact_damping: float = 500.0
    """Agent-agent velocity-dependent damping (N*s/m). Calibrated so that
    two agents closing at 2 m/s with moderate overlap experience ~12 m/s^2
    damping deceleration (comparable to JuPedSim's friction term)."""
    desired_velocity_weight: float = 0.05
    """Weight on desired velocity in v_new = w * v_desired + (1-w) * v_old.
    Higher value = less smoothing (more responsive to policy output);
    lower value = more inertia.

    Layer 1 of plan/agent_dynamics_refactor.md (2026-05-25) lowered the
    default from 0.8 (tau ~12 ms, effectively no filter at dt=0.01s) to
    0.05 (tau ~200 ms). Helbing's social force model uses tau ~500 ms;
    a value of 0.02 here would match that, kept as a future tunable.
    Historical configs in configs/exp_memory_*.yaml pin
    ``desired_velocity_weight: 0.8`` explicitly to preserve their
    pre-Layer-1 behaviour. Formerly named ``velocity_damping`` -- that
    name was misleading because the formula meant the opposite of what
    the word suggested."""

    max_velocity_magnitude: float = 3.0
    """Hard clamp on velocity magnitude (m/s).

    After contact forces are applied, agent speeds are clamped to this
    value. This prevents contact forces from launching agents at
    unrealistic velocities while still allowing brief bursts above
    the desired-speed ceiling (e.g. being pushed by a crowd).

    Sits above ``action.max_forward_speed`` so policy-commanded motion
    is never the binding constraint. Experimental starting point; the
    literature on transient running and emergency-evacuation
    pedestrian speeds should refine this value.
    """

    # Episode
    max_steps: int = 5000
    """Maximum timesteps per episode."""

    # Stuck-agent termination
    stuck_termination_enabled: bool = False
    """When True, terminate individual agents that fail to make progress
    toward their goal over a rolling window. Applies a timeout penalty
    (same as episode truncation) and removes the agent from the episode
    without waiting for the full max_steps budget."""

    stuck_window_steps: int = 300
    """Length of the rolling stuck-detection window, in simulation steps."""

    stuck_progress_threshold: float = 0.2
    """Minimum goal-distance reduction required within the stuck window
    (metres). Agents reducing their goal distance by less than this over
    ``stuck_window_steps`` consecutive steps are terminated as stuck.
    Only applied when ``stuck_termination_enabled`` is True."""


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

        # Stuck-agent tracking (rolling progress window). Mirrors the torch
        # implementation in crowdrl_torch.step; see CrowdEnvConfig for the
        # three knobs that control it.
        self._stuck_window_step: NDArray[np.int32] | None = None
        self._stuck_window_start_nav: NDArray[np.float64] | None = None

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

        # Initialise reward state. The progress reward and stuck check use the
        # path-distance metric (navmesh remaining path, straight-line fallback)
        # so progress is measured along the route, not the bee-line to the goal.
        # ``goal_distances`` (straight-line) is still used below for the temporal-
        # memory features, which are intentionally goal-relative.
        goal_distances = np.linalg.norm(world.goal_positions - world.positions, axis=1)
        nav_distances = self._path_distance_metric()
        self._reward_state.reset(self._n_agents, nav_distances)

        # Initialise stuck-agent tracking (start = initial path distance)
        self._stuck_window_step = np.zeros(self._n_agents, dtype=np.int32)
        self._stuck_window_start_nav = nav_distances.copy()

        # Initialise temporal-memory state on the WorldState so the obs builder
        # can read it. Ring buffers are pre-filled with the spawn position /
        # initial goal distance so early reads return the spawn value.
        if self.config.obs.use_temporal_memory:
            W = self.config.obs.temporal_memory_window
            buf_size = W + 1
            self._world.spawn_positions = self._world.positions.copy()
            self._world.initial_goal_distances = goal_distances.copy()
            self._world.cumulative_path_length = np.zeros(self._n_agents, dtype=np.float64)
            self._world.pos_history = np.broadcast_to(
                self._world.positions[:, np.newaxis, :], (self._n_agents, buf_size, 2)
            ).copy()
            self._world.gdist_history = np.broadcast_to(
                goal_distances[:, np.newaxis], (self._n_agents, buf_size)
            ).copy()
            self._world.preferred_speeds = self._preferred_speeds.copy()
            self._world.step_count = 0

        # Initialise persistent neighbor-ID table. Seed with an initial
        # match so the first observation sees populated slots. Mirrors the
        # torch path in BatchedTorchEnv.reset_all.
        if self.config.obs.use_neighbor_memory:
            from crowdrl_core.sensing import match_persistent_neighbors

            k = self.config.obs.k_neighbours
            prev = np.full((self._n_agents, k), -1, dtype=np.int32)
            self._world.neighbor_ids = match_persistent_neighbors(
                self._world.positions,
                prev,
                self._active_mask,
                sensing_radius=self.config.obs.neighbor_sensing_radius,
                k=k,
            )
            nb_buf = self.config.obs.neighbor_vel_history_window + 1
            self._world.neighbor_vel_history = np.zeros(
                (self._n_agents, nb_buf, k, 2), dtype=np.float64
            )

        # Build initial observations
        obs = self._build_all_observations()

        info = {
            "n_agents": self._n_agents,
            "geometry_tier": geom_metadata.get("tier"),
            "geometry_shape": geom_metadata.get("shape"),
            # Density provenance -- see _generate_episode. "n_agents" above is the
            # count actually simulated; "requested_n" is what was asked for. They
            # differ on a shortfall, and downstream reporting must not conflate them.
            "requested_n": geom_metadata.get("requested_n"),
            "spawn_area_m2": geom_metadata.get("spawn_area_m2"),
            "spawn_capacity": geom_metadata.get("spawn_capacity"),
            "walkable_area_m2": geom_metadata.get("walkable_area_m2"),
            "achieved_density": geom_metadata.get("achieved_density"),
        }

        return obs, info

    def _path_distance_metric(self) -> NDArray[np.float64]:
        """Per-agent distance metric for the progress reward and stuck check.

        Returns the remaining navmesh path length (straight-line goal-distance
        fallback per agent) so both signals measure progress ALONG THE ROUTE,
        not the straight-line bee-line to the goal. Falls back to straight-line
        goal distance entirely when navmesh signals are disabled. Mirrors the
        torch ``path_distance_metric``.
        """
        if not self.config.obs.use_navmesh or self._world.navmesh is None:
            return np.linalg.norm(self._world.goal_positions - self._world.positions, axis=1)
        radii = np.maximum(self._world.shoulder_widths, self._world.chest_depths)
        return remaining_path_lengths(
            self._world.navmesh,
            self._world.positions,
            self._world.goal_positions,
            radii,
        )

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

        # Snapshot pre-step position + active mask for temporal-memory path
        # length accumulation. We use the pre-step active mask so that an
        # agent's final motion step (the one in which it reaches the goal)
        # still contributes to its cumulative path length.
        prev_positions_for_memory = self._world.positions.copy()
        prev_active_for_memory = self._active_mask.copy()

        # --- 1. Interpret actions → desired velocities and orientations ---
        batch_result = interpret_actions_batch(
            actions,
            self._world.torso_orientations,
            self._world.torso_orientations,
            self._world.head_orientations,
            cfg.action,
            current_speeds=np.linalg.norm(self._world.velocities, axis=1),
        )

        # --- 2. Apply velocity update (damped blending) — vectorized ---
        mask = self._active_mask
        self._world.velocities[mask] = (
            cfg.desired_velocity_weight * batch_result.desired_velocities[mask]
            + (1.0 - cfg.desired_velocity_weight) * self._world.velocities[mask]
        )
        self._world.torso_orientations[mask] = batch_result.new_torso_orientations[mask]
        self._world.head_orientations[mask] = batch_result.new_head_orientations[mask]

        # Snapshot the policy's chosen (pre-contact) velocities so the reward's
        # optional impact-speed weighting measures the approach speed the policy
        # controls, not the post-contact bounce. Only materialised when enabled.
        pre_contact_velocities = (
            self._world.velocities.copy() if cfg.reward.use_velocity_weighted_collision else None
        )

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
        # Apply contact accelerations (implicit unit mass) as velocity impulse
        self._world.velocities[mask] += contact_forces[mask] * cfg.dt

        # Clamp velocity magnitudes to prevent contact-force blow-up
        max_vel = cfg.max_velocity_magnitude
        speeds = np.linalg.norm(self._world.velocities[mask], axis=1)
        too_fast = speeds > max_vel
        if np.any(too_fast):
            scale = np.where(too_fast, max_vel / np.maximum(speeds, 1e-10), 1.0)
            self._world.velocities[mask] *= scale[:, np.newaxis]

        # Position update
        self._world.positions[self._active_mask] += (
            self._world.velocities[self._active_mask] * cfg.dt
        )

        # Wall boundary enforcement (returns the hard wall-contact mask)
        wall_collision_mask = enforce_wall_boundaries(self._world)

        # --- 5. Compute rewards ---
        # Distances for proximity penalties (agent-agent pair distances are
        # computed inside compute_rewards so the graded ramp can use per-pair
        # contact distances).
        wall_distances = compute_min_wall_distances(self._world)
        agent_radii = np.maximum(self._world.shoulder_widths, self._world.chest_depths)

        # Path-distance metric (remaining navmesh path, straight-line fallback),
        # shared by the progress reward and the stuck check below so both measure
        # route progress rather than the bee-line to the goal.
        nav_distances = self._path_distance_metric()

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
            current_distances=nav_distances,
            wall_distances=wall_distances,
            wall_collision_mask=wall_collision_mask,
            agent_radii=agent_radii,
            actions=actions,
            collision_velocities=pre_contact_velocities,
        )

        # --- 6. Update active mask ---
        # Deactivate agents that reached their goal
        newly_done = reached_goal & self._active_mask
        self._active_mask[newly_done] = False

        # Zero out velocities for inactive agents
        self._world.velocities[~self._active_mask] = 0.0

        # --- 6b. Stuck-agent termination (rolling progress window) ---
        truncated = np.zeros(self._n_agents, dtype=np.bool_)
        if (
            cfg.stuck_termination_enabled
            and self._stuck_window_step is not None
            and self._stuck_window_start_nav is not None
        ):
            # Progress measured along the navmesh path (reusing nav_distances
            # from the reward computation above), not the straight-line goal
            # distance -- so an agent following a route that bends away from the
            # goal is not falsely flagged as stuck.
            inc_mask = self._active_mask.copy()
            self._stuck_window_step[inc_mask] += 1

            window_full = self._stuck_window_step >= cfg.stuck_window_steps
            progress = self._stuck_window_start_nav - nav_distances
            stuck_mask = window_full & inc_mask & (progress < cfg.stuck_progress_threshold)
            reset_mask = window_full & inc_mask & ~stuck_mask

            # Apply timeout penalty and mark truncated
            rewards[stuck_mask] += cfg.reward.timeout_penalty
            truncated[stuck_mask] = True
            self._active_mask[stuck_mask] = False
            self._world.velocities[stuck_mask] = 0.0

            # Reset window for non-stuck window-full agents
            self._stuck_window_step[window_full & inc_mask] = 0
            self._stuck_window_start_nav[reset_mask] = nav_distances[reset_mask]

        # --- 7. Termination / truncation ---
        terminated = reached_goal.copy()

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

        # --- 7b. Update temporal-memory state ---
        if self.config.obs.use_temporal_memory and self._world.pos_history is not None:
            W = self.config.obs.temporal_memory_window
            buf_size = W + 1

            # Cumulative path length: add per-step delta for agents that
            # were active coming into this step.
            deltas = np.linalg.norm(self._world.positions - prev_positions_for_memory, axis=1)
            deltas = np.where(prev_active_for_memory, deltas, 0.0)
            self._world.cumulative_path_length = self._world.cumulative_path_length + deltas

            # Scatter-write into the ring buffer at index (pre-step step_count % buf_size).
            write_idx = (self._step_count - 1) % buf_size
            self._world.pos_history[:, write_idx, :] = self._world.positions
            new_goal_dists = np.linalg.norm(
                self._world.goal_positions - self._world.positions, axis=1
            )
            self._world.gdist_history[:, write_idx] = new_goal_dists

            self._world.step_count = self._step_count

        # --- 7c. Update persistent neighbor slots + velocity history ---
        if self.config.obs.use_neighbor_memory and self._world.neighbor_ids is not None:
            from crowdrl_core.sensing import match_persistent_neighbors

            prev_nids = self._world.neighbor_ids
            new_nids = match_persistent_neighbors(
                self._world.positions,
                prev_nids,
                self._active_mask,
                sensing_radius=self.config.obs.neighbor_sensing_radius,
                k=self.config.obs.k_neighbours,
            )

            # Zero-reset slot history on reassignment, then scatter-write
            # current velocities into the ring buffer.
            nb_buf = self.config.obs.neighbor_vel_history_window + 1
            hist = self._world.neighbor_vel_history
            slot_changed = new_nids != prev_nids  # (n_agents, K)
            if slot_changed.any():
                # Broadcast (n_agents, K) mask to the full (n_agents, buf, K, 2)
                # shape of hist. Boolean indexing won't work directly because
                # numpy tries to match the mask axes against the leading dims
                # of hist (which include the buf axis before K).
                preserve = ~slot_changed[:, np.newaxis, :, np.newaxis]
                hist = np.where(preserve, hist, 0.0)

            # Gather velocities for the assigned neighbors; zero for -1.
            ids_safe = np.clip(new_nids, 0, self._n_agents - 1)
            nb_vels = self._world.velocities[ids_safe]  # (n_agents, K, 2)
            nb_vels = np.where((new_nids >= 0)[:, :, np.newaxis], nb_vels, 0.0)

            write_idx = (self._step_count - 1) % nb_buf
            hist[:, write_idx, :, :] = nb_vels

            self._world.neighbor_ids = new_nids
            self._world.neighbor_vel_history = hist

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

        for attempt in range(cfg.max_regeneration_attempts):
            is_last_attempt = attempt == cfg.max_regeneration_attempts - 1
            # Pick tier
            if cfg.geometry_tiers is not None:
                weights = cfg.tier_weights
                if weights is not None:
                    p = np.array(weights, dtype=np.float64)
                    p = p / p.sum()
                else:
                    p = None
                tier = self._rng.choice(cfg.geometry_tiers, p=p)
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
                    min_passage_width=cfg.geometry.min_passage_width,
                )
            else:
                geom_config = cfg.geometry

            geom = generate_geometry(self._rng, geom_config)

            # Build navmesh
            navmesh = build_navmesh(geom.polygon)
            wall_segments = extract_wall_segments(geom.polygon)

            # Spawn agents
            # walkable is what lets the spawner (a) keep every body a full radius
            # clear of the walls and (b) dilate the entry zone within the geometry
            # when it is too small to hold the requested crowd.
            spawn_result = spawn_agents(
                self._rng,
                geom.spawn_regions,
                geom.goal_regions,
                cfg.spawn,
                walkable=geom.polygon,
            )

            # Bail early on a spawn shortfall: no point paying for solvability
            # verification on a crowd that is already the wrong size. Mirrors the same
            # check in crowdrl-torch's episode factory, so both engines regenerate on
            # the same episodes for a given seed.
            if (
                spawn_result.is_short
                and not is_last_attempt
                and cfg.spawn.spawn_shortfall_policy == "regenerate"
            ):
                continue

            # Per-agent clearance radius: use the larger body half-dimension
            # (same convention as the observation builder's navmesh signals)
            agent_radii = np.maximum(spawn_result.shoulder_widths, spawn_result.chest_depths)

            # Verify solvability (A* + portal-width + geometric clearance)
            solvable_mask = verify_solvability(
                navmesh,
                spawn_result.positions,
                spawn_result.goal_positions,
                agent_radii,
                cfg.solvability_mode,
                cfg.max_unsolvable_fraction,
                cfg.solvability_clearance_factor,
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
                    masses,
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
                    spawn_result.masses,
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
                masses = spawn_result.masses
                goal_positions = spawn_result.goal_positions
                preferred_speeds = spawn_result.preferred_speeds

            if len(positions) == 0:
                continue

            # Final delivered count, after BOTH drop paths (spawn placement and
            # solvability pruning). Either can silently shrink the crowd, so the
            # shortfall policy is applied to the number that actually gets simulated.
            requested_n = spawn_result.requested_n
            delivered_n = len(positions)
            if delivered_n < requested_n:
                policy = cfg.spawn.spawn_shortfall_policy
                if policy == "raise":
                    raise SpawnShortfallError(
                        requested_n,
                        delivered_n,
                        spawn_result.spawn_area_m2,
                        spawn_result.capacity,
                    )
                if not is_last_attempt and policy == "regenerate":
                    continue
                logger.warning(
                    "agent-count shortfall after %d attempt(s): %s after_solvability=%d tier=%s",
                    attempt + 1,
                    spawn_result.shortfall_summary,
                    delivered_n,
                    geom.tier.name,
                )

            # Orient agents toward their first navmesh waypoint rather than the
            # global goal (which may sit behind a wall relative to that waypoint).
            # Falls back to the goal bearing per-agent when no path exists. When
            # the navmesh is disabled we keep the spawner's global-goal bearing.
            if cfg.obs.use_navmesh:
                # Under the jupedsim-style contract, face the router-style
                # waypoint (fixed 0.2 m inset) the obs builder will serve.
                radii = (
                    np.full(len(positions), JUPEDSIM_ROUTER_INSET)
                    if cfg.obs.use_jupedsim_style_routing
                    else np.maximum(shoulder_widths, chest_depths)
                )
                torso_orientations = first_waypoint_headings(
                    navmesh, positions, goal_positions, radii
                )
                head_orientations = torso_orientations.copy()

            world = WorldState(
                positions=positions,
                velocities=velocities,
                torso_orientations=torso_orientations,
                head_orientations=head_orientations,
                shoulder_widths=shoulder_widths,
                chest_depths=chest_depths,
                masses=masses,
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
                # Density provenance LAST: every episode carries what was asked for,
                # what was delivered, and the area it was delivered into, so no
                # downstream figure can silently mislabel its own density. Placed after
                # the geom.metadata spread so a generator key can never shadow it.
                "requested_n": requested_n,
                "placed_n": delivered_n,
                "spawn_area_m2": spawn_result.spawn_area_m2,
                "spawn_capacity": spawn_result.capacity,
                "walkable_area_m2": float(geom.polygon.area),
                "achieved_density": delivered_n / float(geom.polygon.area),
            }

            return world, preferred_speeds, metadata

        raise RuntimeError(
            f"Failed to generate a solvable episode after {cfg.max_regeneration_attempts} attempts"
        )

    def _build_all_observations(self) -> NDArray[np.float64]:
        """Build observations for all agents (zero for inactive)."""
        self._world.active_mask = self._active_mask
        return build_observations_batch(self._world, self.config.obs)
