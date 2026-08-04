"""PyTorch-compatible state and config types for GPU-accelerated environments.

TorchWorldState is a dataclass holding batched tensors with shape
(E, MAX_AGENTS, ...) where E is the number of parallel environments.
Inactive agents (padding or terminated) are masked via ``active_mask``.

EnvConfig holds static configuration scalars passed to all functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import Tensor

if TYPE_CHECKING:
    from crowdrl_env.crowd_env import CrowdEnvConfig


@dataclass
class TorchWorldState:
    """GPU-resident batched environment state.

    All tensors have shape (E, MAX_AGENTS, ...) or (E,) for scalars,
    where E is the number of parallel environments.
    """

    # Agent state
    positions: Tensor  # (E, N, 2)
    velocities: Tensor  # (E, N, 2)
    torso_orientations: Tensor  # (E, N)
    head_orientations: Tensor  # (E, N)
    shoulder_widths: Tensor  # (E, N)
    chest_depths: Tensor  # (E, N)
    masses: Tensor  # (E, N) — agent mass in kg
    goal_positions: Tensor  # (E, N, 2)
    preferred_speeds: Tensor  # (E, N)

    # Masks
    active_mask: Tensor  # (E, N) bool
    cumulative_terminated: Tensor  # (E, N) bool

    # Geometry — static within an episode
    wall_segments: Tensor  # (E, S, 2, 2)
    n_segments: Tensor  # (E,) int32

    # Reward temporal state
    prev_velocities: Tensor  # (E, N, 2)
    prev_nav_distances: Tensor  # (E, N) -- previous-step path-distance metric (progress reward)
    prev_accelerations: Tensor  # (E, N, 2)
    prev_headings: Tensor  # (E, N)
    prev_heading_changes: Tensor  # (E, N)

    # Previous actions (for action rate penalty)
    prev_actions: Tensor  # (E, N, 4)

    # Navmesh waypoints (static per episode, pre-computed at reset)
    waypoints: Tensor  # (E, N, MAX_WP, 2) world-frame XY
    n_waypoints: Tensor  # (E, N) int32 — actual count per agent
    waypoint_cursor: Tensor  # (E, N) int32 — current progress index
    waypoint_path_lengths: Tensor  # (E, N, MAX_WP) cumulative remaining distance to goal

    # Bookkeeping
    n_agents: Tensor  # (E,) int32
    step_count: Tensor  # (E,) int32

    # Stuck-agent tracking (rolling progress window).
    # ``stuck_window_step`` counts how many simulation steps have passed
    # since the last stuck-check for each agent; ``stuck_window_start_nav``
    # is that agent's path-distance metric (navmesh remaining path, or
    # straight-line goal distance as fallback) when the current window
    # started. Progress is measured along the route, not the bee-line, so
    # an agent following a detour is not falsely flagged as stuck. Used by
    # step.py when ``config.stuck_termination_enabled`` is True.
    stuck_window_step: Tensor  # (E, N) int32
    stuck_window_start_nav: Tensor  # (E, N) float32

    # Temporal memory state (per-agent trajectory history).
    # Populated at reset and updated every step. The obs builder reads these
    # when ``config.use_temporal_memory`` is True to emit the 6 temporal
    # features. See crowdrl_core.observation for the exact semantics.
    spawn_positions: Tensor  # (E, N, 2) float32 — position at episode start
    initial_goal_distances: Tensor  # (E, N) float32 — ||goal - spawn|| at start
    cumulative_path_length: Tensor  # (E, N) float32 — running path length
    pos_history: Tensor  # (E, N, W+1, 2) float32 — ring buffer of positions
    gdist_history: Tensor  # (E, N, W+1) float32 — ring buffer of goal distances

    # Neighbor memory state: persistent per-agent neighbor slot assignments.
    # Updated once per step by sensing.match_persistent_neighbors so that
    # neighbor identity stays stable across steps, which is the prerequisite
    # for per-neighbor temporal memory features (commits 3-5 of the
    # neighbor-memory plan). All -1 when neighbor memory is disabled.
    neighbor_ids: Tensor  # (E, N, K) int32 -- global agent index or -1

    # Ring buffer of global-frame neighbor velocities, one entry per step
    # for each of the K persistent slots. Zero-reset on slot reassignment
    # so that observation features never mix the old assignee's history
    # with the new one. Buffer size W_n+1 where W_n =
    # EnvConfig.neighbor_vel_history_window. Only written when
    # ``config.use_neighbor_memory`` is True.
    neighbor_vel_history: Tensor  # (E, N, W_n+1, K, 2) float32

    def clone(self) -> "TorchWorldState":
        """Return a copy with all tensors cloned (breaks CUDA graph aliasing)."""
        return TorchWorldState(
            **{f.name: getattr(self, f.name).clone() for f in self.__dataclass_fields__.values()}
        )


class EnvConfig(NamedTuple):
    """Static environment configuration.

    All fields are Python scalars — no tensors — so they can be used
    as compile-time constants by ``torch.compile``.
    """

    # Dimensions
    max_agents: int = 64
    max_segments: int = 128
    n_rays: int = 16
    fov_deg: float = 200.0
    max_range: float = 5.0
    k_neighbours: int = 8
    obs_dim: int = 80  # 8 + 8*7 + 16  (ego now includes preferred_speed)
    use_navmesh: bool = False
    use_goal_direction: bool = (
        True  # expose global-goal bearing in ego block; False = navmesh waypoint only
    )
    # Serve the deployed JuPedSim signal: stored waypoint path precomputed at
    # the router's fixed 0.2 m portal inset (episode factory) and the path_dev
    # obs channel pinned to 0.0 (compute_navmesh_signals). Mirrors
    # ObsConfig.use_jupedsim_style_routing -- see its docstring for the contract.
    use_jupedsim_style_routing: bool = False
    # Derived from ObsConfig.navmesh_max_waypoints in from_crowd_env_config. The
    # waypoints tensor is (E, N, max_waypoints, 2); at 64 envs x 100 agents the
    # default 1024 is ~50 MiB (+ ~25 MiB for path lengths). Far above any real 2D
    # pedestrian route -- the episode factory regenerates rather than truncate a
    # longer path, so the stored path's final waypoint is always the goal.
    max_waypoints: int = 1024
    waypoint_crossing_threshold: float = 0.5

    # Action (asymmetric: humans walk forward faster than backward).
    # Experimental defaults; to be backed by literature -- see ActionConfig.
    max_forward_speed: float = 2.0
    max_backward_speed: float = 0.5
    max_heading_change: float = 0.020  # ~115 deg/s at dt=0.01s (Layer 1; was pi/12)
    max_torso_change: float = 0.010  # ~57 deg/s at dt=0.01s (Layer 1; was pi/12)
    max_head_change: float = 0.030  # ~172 deg/s at dt=0.01s (Layer 1; was pi/3)
    head_limit: float = 1.5707963267948966  # pi/2

    # Speed-turn coupling (lateral-acceleration cap). When enabled, per-step
    # heading + torso change are clamped to omega_max(v) = min(turn_pivot_rate,
    # turn_lat_accel / v) * dt, so agents must slow down to turn sharply
    # ("slow before the turn"). Default off preserves the flat-cap behaviour.
    speed_turn_coupling: bool = False
    turn_lat_accel: float = 2.0  # m/s^2 comfortable centripetal accel
    turn_pivot_rate: float = 2.0943951023931953  # rad/s (~120 deg/s) v->0 cap

    # Physics
    dt: float = 0.01
    # Fraction of the policy's DESIRED velocity blended into the actual velocity
    # each step:  v <- w * v_desired + (1 - w) * v_prev.  Mind the (counter-
    # intuitive) direction: LOW w = strong low-pass / HIGH inertia (sluggish, and
    # the "ice-skating" mechanism if too low); w -> 1 = responsive, almost no
    # filter. At dt=0.01s, w=0.05 ~ tau 200 ms. Most configs pin this to 0.8.
    desired_velocity_weight: float = 0.05
    contact_stiffness: float = 30000.0
    contact_damping: float = 500.0
    wall_strength: float = 400.0
    wall_range: float = 0.3
    max_velocity_magnitude: float = 3.0
    """Hard clamp on actual velocity magnitude (m/s). Safety against
    contact-force-induced blowup, not a behavioural constraint -- sits
    above max_forward_speed so policy-commanded motion is never the
    binding constraint. Experimental starting point; the literature on
    transient running and emergency-evacuation pedestrian speeds should
    refine this."""

    # Reward
    goal_bonus: float = 10.0
    collision_penalty: float = -1.0
    timeout_penalty: float = -5.0
    goal_radius: float = 0.5
    progress_weight: float = 1.0
    wall_proximity_penalty: float = -0.1
    wall_proximity_threshold: float = 1.5
    wall_collision_penalty: float = -1.0  # per-step penalty for wall contact (0 = off)
    agent_proximity_penalty_near: float = -0.005
    agent_proximity_penalty_far: float = -0.0001
    personal_space_radius: float = 1.0
    action_rate_weight: float = -0.01  # Layer 1: was 0.0 (disabled)
    existence_penalty: float = -0.01
    # Gates the jerk + angular-accel penalties ONLY. It does NOT gate
    # speed_deviation_weight -- see that field below.
    use_smoothness: bool = True
    jerk_penalty_weight: float = (
        -1e-5
    )  # Layer 1 v2: was -1e-4 (10x down -- see reward.py docstring)
    angular_accel_penalty_weight: float = -0.01  # Layer 1: was -1e-4 (100x up)
    # Applies whenever non-zero, independent of use_smoothness (it needs no motion
    # history). It used to be nested under use_smoothness, silently disabling speed
    # matching for every use_smoothness=False config -- the baseline setting.
    # Reported on the ``smoothness`` component channel.
    speed_deviation_weight: float = (
        -0.005
    )  # Layer 1 v2: was -0.1 (20x down -- see reward.py docstring)

    # Velocity-weighted collision cost (P1). When enabled, the agent-collision
    # and wall-contact penalties are scaled by impact speed (closing speed for
    # agents, own speed for walls) instead of being binary. Default OFF ->
    # binary, identical to before. See crowdrl_env.reward.RewardConfig for the
    # full rationale; this must stay in lockstep with that dataclass.
    use_velocity_weighted_collision: bool = False
    collision_speed_floor: float = 0.5
    collision_speed_scale: float = 0.5
    collision_penalty_cap: float = 0.0  # per-step floor on the collision penalty (0 = off)

    # Velocity-weighted agent-proximity penalty (option 1). Scales the proximity
    # ramp by closing speed so coexisting / threading slowly is cheap and only
    # fast approach is taxed -- targets the freezing mode. Lockstep with
    # crowdrl_env.reward.RewardConfig.
    use_velocity_weighted_proximity: bool = False
    proximity_speed_floor: float = 0.25
    proximity_speed_scale: float = 0.5

    # Episode
    max_steps: int = 5000

    # Stuck-agent termination (see diagnose_stuck_agents.py). When enabled,
    # any active agent that fails to reduce its goal distance by at least
    # ``stuck_progress_threshold`` metres over a rolling window of
    # ``stuck_window_steps`` simulation steps is terminated as "failed"
    # (same reward as a timeout). The window restarts after every check.
    # Disabled by default so existing behaviour is unchanged.
    stuck_termination_enabled: bool = False
    stuck_window_steps: int = 300
    stuck_progress_threshold: float = 0.2

    # Temporal memory observation features. When enabled, 6 scalar features
    # derived from the agent's own trajectory history are appended to the
    # observation vector. See crowdrl_core.observation for details.
    use_temporal_memory: bool = False
    temporal_memory_window: int = 50

    # Neighbor memory: persistent per-agent neighbor ID table, the prerequisite
    # for per-neighbor temporal features (velocity history, trajectory feats).
    # When this flag is False, the matcher is not called and neighbor_ids
    # stays all -1 -- used to run a clean A-vs-A+ ablation on the same build.
    use_neighbor_memory: bool = False
    neighbor_sensing_radius: float = 5.0
    neighbor_vel_history_window: int = 5
    use_neighbor_vel_history: bool = False
    use_neighbor_trajectory_features: bool = False

    @staticmethod
    def from_crowd_env_config(
        cfg: CrowdEnvConfig,
        max_agents: int = 64,
        max_segments: int = 128,
    ) -> "EnvConfig":
        """Create an EnvConfig from a CrowdEnvConfig.

        Maps physics, action, observation, and reward parameters from the
        Gymnasium env config to the flat scalar config used by the GPU env.
        """
        return EnvConfig(
            max_agents=max_agents,
            max_segments=max_segments,
            n_rays=cfg.obs.raycast.n_rays,
            fov_deg=cfg.obs.raycast.fov_deg,
            max_range=cfg.obs.raycast.max_range,
            k_neighbours=cfg.obs.k_neighbours,
            obs_dim=cfg.obs.obs_dim,
            max_forward_speed=cfg.action.max_forward_speed,
            max_backward_speed=cfg.action.max_backward_speed,
            max_heading_change=cfg.action.max_heading_change,
            max_torso_change=cfg.action.max_torso_change,
            max_head_change=cfg.action.max_head_change,
            head_limit=cfg.action.head_limit,
            speed_turn_coupling=cfg.action.speed_turn_coupling,
            turn_lat_accel=cfg.action.turn_lat_accel,
            turn_pivot_rate=cfg.action.turn_pivot_rate,
            dt=cfg.dt,
            desired_velocity_weight=cfg.desired_velocity_weight,
            contact_stiffness=cfg.contact_stiffness,
            contact_damping=cfg.contact_damping,
            max_velocity_magnitude=cfg.max_velocity_magnitude,
            goal_bonus=cfg.reward.goal_bonus,
            collision_penalty=cfg.reward.collision_penalty,
            timeout_penalty=cfg.reward.timeout_penalty,
            goal_radius=cfg.reward.goal_radius,
            progress_weight=cfg.reward.progress_weight,
            wall_proximity_penalty=cfg.reward.wall_proximity_penalty,
            wall_proximity_threshold=cfg.reward.wall_proximity_threshold,
            wall_collision_penalty=cfg.reward.wall_collision_penalty,
            agent_proximity_penalty_near=cfg.reward.agent_proximity_penalty_near,
            agent_proximity_penalty_far=cfg.reward.agent_proximity_penalty_far,
            personal_space_radius=cfg.reward.personal_space_radius,
            action_rate_weight=cfg.reward.action_rate_weight,
            existence_penalty=cfg.reward.existence_penalty,
            use_smoothness=cfg.reward.use_smoothness,
            jerk_penalty_weight=cfg.reward.jerk_penalty_weight,
            angular_accel_penalty_weight=cfg.reward.angular_accel_penalty_weight,
            speed_deviation_weight=cfg.reward.speed_deviation_weight,
            use_velocity_weighted_collision=cfg.reward.use_velocity_weighted_collision,
            collision_speed_floor=cfg.reward.collision_speed_floor,
            collision_speed_scale=cfg.reward.collision_speed_scale,
            collision_penalty_cap=cfg.reward.collision_penalty_cap,
            use_velocity_weighted_proximity=cfg.reward.use_velocity_weighted_proximity,
            proximity_speed_floor=cfg.reward.proximity_speed_floor,
            proximity_speed_scale=cfg.reward.proximity_speed_scale,
            max_steps=cfg.max_steps,
            use_navmesh=cfg.obs.use_navmesh,
            use_goal_direction=cfg.obs.use_goal_direction,
            use_jupedsim_style_routing=cfg.obs.use_jupedsim_style_routing,
            max_waypoints=cfg.obs.navmesh_max_waypoints,
            stuck_termination_enabled=cfg.stuck_termination_enabled,
            stuck_window_steps=cfg.stuck_window_steps,
            stuck_progress_threshold=cfg.stuck_progress_threshold,
            use_temporal_memory=cfg.obs.use_temporal_memory,
            temporal_memory_window=cfg.obs.temporal_memory_window,
            use_neighbor_memory=cfg.obs.use_neighbor_memory,
            neighbor_sensing_radius=cfg.obs.neighbor_sensing_radius,
            neighbor_vel_history_window=cfg.obs.neighbor_vel_history_window,
            use_neighbor_vel_history=cfg.obs.use_neighbor_vel_history,
            use_neighbor_trajectory_features=cfg.obs.use_neighbor_trajectory_features,
        )


def make_initial_state(
    n_envs: int = 1,
    max_agents: int = 64,
    max_segments: int = 128,
    max_waypoints: int = 1024,
    memory_window: int = 50,
    k_neighbours: int = 8,
    neighbor_vel_history_window: int = 5,
    device: torch.device | str = "cpu",
) -> TorchWorldState:
    """Create a zeroed-out TorchWorldState with the given sizes.

    ``memory_window`` determines the ring-buffer size for the temporal-memory
    history (buffer has ``memory_window + 1`` slots). Passing a value that
    differs from the config at runtime will silently miscompute the memory
    features, so prefer to derive it from the same EnvConfig field.

    ``k_neighbours`` sizes the persistent neighbor-ID table. Prefer to pass
    the same value used by the observation builder so the table and the
    social obs channel stay aligned.

    ``neighbor_vel_history_window`` sizes the ring buffer that stores the
    recent velocities of each persistent neighbor slot. Buffer has
    ``neighbor_vel_history_window + 1`` slots. Must match the EnvConfig
    field of the same name.
    """
    buf_size = memory_window + 1
    nb_buf_size = neighbor_vel_history_window + 1
    return TorchWorldState(
        positions=torch.zeros((n_envs, max_agents, 2), dtype=torch.float32, device=device),
        velocities=torch.zeros((n_envs, max_agents, 2), dtype=torch.float32, device=device),
        torso_orientations=torch.zeros((n_envs, max_agents), dtype=torch.float32, device=device),
        head_orientations=torch.zeros((n_envs, max_agents), dtype=torch.float32, device=device),
        shoulder_widths=torch.zeros((n_envs, max_agents), dtype=torch.float32, device=device),
        chest_depths=torch.zeros((n_envs, max_agents), dtype=torch.float32, device=device),
        masses=torch.full((n_envs, max_agents), 80.0, dtype=torch.float32, device=device),
        goal_positions=torch.zeros((n_envs, max_agents, 2), dtype=torch.float32, device=device),
        preferred_speeds=torch.zeros((n_envs, max_agents), dtype=torch.float32, device=device),
        active_mask=torch.zeros((n_envs, max_agents), dtype=torch.bool, device=device),
        cumulative_terminated=torch.zeros((n_envs, max_agents), dtype=torch.bool, device=device),
        wall_segments=torch.zeros(
            (n_envs, max_segments, 2, 2), dtype=torch.float32, device=device
        ),
        n_segments=torch.zeros(n_envs, dtype=torch.int32, device=device),
        prev_velocities=torch.zeros((n_envs, max_agents, 2), dtype=torch.float32, device=device),
        prev_nav_distances=torch.zeros((n_envs, max_agents), dtype=torch.float32, device=device),
        prev_accelerations=torch.zeros(
            (n_envs, max_agents, 2), dtype=torch.float32, device=device
        ),
        prev_headings=torch.zeros((n_envs, max_agents), dtype=torch.float32, device=device),
        prev_heading_changes=torch.zeros((n_envs, max_agents), dtype=torch.float32, device=device),
        prev_actions=torch.zeros((n_envs, max_agents, 4), dtype=torch.float32, device=device),
        waypoints=torch.zeros(
            (n_envs, max_agents, max_waypoints, 2), dtype=torch.float32, device=device
        ),
        n_waypoints=torch.zeros((n_envs, max_agents), dtype=torch.int32, device=device),
        waypoint_cursor=torch.zeros((n_envs, max_agents), dtype=torch.int32, device=device),
        waypoint_path_lengths=torch.zeros(
            (n_envs, max_agents, max_waypoints), dtype=torch.float32, device=device
        ),
        n_agents=torch.zeros(n_envs, dtype=torch.int32, device=device),
        step_count=torch.zeros(n_envs, dtype=torch.int32, device=device),
        stuck_window_step=torch.zeros((n_envs, max_agents), dtype=torch.int32, device=device),
        stuck_window_start_nav=torch.zeros(
            (n_envs, max_agents), dtype=torch.float32, device=device
        ),
        spawn_positions=torch.zeros((n_envs, max_agents, 2), dtype=torch.float32, device=device),
        initial_goal_distances=torch.zeros(
            (n_envs, max_agents), dtype=torch.float32, device=device
        ),
        cumulative_path_length=torch.zeros(
            (n_envs, max_agents), dtype=torch.float32, device=device
        ),
        pos_history=torch.zeros(
            (n_envs, max_agents, buf_size, 2), dtype=torch.float32, device=device
        ),
        gdist_history=torch.zeros(
            (n_envs, max_agents, buf_size), dtype=torch.float32, device=device
        ),
        neighbor_ids=torch.full(
            (n_envs, max_agents, k_neighbours), -1, dtype=torch.int32, device=device
        ),
        neighbor_vel_history=torch.zeros(
            (n_envs, max_agents, nb_buf_size, k_neighbours, 2),
            dtype=torch.float32,
            device=device,
        ),
    )
