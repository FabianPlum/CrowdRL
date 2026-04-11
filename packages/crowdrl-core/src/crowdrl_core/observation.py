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
  Temporal memory (optional, 6D):
    displacement_from_spawn_norm (1), cum_path_norm (1), path_efficiency (1),
    elapsed_fraction (1), disp_window_norm (1), goal_progress_window_norm (1)

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

_TEMPORAL_EPS = 1e-6
"""Small epsilon used to guard divisions when computing normalised temporal
features. Chosen small enough that it doesn't affect realistic scenarios
(initial goal distance is always > 0.5m for solvable episodes) but large
enough to avoid division-by-zero artefacts.
"""


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

    use_temporal_memory: bool = False
    """Whether to include 6 scalar temporal-memory features derived from the
    agent's own trajectory history. See ``crowdrl_core.observation`` docstring
    for the exact features. Adds 6D to the observation vector.
    """

    temporal_memory_window: int = 50
    """Length of the displacement/progress window (in simulation steps) used
    by the windowed temporal features. The ring buffer backing this feature
    has size ``temporal_memory_window + 1`` slots.
    """

    temporal_memory_max_steps: int = 2000
    """Max episode length used to normalise ``elapsed_fraction``. Must match
    the environment's ``max_steps`` — passing a mismatched value will give the
    policy a miscalibrated time-pressure signal. Only used when
    ``use_temporal_memory`` is True.
    """

    temporal_memory_dt: float = 0.01
    """Simulation timestep used to normalise window-based displacement and
    goal-progress features (``v_pref * W * dt`` is the expected distance moved
    in W steps at preferred speed). Must match the environment's ``dt``.
    Only used when ``use_temporal_memory`` is True.
    """

    use_neighbor_memory: bool = False
    """Whether to maintain a persistent neighbor-ID table for per-neighbor
    temporal memory features. On its own (commit 2 of the neighbor memory
    plan) this only turns on the matcher -- no observation dims are added.
    The dims come from ``use_neighbor_vel_history`` and
    ``use_neighbor_trajectory`` in later commits.
    """

    neighbor_sensing_radius: float = 5.0
    """Metres. A previously-tracked neighbor beyond this radius is evicted
    from its persistent slot; empty slots are only filled with agents
    within this radius. Defaults to the same 5m as the raycast max_range
    -- anything further than that is already invisible to the policy
    through the existing social channel, so tracking it here would be
    wasteful. Only used when ``use_neighbor_memory`` is True.
    """

    neighbor_vel_history_window: int = 5
    """Length of the neighbor-velocity ring buffer (in simulation steps).
    W_n=5 at dt=0.01 = 50ms, roughly one reaction timescale -- long
    enough to smooth out single-step noise, short enough to react to
    genuine acceleration changes. Buffer has W_n+1 slots. Only used
    when ``use_neighbor_memory`` is True.
    """

    use_neighbor_vel_history: bool = False
    """Whether to emit the (+K*2 = 16D) neighbor velocity-history feature
    block: for each of the K persistent slots, the difference between the
    neighbor's current velocity and its velocity W_n steps ago, rotated
    into the current ego frame. This is an acceleration proxy -- positive
    along the ego x-axis = neighbor accelerating forward relative to ego.

    Requires ``use_neighbor_memory=True``. With both flags set, obs_dim
    grows by K*2 (16 for K=8). Corresponds to the A+ ablation step in
    plan/neighbor_memory_extension.md section 2.2.
    """

    @property
    def obs_dim(self) -> int:
        """Total observation dimensionality."""
        ego = 7
        social = self.k_neighbours * 7
        rays = self.raycast.n_rays * (2 if self.raycast.two_channel else 1)
        nav = 3 if self.use_navmesh else 0
        memory = 6 if self.use_temporal_memory else 0
        nb_vel = (
            self.k_neighbours * 2
            if (self.use_neighbor_memory and self.use_neighbor_vel_history)
            else 0
        )
        return ego + social + rays + nav + memory + nb_vel


def _temporal_features(
    pos_now: NDArray[np.float64],
    gdist_now: NDArray[np.float64],
    spawn_pos: NDArray[np.float64],
    initial_gdist: NDArray[np.float64],
    cum_path: NDArray[np.float64],
    pos_window: NDArray[np.float64],
    gdist_window: NDArray[np.float64],
    step_count: int,
    max_steps: int,
    preferred_speeds: NDArray[np.float64],
    dt: float,
    window: int,
) -> NDArray[np.float64]:
    """Compute the 6 temporal-memory scalar features for one or many agents.

    Vectorised — accepts either (2,) / scalar inputs for one agent or
    (M, 2) / (M,) inputs for a batch of M agents. Returns array with
    shape (6,) or (M, 6) respectively.

    Parameters
    ----------
    pos_now : (M, 2) or (2,) — current position
    gdist_now : (M,) or scalar — current ||goal - pos||
    spawn_pos : (M, 2) or (2,) — position at episode start
    initial_gdist : (M,) or scalar — ||goal - spawn|| at episode start
    cum_path : (M,) or scalar — running cumulative path length
    pos_window : (M, 2) or (2,) — position W steps ago (or spawn if t<W)
    gdist_window : (M,) or scalar — goal distance W steps ago
    step_count : int — current episode step (same for all agents in one env)
    max_steps : int — episode length budget
    preferred_speeds : (M,) or scalar — per-agent preferred speed
    dt : float — simulation timestep
    window : int — W (steps)

    Returns
    -------
    features : (M, 6) or (6,) float64 — the six normalised features
    """
    # Shapes that work for both per-agent (2,) and batched (M, 2)
    is_batched = pos_now.ndim == 2

    # 1. Displacement from spawn, normalised by initial goal distance
    disp_spawn = np.linalg.norm(pos_now - spawn_pos, axis=-1)
    disp_spawn_norm = disp_spawn / np.maximum(initial_gdist, _TEMPORAL_EPS)

    # 2. Cumulative path length, normalised by initial goal distance
    cum_path_norm = cum_path / np.maximum(initial_gdist, _TEMPORAL_EPS)

    # 3. Path efficiency: displacement / cum_path. 1.0 = straight line, ~0 = looping
    path_eff = disp_spawn / np.maximum(cum_path, _TEMPORAL_EPS)
    path_eff = np.clip(path_eff, 0.0, 1.0)

    # 4. Elapsed fraction of max episode steps
    elapsed = float(step_count) / max(max_steps, 1)
    if is_batched:
        elapsed_arr = np.full(pos_now.shape[0], elapsed, dtype=np.float64)
    else:
        elapsed_arr = np.float64(elapsed)

    # 5. Displacement over the last W steps, normalised by the distance an
    #    agent moving at preferred speed would cover in W steps
    disp_window = np.linalg.norm(pos_now - pos_window, axis=-1)
    expected_window = np.maximum(preferred_speeds * window * dt, _TEMPORAL_EPS)
    disp_window_norm = disp_window / expected_window

    # 6. Goal progress over the last W steps (positive = approaching goal),
    #    normalised the same way as displacement_window
    goal_progress_window = gdist_window - gdist_now
    goal_progress_window_norm = goal_progress_window / expected_window

    if is_batched:
        return np.stack(
            [
                disp_spawn_norm,
                cum_path_norm,
                path_eff,
                elapsed_arr,
                disp_window_norm,
                goal_progress_window_norm,
            ],
            axis=-1,
        )
    return np.array(
        [
            disp_spawn_norm,
            cum_path_norm,
            path_eff,
            elapsed_arr,
            disp_window_norm,
            goal_progress_window_norm,
        ],
        dtype=np.float64,
    )


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

    # --- Temporal memory (optional, 6D) ---
    if config.use_temporal_memory:
        parts.append(_per_agent_temporal_features(world, agent_idx, config, ego_pos, goal_dist))

    # --- Neighbor velocity history (optional, K*2 D) ---
    if config.use_neighbor_memory and config.use_neighbor_vel_history:
        parts.append(_per_agent_neighbor_vel_history_features(world, agent_idx, config, rot))

    return np.concatenate(parts)


def _per_agent_temporal_features(
    world: WorldState,
    agent_idx: int,
    config: ObsConfig,
    ego_pos: NDArray[np.float64],
    ego_goal_dist: float,
) -> NDArray[np.float64]:
    """Compute the 6D temporal-memory feature block for a single agent.

    Reads the per-agent trajectory state from ``world`` (spawn position,
    cumulative path length, ring-buffer history, etc.). Returns a zero
    vector if the state arrays aren't populated yet — this keeps the
    obs builder robust when called on a freshly-built WorldState from
    a hand-written test or an adapter that hasn't wired up memory yet.
    """
    if (
        world.spawn_positions is None
        or world.initial_goal_distances is None
        or world.cumulative_path_length is None
        or world.pos_history is None
        or world.gdist_history is None
    ):
        return np.zeros(6, dtype=np.float64)

    W = config.temporal_memory_window
    buf_size = W + 1
    # Read oldest entry in the ring buffer. The writer stores pos_{t+1} at
    # index (t % buf_size) where t is the pre-step step_count, and after the
    # step ``world.step_count`` equals (t+1). The oldest valid entry is at
    # index ``world.step_count % buf_size`` (all slots initialised to
    # spawn_pos at reset, so early reads return the spawn position).
    read_idx = world.step_count % buf_size
    pos_window = world.pos_history[agent_idx, read_idx]
    gdist_window = world.gdist_history[agent_idx, read_idx]

    preferred_speed = (
        world.preferred_speeds[agent_idx] if world.preferred_speeds is not None else 1.3
    )

    return _temporal_features(
        pos_now=ego_pos,
        gdist_now=ego_goal_dist,
        spawn_pos=world.spawn_positions[agent_idx],
        initial_gdist=world.initial_goal_distances[agent_idx],
        cum_path=world.cumulative_path_length[agent_idx],
        pos_window=pos_window,
        gdist_window=gdist_window,
        step_count=world.step_count,
        max_steps=config.temporal_memory_max_steps or 1,
        preferred_speeds=preferred_speed,
        dt=config.temporal_memory_dt,
        window=W,
    )


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

        offset += 3

    # --- Temporal memory (batched, 6D) ---
    if config.use_temporal_memory and _memory_state_populated(world):
        W = config.temporal_memory_window
        buf_size = W + 1
        read_idx = world.step_count % buf_size

        pos_now_a = world.positions[active_idx]  # (M, 2)
        spawn_a = world.spawn_positions[active_idx]  # (M, 2)
        init_g_a = world.initial_goal_distances[active_idx]  # (M,)
        cum_a = world.cumulative_path_length[active_idx]  # (M,)
        pos_window_a = world.pos_history[active_idx, read_idx]  # (M, 2)
        gdist_window_a = world.gdist_history[active_idx, read_idx]  # (M,)
        gdist_now_a = np.linalg.norm(world.goal_positions[active_idx] - pos_now_a, axis=-1)  # (M,)

        if world.preferred_speeds is not None:
            pref_a = world.preferred_speeds[active_idx]
        else:
            pref_a = np.full(M, 1.3, dtype=np.float64)

        temporal = _temporal_features(
            pos_now=pos_now_a,
            gdist_now=gdist_now_a,
            spawn_pos=spawn_a,
            initial_gdist=init_g_a,
            cum_path=cum_a,
            pos_window=pos_window_a,
            gdist_window=gdist_window_a,
            step_count=world.step_count,
            max_steps=config.temporal_memory_max_steps or 1,
            preferred_speeds=pref_a,
            dt=config.temporal_memory_dt,
            window=W,
        )  # (M, 6)
        obs[active_idx, offset : offset + 6] = temporal
        offset += 6

    # --- Neighbor velocity history (batched, K*2 D) ---
    if (
        config.use_neighbor_memory
        and config.use_neighbor_vel_history
        and world.neighbor_ids is not None
        and world.neighbor_vel_history is not None
    ):
        K = config.k_neighbours
        W_n = config.neighbor_vel_history_window
        nb_buf = W_n + 1
        newest = (world.step_count - 1) % nb_buf
        oldest = world.step_count % nb_buf

        nb_ids_a = world.neighbor_ids[active_idx]  # (M, K)
        hist_a = world.neighbor_vel_history[active_idx]  # (M, W_n+1, K, 2)
        vel_now = hist_a[:, newest, :, :]  # (M, K, 2)
        vel_old = hist_a[:, oldest, :, :]  # (M, K, 2)
        diff_global = vel_now - vel_old  # (M, K, 2)

        # Rotate each diff into ego frame: (M, K, 2) by per-M rotation.
        diff_ex = (
            cos_h[:, np.newaxis] * diff_global[..., 0] - sin_h[:, np.newaxis] * diff_global[..., 1]
        )
        diff_ey = (
            sin_h[:, np.newaxis] * diff_global[..., 0] + cos_h[:, np.newaxis] * diff_global[..., 1]
        )
        diff_ego = np.stack([diff_ex, diff_ey], axis=-1)  # (M, K, 2)

        # Mask empty slots
        valid = nb_ids_a >= 0  # (M, K)
        diff_ego = np.where(valid[..., np.newaxis], diff_ego, 0.0)

        flat = diff_ego.reshape(M, K * 2)
        obs[active_idx, offset : offset + K * 2] = flat
        offset += K * 2

    return obs


def _memory_state_populated(world: WorldState) -> bool:
    """Return True only if all temporal-memory state arrays are present."""
    return (
        world.spawn_positions is not None
        and world.initial_goal_distances is not None
        and world.cumulative_path_length is not None
        and world.pos_history is not None
        and world.gdist_history is not None
    )


def _per_agent_neighbor_vel_history_features(
    world: WorldState,
    agent_idx: int,
    config: ObsConfig,
    rot: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the (K*2,) neighbor velocity-history feature block for one
    agent. Returns the difference ``vel_now - vel_W_ago`` for each of the
    K persistent slots, rotated into the current ego frame.

    Empty slots (neighbor_ids == -1) return zeros. Until the ring buffer
    fills (first W_n+1 steps), ``vel_W_ago`` is zero by reset-initialisation,
    so the feature is just the current neighbor velocity -- a cautious
    fallback that degrades gracefully.

    The flat layout is (K * 2,): [slot_0_dx, slot_0_dy, slot_1_dx, ...].

    Parameters
    ----------
    rot : (2, 2) -- precomputed global->ego rotation matrix for this agent.
    """
    if world.neighbor_ids is None or world.neighbor_vel_history is None:
        return np.zeros(config.k_neighbours * 2, dtype=np.float64)

    W_n = config.neighbor_vel_history_window
    buf = W_n + 1

    # Newest write index corresponds to the PRE-step step_count used by the
    # scatter write, i.e. (world.step_count - 1) post-step. Read oldest at
    # world.step_count % buf. Subtle: on a fresh reset where world.step_count
    # is 0, the newest slot is buf-1 (holds zero, never written), and the
    # oldest slot is 0 (also zero). Diff is zero -- correct.
    newest = (world.step_count - 1) % buf
    oldest = world.step_count % buf

    nb_ids = world.neighbor_ids[agent_idx]  # (K,)
    hist = world.neighbor_vel_history[agent_idx]  # (W_n+1, K, 2)
    vel_now = hist[newest]  # (K, 2)
    vel_old = hist[oldest]  # (K, 2)

    diff_global = vel_now - vel_old  # (K, 2)
    # Rotate each slot's diff into ego frame: (2, 2) @ (2,) per slot
    diff_ego = diff_global @ rot.T  # (K, 2)

    # Zero out empty slots
    diff_ego = np.where(nb_ids[:, np.newaxis] >= 0, diff_ego, 0.0)

    return diff_ego.flatten()
