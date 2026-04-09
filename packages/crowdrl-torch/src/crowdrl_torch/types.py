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
    prev_goal_distances: Tensor  # (E, N)
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
    obs_dim: int = 79  # 7 + 8*7 + 16
    use_navmesh: bool = False
    max_waypoints: int = 16
    waypoint_crossing_threshold: float = 0.5

    # Action
    max_speed: float = 1.5
    max_heading_change: float = 0.2617993877991494  # pi/12
    max_torso_change: float = 0.2617993877991494  # pi/12
    max_head_change: float = 1.0471975511965976  # pi/3
    head_limit: float = 1.5707963267948966  # pi/2

    # Physics
    dt: float = 0.01
    velocity_damping: float = 0.8
    contact_stiffness: float = 30000.0
    contact_damping: float = 500.0
    wall_strength: float = 400.0
    wall_range: float = 0.3
    max_speed_multiplier: float = 2.0

    # Reward
    goal_bonus: float = 10.0
    collision_penalty: float = -1.0
    timeout_penalty: float = -5.0
    goal_radius: float = 0.5
    progress_weight: float = 1.0
    wall_proximity_penalty: float = -0.1
    wall_proximity_threshold: float = 1.5
    agent_proximity_penalty: float = -0.005
    agent_proximity_threshold: float = 2.0
    action_rate_weight: float = 0.0
    existence_penalty: float = -0.01
    use_smoothness: bool = True
    jerk_penalty_weight: float = -0.000001
    angular_accel_penalty_weight: float = -0.0001
    speed_deviation_weight: float = -0.001

    # Episode
    max_steps: int = 5000

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
            max_speed=cfg.action.max_speed,
            max_heading_change=cfg.action.max_heading_change,
            max_torso_change=cfg.action.max_torso_change,
            max_head_change=cfg.action.max_head_change,
            head_limit=cfg.action.head_limit,
            dt=cfg.dt,
            velocity_damping=cfg.velocity_damping,
            contact_stiffness=cfg.contact_stiffness,
            contact_damping=cfg.contact_damping,
            max_speed_multiplier=cfg.max_speed_multiplier,
            goal_bonus=cfg.reward.goal_bonus,
            collision_penalty=cfg.reward.collision_penalty,
            timeout_penalty=cfg.reward.timeout_penalty,
            goal_radius=cfg.reward.goal_radius,
            progress_weight=cfg.reward.progress_weight,
            wall_proximity_penalty=cfg.reward.wall_proximity_penalty,
            wall_proximity_threshold=cfg.reward.wall_proximity_threshold,
            agent_proximity_penalty=cfg.reward.agent_proximity_penalty,
            agent_proximity_threshold=cfg.reward.agent_proximity_threshold,
            action_rate_weight=cfg.reward.action_rate_weight,
            existence_penalty=cfg.reward.existence_penalty,
            use_smoothness=cfg.reward.use_smoothness,
            jerk_penalty_weight=cfg.reward.jerk_penalty_weight,
            angular_accel_penalty_weight=cfg.reward.angular_accel_penalty_weight,
            speed_deviation_weight=cfg.reward.speed_deviation_weight,
            max_steps=cfg.max_steps,
            use_navmesh=cfg.obs.use_navmesh,
        )


def make_initial_state(
    n_envs: int = 1,
    max_agents: int = 64,
    max_segments: int = 128,
    max_waypoints: int = 16,
    device: torch.device | str = "cpu",
) -> TorchWorldState:
    """Create a zeroed-out TorchWorldState with the given sizes."""
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
        prev_goal_distances=torch.zeros((n_envs, max_agents), dtype=torch.float32, device=device),
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
    )
