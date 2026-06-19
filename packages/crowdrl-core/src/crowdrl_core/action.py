"""Action interpreter: maps 4D policy output to kinematic quantities.

Action space (4D continuous):
  0. Desired speed (scalar, mapped from [-1,1] -> [-max_backward_speed,
     +max_forward_speed]; negative values mean motion opposite to heading)
  1. Desired heading change (scalar, mapped from [-1,1] -> [-max_turn, max_turn])
  2. Desired torso orientation change (scalar, same range as heading)
  3. Desired head orientation change relative to torso (scalar, clamped +-90 deg)

The head and torso are independently actuated:
- Head can rotate up to +-90 deg relative to torso (cheap information-gathering)
- Torso rotation alters the collision ellipse orientation (physical reorientation)
- Raycasts follow the head, not the torso

Desired-speed range is asymmetric: humans walk forward much faster than
backward. The signed value enables backing up (useful for queuing and
tight reversals) at a biomechanically realistic ratio. Current limits
are experimental starting points and should be backed by literature.

During training, outputs feed back into the physics step.
During deployment, the desired velocity feeds into JuPedSim's simulation loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ActionConfig:
    """Configuration for the action interpreter."""

    max_forward_speed: float = 2.0
    """Maximum desired forward speed (m/s); action[0] = +1 maps here.

    Experimental starting point. 2.0 m/s sits at the upper end of
    comfortable walking / lower end of slow running per Bohannon
    (1997) and the walking-to-running gait transition literature.
    The matching spawn-time preferred_speed clip in SpawnConfig is
    also 2.0 m/s, so every agent's preferred speed is reachable by
    the action interpreter. To be confirmed and refined via
    literature review of pedestrian-dynamics evacuation data.
    """

    max_backward_speed: float = 0.5
    """Maximum desired backward speed magnitude (m/s); action[0] = -1
    maps to -max_backward_speed (i.e. motion opposite to heading).

    Asymmetric with max_forward_speed because humans can walk backward
    at only a fraction of their forward speed (roughly 30-50% in normal
    locomotion). The 0.5 m/s starting point is biomechanically plausible
    but experimental; literature on reverse-gait pedestrian speeds is
    sparse and should be filled in as the model matures.
    """

    max_heading_change: float = 0.020
    """Max heading change per step (radians). 0.020 rad/step = 1.15 deg/step,
    115 deg/s at dt=0.01s -- mid-range of human walking yaw (Hicheur 2007:
    30-60 deg/s comfortable, ~120 deg/s aggressive). Layer 1 of
    plan/agent_dynamics_refactor.md (2026-05-25); was pi/12 = 1500 deg/s."""

    max_torso_change: float = 0.010
    """Max torso orientation change per step (radians). 0.010 rad/step =
    0.57 deg/step, 57 deg/s at dt=0.01s. Slower than max_heading_change
    because hip constraints cap sustained torso rotation around 60-90 deg/s.
    Layer 1 of plan/agent_dynamics_refactor.md (2026-05-25); was pi/12."""

    max_head_change: float = 0.030
    """Max head orientation change per step (radians). 0.030 rad/step =
    1.72 deg/step, 172 deg/s at dt=0.01s. Fastest axis (head scans faster
    than body commits). Layer 1 of plan/agent_dynamics_refactor.md
    (2026-05-25); was pi/3 = 6000 deg/s."""

    head_limit: float = np.pi / 2
    """Maximum head angle relative to torso (±90°)."""

    action_dim: int = 4
    """Dimensionality of the action space. 2=speed+heading, 3=+torso, 4=+head."""

    # --- Speed-turn coupling (lateral-acceleration cap) ---
    dt: float = 0.01
    """Timestep (s); converts the speed-coupled yaw envelope (rad/s) to a
    per-step cap. Should match the env dt."""

    speed_turn_coupling: bool = False
    """When True, per-step heading and torso change are clamped to a
    speed-dependent envelope so agents must slow to turn sharply (kills
    'ice-skating'). False preserves the flat-cap behaviour."""

    turn_lat_accel: float = 2.0
    """Comfortable centripetal acceleration (m/s^2): a_lat = v * omega is
    bounded by this. ~1.5-2.5 m/s^2 for human walking turns."""

    turn_pivot_rate: float = 2.0943951023931953
    """Max in-place yaw rate (rad/s, ~120 deg/s) at v->0; caps the
    envelope at low speed so standing pivots stay finite."""


@dataclass
class ActionResult:
    """Interpreted action output."""

    desired_velocity: NDArray[np.float64]
    """(2,) — desired velocity vector [vx, vy]."""

    new_heading: float
    """New heading angle (radians)."""

    new_torso_orientation: float
    """New torso orientation (radians)."""

    new_head_orientation: float
    """New absolute head orientation (radians)."""


def interpret_action(
    raw_action: NDArray[np.float64],
    current_heading: float,
    current_torso: float,
    current_head: float,
    config: ActionConfig = ActionConfig(),
    current_speed: float | None = None,
) -> ActionResult:
    """Interpret a raw policy action (values in [-1, 1]) into kinematic quantities.

    Parameters
    ----------
    raw_action : (action_dim,) array
        Raw policy output, each component in [-1, 1] (tanh output).
    current_heading : float
        Current heading angle (radians).
    current_torso : float
        Current torso orientation (radians).
    current_head : float
        Current absolute head orientation (radians).
    config : ActionConfig

    Returns
    -------
    ActionResult
    """
    # Clamp raw action to [-1, 1]
    action = np.clip(raw_action, -1.0, 1.0)

    # 1. Desired speed: linear remap [-1, 1] -> [-max_backward_speed, +max_forward_speed].
    # Negative values mean motion opposite to heading (i.e. backing up).
    speed_range = config.max_forward_speed + config.max_backward_speed
    desired_speed = -config.max_backward_speed + (action[0] + 1.0) / 2.0 * speed_range

    # 2. Heading + torso change, optionally clamped by the speed-turn
    #    coupling envelope so agents must slow down to turn sharply.
    heading_change = action[1] * config.max_heading_change
    has_torso = config.action_dim >= 3 and len(action) >= 3
    torso_change = action[2] * config.max_torso_change if has_torso else None
    if config.speed_turn_coupling and current_speed is not None:
        max_delta = float(_max_turn_per_step(np.asarray(current_speed), config))
        heading_change = float(np.clip(heading_change, -max_delta, max_delta))
        if torso_change is not None:
            torso_change = float(np.clip(torso_change, -max_delta, max_delta))

    new_heading = current_heading + heading_change
    new_torso = current_torso + torso_change if torso_change is not None else new_heading

    # 4. Head orientation change relative to torso (if action_dim >= 4)
    if config.action_dim >= 4 and len(action) >= 4:
        head_change = action[3] * config.max_head_change
        new_head = current_head + head_change
        # Enforce ±90° constraint relative to torso
        head_rel_torso = new_head - new_torso
        head_rel_torso = np.clip(head_rel_torso, -config.head_limit, config.head_limit)
        new_head = new_torso + head_rel_torso
    else:
        # Fuse head with torso
        new_head = new_torso

    # Normalise all angles to [-π, π]
    new_heading = float((new_heading + np.pi) % (2 * np.pi) - np.pi)
    new_torso = float((new_torso + np.pi) % (2 * np.pi) - np.pi)
    new_head = float((new_head + np.pi) % (2 * np.pi) - np.pi)

    # Desired velocity vector from heading and speed
    desired_velocity = np.array(
        [desired_speed * np.cos(new_heading), desired_speed * np.sin(new_heading)],
        dtype=np.float64,
    )

    return ActionResult(
        desired_velocity=desired_velocity,
        new_heading=new_heading,
        new_torso_orientation=new_torso,
        new_head_orientation=new_head,
    )


@dataclass
class BatchActionResult:
    """Vectorized action output for all agents."""

    desired_velocities: NDArray[np.float64]
    """(N, 2) — desired velocity vectors."""

    new_headings: NDArray[np.float64]
    """(N,) — new heading angles."""

    new_torso_orientations: NDArray[np.float64]
    """(N,) — new torso orientations."""

    new_head_orientations: NDArray[np.float64]
    """(N,) — new absolute head orientations."""


def _normalize_angles(angles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize angles to [-pi, pi]."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


def _max_turn_per_step(speeds: NDArray[np.float64], config: ActionConfig) -> NDArray[np.float64]:
    """Speed-coupled per-step yaw cap (radians).

    omega_max(v) = min(turn_pivot_rate, turn_lat_accel / v); the per-step
    cap is omega_max * dt. Bounds centripetal accel a_lat = v * omega, so
    turning sharply requires slowing down ("slow before the turn").
    """
    v = np.maximum(np.abs(speeds), 1e-3)
    omega_max = np.minimum(config.turn_pivot_rate, config.turn_lat_accel / v)
    return omega_max * config.dt


def interpret_actions_batch(
    raw_actions: NDArray[np.float64],
    current_headings: NDArray[np.float64],
    current_torsos: NDArray[np.float64],
    current_heads: NDArray[np.float64],
    config: ActionConfig = ActionConfig(),
    current_speeds: NDArray[np.float64] | None = None,
) -> BatchActionResult:
    """Interpret actions for a batch of agents (fully vectorized).

    Parameters
    ----------
    raw_actions : (n_agents, action_dim) array
    current_headings : (n_agents,) array
    current_torsos : (n_agents,) array
    current_heads : (n_agents,) array
    config : ActionConfig

    Returns
    -------
    BatchActionResult with (N, ...) arrays
    """
    actions = np.clip(raw_actions, -1.0, 1.0)

    # 1. Desired speed: linear remap [-1, 1] -> [-max_backward_speed, +max_forward_speed].
    speed_range = config.max_forward_speed + config.max_backward_speed
    desired_speeds = -config.max_backward_speed + (actions[:, 0] + 1.0) / 2.0 * speed_range

    # 2. Heading + torso change, optionally clamped by the speed-turn
    #    coupling envelope so agents must slow down to turn sharply.
    heading_delta = actions[:, 1] * config.max_heading_change
    has_torso = config.action_dim >= 3 and actions.shape[1] >= 3
    torso_delta = actions[:, 2] * config.max_torso_change if has_torso else None
    if config.speed_turn_coupling and current_speeds is not None:
        max_delta = _max_turn_per_step(current_speeds, config)
        heading_delta = np.clip(heading_delta, -max_delta, max_delta)
        if torso_delta is not None:
            torso_delta = np.clip(torso_delta, -max_delta, max_delta)

    new_headings = current_headings + heading_delta
    new_torsos = current_torsos + torso_delta if torso_delta is not None else new_headings.copy()

    # 4. Head orientation change relative to torso
    if config.action_dim >= 4 and actions.shape[1] >= 4:
        new_heads = current_heads + actions[:, 3] * config.max_head_change
        head_rel_torso = np.clip(new_heads - new_torsos, -config.head_limit, config.head_limit)
        new_heads = new_torsos + head_rel_torso
    else:
        new_heads = new_torsos.copy()

    # Normalize all angles to [-pi, pi]
    new_headings = _normalize_angles(new_headings)
    new_torsos = _normalize_angles(new_torsos)
    new_heads = _normalize_angles(new_heads)

    # Desired velocity vectors from heading and speed
    desired_velocities = np.column_stack(
        [
            desired_speeds * np.cos(new_headings),
            desired_speeds * np.sin(new_headings),
        ]
    )

    return BatchActionResult(
        desired_velocities=desired_velocities,
        new_headings=new_headings,
        new_torso_orientations=new_torsos,
        new_head_orientations=new_heads,
    )
