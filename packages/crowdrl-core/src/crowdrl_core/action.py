"""Action interpreter: maps 4D policy output to kinematic quantities.

Action space (4D continuous):
  0. Desired speed (scalar, mapped from [-1,1] → [0, max_speed])
  1. Desired heading change (scalar, mapped from [-1,1] → [-max_turn, max_turn])
  2. Desired torso orientation change (scalar, same range as heading)
  3. Desired head orientation change relative to torso (scalar, clamped ±90°)

The head and torso are independently actuated:
- Head can rotate up to ±90° relative to torso (cheap information-gathering)
- Torso rotation alters the collision ellipse orientation (physical reorientation)
- Raycasts follow the head, not the torso

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

    max_speed: float = 1.5
    """Maximum desired speed (m/s). Typical preferred pedestrian speed ~1.34 m/s."""

    max_heading_change: float = np.pi / 12
    """Maximum heading change per step (radians). π/12 = 15° per step."""

    max_torso_change: float = np.pi / 12
    """Maximum torso orientation change per step (radians). π/12 = 15° per step."""

    max_head_change: float = np.pi / 3
    """Maximum head orientation change per step (radians). π/3 = 60° per step."""

    head_limit: float = np.pi / 2
    """Maximum head angle relative to torso (±90°)."""

    action_dim: int = 4
    """Dimensionality of the action space. 2=speed+heading, 3=+torso, 4=+head."""


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

    # 1. Desired speed: map [−1, 1] → [0, max_speed]
    desired_speed = (action[0] + 1.0) / 2.0 * config.max_speed

    # 2. Heading change
    heading_change = action[1] * config.max_heading_change
    new_heading = current_heading + heading_change

    # 3. Torso orientation change (if action_dim >= 3)
    if config.action_dim >= 3 and len(action) >= 3:
        torso_change = action[2] * config.max_torso_change
        new_torso = current_torso + torso_change
    else:
        # Fuse torso with heading
        new_torso = new_heading

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


def interpret_actions_batch(
    raw_actions: NDArray[np.float64],
    current_headings: NDArray[np.float64],
    current_torsos: NDArray[np.float64],
    current_heads: NDArray[np.float64],
    config: ActionConfig = ActionConfig(),
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

    # 1. Desired speed: map [-1, 1] -> [0, max_speed]
    desired_speeds = (actions[:, 0] + 1.0) / 2.0 * config.max_speed

    # 2. Heading change
    new_headings = current_headings + actions[:, 1] * config.max_heading_change

    # 3. Torso orientation change
    if config.action_dim >= 3 and actions.shape[1] >= 3:
        new_torsos = current_torsos + actions[:, 2] * config.max_torso_change
    else:
        new_torsos = new_headings.copy()

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
