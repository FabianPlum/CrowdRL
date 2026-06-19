"""Action interpretation in PyTorch — maps 4D policy output to kinematic quantities.

Port of ``crowdrl_core.action.interpret_actions_batch``.
All operations are pure PyTorch tensor math, no side effects.

Shapes carry a leading (E,) environment batch dimension throughout.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from crowdrl_torch.types import EnvConfig


def _normalize_angles(angles: Tensor) -> Tensor:
    """Normalize angles to [-pi, pi]."""
    return (angles + math.pi) % (2 * math.pi) - math.pi


def interpret_actions(
    raw_actions: Tensor,
    current_headings: Tensor,
    current_torsos: Tensor,
    current_heads: Tensor,
    config: EnvConfig,
    current_speeds: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Interpret raw actions for all agents (vectorized).

    Parameters
    ----------
    raw_actions : (E, N, 4)
    current_headings : (E, N)
    current_torsos : (E, N)
    current_heads : (E, N)
    config : EnvConfig

    Returns
    -------
    desired_velocities : (E, N, 2)
    new_headings : (E, N)
    new_torsos : (E, N)
    new_heads : (E, N)
    """
    actions = torch.clamp(raw_actions, -1.0, 1.0)

    # 1. Desired speed: linear remap [-1, 1] -> [-max_backward_speed, +max_forward_speed].
    # Negative values mean motion opposite to heading (backing up).
    speed_range = config.max_forward_speed + config.max_backward_speed
    desired_speeds = -config.max_backward_speed + (actions[..., 0] + 1.0) / 2.0 * speed_range

    # 2. Heading + torso change, optionally clamped by the speed-turn
    #    coupling envelope so agents must slow down to turn sharply.
    heading_delta = actions[..., 1] * config.max_heading_change
    torso_delta = actions[..., 2] * config.max_torso_change
    if config.speed_turn_coupling and current_speeds is not None:
        v = current_speeds.abs().clamp(min=1e-3)
        omega_max = (config.turn_lat_accel / v).clamp(max=config.turn_pivot_rate)
        max_delta = omega_max * config.dt
        heading_delta = torch.minimum(torch.maximum(heading_delta, -max_delta), max_delta)
        torso_delta = torch.minimum(torch.maximum(torso_delta, -max_delta), max_delta)
    new_headings = current_headings + heading_delta
    new_torsos = current_torsos + torso_delta

    # 4. Head orientation change relative to torso
    new_heads = current_heads + actions[..., 3] * config.max_head_change
    head_rel_torso = torch.clamp(new_heads - new_torsos, -config.head_limit, config.head_limit)
    new_heads = new_torsos + head_rel_torso

    # Normalize all angles to [-pi, pi]
    new_headings = _normalize_angles(new_headings)
    new_torsos = _normalize_angles(new_torsos)
    new_heads = _normalize_angles(new_heads)

    # Desired velocity vectors
    desired_velocities = torch.stack(
        [desired_speeds * torch.cos(new_headings), desired_speeds * torch.sin(new_headings)],
        dim=-1,
    )

    return desired_velocities, new_headings, new_torsos, new_heads
