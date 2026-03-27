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

    # 1. Desired speed: map [-1, 1] -> [0, max_speed]
    desired_speeds = (actions[..., 0] + 1.0) / 2.0 * config.max_speed

    # 2. Heading change
    new_headings = current_headings + actions[..., 1] * config.max_heading_change

    # 3. Torso orientation change
    new_torsos = current_torsos + actions[..., 2] * config.max_torso_change

    # 4. Head orientation change relative to torso
    new_heads = current_heads + actions[..., 3] * config.max_head_change
    head_rel_torso = torch.clamp(
        new_heads - new_torsos, -config.head_limit, config.head_limit
    )
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
