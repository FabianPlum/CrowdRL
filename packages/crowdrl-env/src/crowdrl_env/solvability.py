"""Solvability verifier: ensures all (spawn, goal) pairs have valid paths.

Three modes:
- Prune: remove unsolvable agents, keep geometry
- Regenerate: discard geometry if too many agents are unsolvable
- Strict: all agents must be solvable (validation runs)

Uses crowdrl-core's navmesh.is_passable() which combines A* reachability
with portal-width checks for agent clearance.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray

from crowdrl_core.navmesh import is_passable
from crowdrl_core.world_state import NavMesh


class SolvabilityMode(Enum):
    PRUNE = "prune"
    REGENERATE = "regenerate"
    STRICT = "strict"


def verify_solvability(
    navmesh: NavMesh,
    positions: NDArray[np.float64],
    goal_positions: NDArray[np.float64],
    agent_radii: NDArray[np.float64],
    mode: SolvabilityMode = SolvabilityMode.PRUNE,
    max_unsolvable_fraction: float = 0.3,
    clearance_factor: float = 1.2,
) -> NDArray[np.bool_] | None:
    """Verify that agents can reach their goals via the navmesh.

    Each agent's radius is checked against portal widths and geometric
    clearance along the A* path, so agents that are too wide for a
    bottleneck or a narrow gap between obstacles are correctly marked
    unsolvable.

    Parameters
    ----------
    navmesh : NavMesh
    positions : (n_agents, 2)
    goal_positions : (n_agents, 2)
    agent_radii : (n_agents,)
        Per-agent clearance radius (typically max(shoulder_width, chest_depth)).
    mode : SolvabilityMode
    max_unsolvable_fraction : float
        For REGENERATE mode: if more than this fraction is unsolvable,
        signal geometry regeneration.
    clearance_factor : float
        Safety margin multiplier for agent radius (default 1.2 = 20%).
        Accounts for agent rotation (widest orientation) and prevents
        agents from being routed through gaps they cannot physically
        traverse.

    Returns
    -------
    solvable_mask : (n_agents,) bool array or None
        True for solvable agents. Returns None if geometry should be
        regenerated (REGENERATE mode threshold exceeded, or STRICT mode
        with any unsolvable agent).
    """
    n_agents = len(positions)
    solvable = np.array(
        [
            is_passable(
                navmesh,
                positions[i],
                goal_positions[i],
                float(agent_radii[i]),
                clearance_factor,
            )
            for i in range(n_agents)
        ],
        dtype=np.bool_,
    )

    n_unsolvable = int(np.sum(~solvable))

    if mode == SolvabilityMode.STRICT:
        if n_unsolvable > 0:
            return None  # Signal: regenerate

    elif mode == SolvabilityMode.REGENERATE:
        if n_agents > 0 and n_unsolvable / n_agents > max_unsolvable_fraction:
            return None  # Signal: regenerate

    # PRUNE mode (or REGENERATE within threshold): return mask
    return solvable


def filter_by_solvability(
    solvable_mask: NDArray[np.bool_],
    *arrays: NDArray,
) -> tuple[NDArray, ...]:
    """Filter arrays to keep only solvable agents.

    Parameters
    ----------
    solvable_mask : (n_agents,) bool array
    *arrays : arrays with first axis = n_agents

    Returns
    -------
    Filtered arrays (only solvable agents kept).
    """
    return tuple(arr[solvable_mask] for arr in arrays)
