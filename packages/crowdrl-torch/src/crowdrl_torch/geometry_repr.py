"""Convert CPU-side Shapely geometry to GPU-compatible arrays.

Used at episode reset time to transfer geometry from the CPU
(Shapely polygon + spawner) to GPU (padded tensors).

This module is pure NumPy — no PyTorch dependency. The resulting
arrays are transferred to tensors by BatchedTorchEnv.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def polygon_to_segments(
    wall_segments: NDArray[np.float64],
    max_segments: int,
) -> tuple[NDArray[np.float32], int]:
    """Pad wall segments to fixed MAX_SEGMENTS shape.

    Parameters
    ----------
    wall_segments : (S, 2, 2) — from WorldState.wall_segments
    max_segments : int — target padding size

    Returns
    -------
    padded : (max_segments, 2, 2) float32
    n_segments : int — actual segment count
    """
    n_segments = len(wall_segments)
    if n_segments > max_segments:
        raise ValueError(
            f"Geometry has {n_segments} wall segments, "
            f"exceeds MAX_SEGMENTS={max_segments}. "
            f"Increase max_segments or simplify geometry."
        )

    padded = np.zeros((max_segments, 2, 2), dtype=np.float32)
    padded[:n_segments] = wall_segments[:n_segments].astype(np.float32)
    return padded, n_segments


def prepare_reset_data(
    positions: NDArray,
    velocities: NDArray,
    torso_orientations: NDArray,
    head_orientations: NDArray,
    shoulder_widths: NDArray,
    chest_depths: NDArray,
    goal_positions: NDArray,
    preferred_speeds: NDArray,
    wall_segments: NDArray,
    max_agents: int,
    max_segments: int,
) -> dict[str, NDArray[np.float32]]:
    """Prepare CPU-generated episode data for GPU transfer.

    Pads all agent arrays to MAX_AGENTS and wall segments to MAX_SEGMENTS.

    Returns
    -------
    dict of padded float32 arrays ready for ``torch.tensor()``
    """
    n_agents = len(positions)
    if n_agents > max_agents:
        raise ValueError(
            f"Episode has {n_agents} agents, exceeds MAX_AGENTS={max_agents}."
        )

    def pad_1d(arr: NDArray, size: int) -> NDArray[np.float32]:
        out = np.zeros(size, dtype=np.float32)
        out[: len(arr)] = arr.astype(np.float32)
        return out

    def pad_2d(arr: NDArray, size: int) -> NDArray[np.float32]:
        out = np.zeros((size, arr.shape[1]), dtype=np.float32)
        out[: len(arr)] = arr.astype(np.float32)
        return out

    padded_segs, n_segs = polygon_to_segments(wall_segments, max_segments)

    active = np.zeros(max_agents, dtype=np.bool_)
    active[:n_agents] = True

    goal_dists = np.linalg.norm(goal_positions - positions, axis=1).astype(np.float32)
    padded_goal_dists = np.zeros(max_agents, dtype=np.float32)
    padded_goal_dists[:n_agents] = goal_dists

    return {
        "positions": pad_2d(positions, max_agents),
        "velocities": pad_2d(velocities, max_agents),
        "torso_orientations": pad_1d(torso_orientations, max_agents),
        "head_orientations": pad_1d(head_orientations, max_agents),
        "shoulder_widths": pad_1d(shoulder_widths, max_agents),
        "chest_depths": pad_1d(chest_depths, max_agents),
        "goal_positions": pad_2d(goal_positions, max_agents),
        "preferred_speeds": pad_1d(preferred_speeds, max_agents),
        "active_mask": active,
        "wall_segments": padded_segs,
        "n_segments": n_segs,
        "n_agents": n_agents,
        "goal_distances": padded_goal_dists,
    }
