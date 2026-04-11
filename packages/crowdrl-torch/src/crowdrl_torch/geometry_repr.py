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
    masses: NDArray | None = None,
    waypoints: NDArray | None = None,
    n_waypoints: NDArray | None = None,
    waypoint_path_lengths: NDArray | None = None,
    max_waypoints: int = 16,
    memory_window: int = 50,
) -> dict[str, NDArray[np.float32]]:
    """Prepare CPU-generated episode data for GPU transfer.

    Pads all agent arrays to MAX_AGENTS and wall segments to MAX_SEGMENTS.

    ``memory_window`` sets the temporal-memory ring buffer size to
    ``memory_window + 1``. The buffers are pre-filled with the spawn position
    (resp. initial goal distance) so that reads before the buffer fills
    return a sensible default.

    Returns
    -------
    dict of padded float32 arrays ready for ``torch.tensor()``
    """
    n_agents = len(positions)
    if n_agents > max_agents:
        raise ValueError(f"Episode has {n_agents} agents, exceeds MAX_AGENTS={max_agents}.")

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

    # Waypoint arrays — zero-padded to (MAX_AGENTS, MAX_WP, ...)
    padded_wp = np.zeros((max_agents, max_waypoints, 2), dtype=np.float32)
    padded_n_wp = np.zeros(max_agents, dtype=np.int32)
    padded_wp_pl = np.zeros((max_agents, max_waypoints), dtype=np.float32)
    if waypoints is not None:
        n = min(n_agents, waypoints.shape[0])
        padded_wp[:n, : waypoints.shape[1]] = waypoints[:n].astype(np.float32)
    if n_waypoints is not None:
        padded_n_wp[:n_agents] = n_waypoints[:n_agents].astype(np.int32)
    if waypoint_path_lengths is not None:
        n = min(n_agents, waypoint_path_lengths.shape[0])
        padded_wp_pl[:n, : waypoint_path_lengths.shape[1]] = waypoint_path_lengths[:n].astype(
            np.float32
        )

    # Temporal-memory ring buffers. Pre-fill all W+1 slots with the spawn
    # position / initial goal distance so that reads before the buffer fills
    # return the spawn value (equivalent to "we haven't moved from here yet").
    buf_size = memory_window + 1
    padded_pos_history = np.zeros((max_agents, buf_size, 2), dtype=np.float32)
    padded_pos_history[:n_agents] = positions[:n_agents, np.newaxis, :].astype(np.float32)
    padded_gdist_history = np.zeros((max_agents, buf_size), dtype=np.float32)
    padded_gdist_history[:n_agents] = goal_dists[:, np.newaxis]

    return {
        "positions": pad_2d(positions, max_agents),
        "velocities": pad_2d(velocities, max_agents),
        "torso_orientations": pad_1d(torso_orientations, max_agents),
        "head_orientations": pad_1d(head_orientations, max_agents),
        "shoulder_widths": pad_1d(shoulder_widths, max_agents),
        "chest_depths": pad_1d(chest_depths, max_agents),
        "masses": pad_1d(
            masses if masses is not None else np.full(n_agents, 80.0, dtype=np.float64), max_agents
        ),
        "goal_positions": pad_2d(goal_positions, max_agents),
        "preferred_speeds": pad_1d(preferred_speeds, max_agents),
        "active_mask": active,
        "wall_segments": padded_segs,
        "n_segments": n_segs,
        "n_agents": n_agents,
        "goal_distances": padded_goal_dists,
        "waypoints": padded_wp,
        "n_waypoints": padded_n_wp,
        "waypoint_path_lengths": padded_wp_pl,
        "spawn_positions": pad_2d(positions, max_agents),
        "initial_goal_distances": padded_goal_dists.copy(),
        "pos_history": padded_pos_history,
        "gdist_history": padded_gdist_history,
    }
