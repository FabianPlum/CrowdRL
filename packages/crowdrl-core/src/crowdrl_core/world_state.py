"""WorldState — the critical interface between training and deployment.

Both crowdrl-env and crowdrl-jupedsim populate WorldState. The observation builder
and sensing modules consume only WorldState — they never know which system produced it.

If WorldState is populated correctly, observations are numerically identical
between training and deployment. This is the transfer guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from shapely.geometry import Polygon


@dataclass
class NavMesh:
    """Precomputed navigation mesh for A* pathfinding + funnel algorithm.

    Built once per geometry (on reset), reused every step.
    """

    triangles: NDArray[np.float64]
    """(T, 3, 2) — vertex coordinates of each triangle."""

    centroids: NDArray[np.float64]
    """(T, 2) — centroid of each triangle."""

    adjacency: list[list[int]]
    """adjacency[i] = list of triangle indices adjacent to triangle i."""

    portals: dict[tuple[int, int], tuple[NDArray[np.float64], NDArray[np.float64]]]
    """portals[(i, j)] = (left, right) — the shared edge endpoints between
    triangles i and j, oriented so that 'left' is to the left of the
    travel direction from i to j. Used by the funnel algorithm."""

    polygon: Polygon = field(default=None)  # type: ignore[assignment]
    """The walkable polygon used to build this navmesh. Stored for geometric
    clearance validation in the solvability checker."""


@dataclass
class WorldState:
    """Flat snapshot of everything the perception system needs.

    All arrays are indexed by agent index [0, n_agents).
    Coordinate system: global 2D (metres). Angles in radians.
    """

    # --- Agent state arrays (all shape (n_agents, ...)) ---

    positions: NDArray[np.float64]
    """(n_agents, 2) — [x, y] position of each agent's centre."""

    velocities: NDArray[np.float64]
    """(n_agents, 2) — [vx, vy] velocity of each agent."""

    torso_orientations: NDArray[np.float64]
    """(n_agents,) — torso angle in radians (0 = +x, CCW positive)."""

    head_orientations: NDArray[np.float64]
    """(n_agents,) — head angle in radians, *absolute* (not relative to torso).
    The observation builder computes the relative angle internally."""

    shoulder_widths: NDArray[np.float64]
    """(n_agents,) — half-width of the collision ellipse along the torso-perpendicular axis (metres)."""

    chest_depths: NDArray[np.float64]
    """(n_agents,) — half-depth of the collision ellipse along the torso-forward axis (metres)."""

    masses: NDArray[np.float64]
    """(n_agents,) — agent mass in kg (default ~80 kg). Used for F=ma in contact forces."""

    goal_positions: NDArray[np.float64]
    """(n_agents, 2) — [x, y] goal position for each agent."""

    # --- Geometry (shared across all agents) ---

    walkable_polygon: Polygon = field(default=None)  # type: ignore[assignment]
    """Shapely Polygon with holes. Exterior = walkable boundary, holes = obstacles."""

    wall_segments: NDArray[np.float64] = field(default=None)  # type: ignore[assignment]
    """(S, 2, 2) — precomputed wall segments [[x1,y1],[x2,y2]] from polygon boundaries."""

    navmesh: NavMesh | None = None
    """Precomputed navigation mesh. None if navmesh signals are disabled."""

    # --- Masks ---

    active_mask: NDArray[np.bool_] | None = None
    """(n_agents,) — True if agent is active (hasn't reached goal / been removed).
    None means all agents are active."""

    # --- Temporal memory state (optional; populated when ObsConfig.use_temporal_memory is True) ---
    # See ``crowdrl_core.observation`` for the semantics. Adapters that do not
    # enable temporal memory can leave these as None.

    spawn_positions: NDArray[np.float64] | None = None
    """(n_agents, 2) — position at the start of the current episode, frozen
    after reset. Used to compute displacement-from-spawn and path efficiency.
    """

    initial_goal_distances: NDArray[np.float64] | None = None
    """(n_agents,) — ||goal_positions - spawn_positions|| at the start of the
    episode, used as the normalising constant for spawn-relative features.
    """

    cumulative_path_length: NDArray[np.float64] | None = None
    """(n_agents,) — running total of per-step position deltas, accumulated
    from reset. Used for the path-efficiency ratio and cumulative path length
    feature.
    """

    pos_history: NDArray[np.float64] | None = None
    """(n_agents, W+1, 2) — ring buffer of recent agent positions, where W =
    ``ObsConfig.temporal_memory_window``. Writes happen at index
    ``step_count % (W+1)``; reads from ``(step_count + 1) % (W+1)`` return
    the position W steps ago. Initialised to spawn_positions at reset so
    early reads (t < W) return the spawn position.
    """

    gdist_history: NDArray[np.float64] | None = None
    """(n_agents, W+1) — ring buffer of recent goal distances, indexed the
    same way as ``pos_history``.
    """

    preferred_speeds: NDArray[np.float64] | None = None
    """(n_agents,) — per-agent preferred walking speed, used to normalise
    the window-based displacement and goal-progress features. Optional: when
    None the obs builder falls back to a constant 1.3 m/s.
    """

    step_count: int = 0
    """Current episode step (same for all agents in an episode). Used to
    compute elapsed_fraction and to read the correct slot from the
    ring buffers.
    """

    @property
    def n_agents(self) -> int:
        return self.positions.shape[0]

    def validate(self) -> None:
        """Raise ValueError if array shapes are inconsistent."""
        n = self.n_agents
        checks = {
            "velocities": (self.velocities, (n, 2)),
            "torso_orientations": (self.torso_orientations, (n,)),
            "head_orientations": (self.head_orientations, (n,)),
            "shoulder_widths": (self.shoulder_widths, (n,)),
            "chest_depths": (self.chest_depths, (n,)),
            "masses": (self.masses, (n,)),
            "goal_positions": (self.goal_positions, (n, 2)),
        }
        for name, (arr, expected_shape) in checks.items():
            if arr.shape != expected_shape:
                raise ValueError(f"{name} has shape {arr.shape}, expected {expected_shape}")
        if self.active_mask is not None and self.active_mask.shape != (n,):
            raise ValueError(f"active_mask has shape {self.active_mask.shape}, expected ({n},)")
        if self.wall_segments is not None and self.wall_segments.ndim != 3:
            raise ValueError(
                f"wall_segments must be (S, 2, 2), got shape {self.wall_segments.shape}"
            )
