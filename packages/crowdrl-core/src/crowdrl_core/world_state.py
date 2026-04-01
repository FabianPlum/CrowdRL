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
