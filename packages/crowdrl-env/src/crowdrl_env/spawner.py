"""Crowd composition sampler and agent spawner.

Samples agent count, body dimensions (from anthropometric distributions),
desired speeds, spawn positions, and goal positions for each episode.
Returns arrays ready to populate a WorldState.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

from crowdrl_core.geometry import sample_point_in_polygon


@dataclass(frozen=True)
class SpawnConfig:
    """Configuration for crowd composition and spawning."""

    n_agents_range: tuple[int, int] = (5, 30)
    """(min, max) agent count per episode."""

    # Anthropometric body dimensions (half-widths for collision ellipse)
    shoulder_width_mean: float = 0.22
    """Mean half-shoulder-width (metres). Full shoulder ~0.44m."""
    shoulder_width_std: float = 0.02

    chest_depth_mean: float = 0.12
    """Mean half-chest-depth (metres). Full chest ~0.24m."""
    chest_depth_std: float = 0.015

    # Desired speed distribution
    preferred_speed_mean: float = 1.34
    """Mean preferred walking speed (m/s). Literature: ~1.34 m/s."""
    preferred_speed_std: float = 0.26

    min_body_dim: float = 0.08
    """Minimum allowed half-dimension for collision ellipse (metres)."""

    min_speed: float = 0.5
    """Minimum preferred speed (m/s)."""

    max_speed: float = 2.0
    """Maximum preferred speed (m/s)."""

    min_spawn_separation: float = 0.5
    """Minimum distance between spawned agents (metres)."""

    max_spawn_attempts: int = 50
    """Maximum rejection-sampling attempts per agent before giving up."""


@dataclass
class SpawnResult:
    """Output of the spawner: arrays ready for WorldState."""

    positions: NDArray[np.float64]
    """(n_agents, 2)"""
    velocities: NDArray[np.float64]
    """(n_agents, 2) — initialised to zero."""
    torso_orientations: NDArray[np.float64]
    """(n_agents,) — initial heading toward goal."""
    head_orientations: NDArray[np.float64]
    """(n_agents,) — same as torso initially."""
    shoulder_widths: NDArray[np.float64]
    """(n_agents,)"""
    chest_depths: NDArray[np.float64]
    """(n_agents,)"""
    goal_positions: NDArray[np.float64]
    """(n_agents, 2)"""
    preferred_speeds: NDArray[np.float64]
    """(n_agents,) — per-agent preferred speed for reward computation."""

    @property
    def n_agents(self) -> int:
        return self.positions.shape[0]


def spawn_agents(
    rng: np.random.Generator,
    spawn_regions: list[Polygon],
    goal_regions: list[Polygon],
    config: SpawnConfig = SpawnConfig(),
    n_agents: int | None = None,
    walkable: Polygon | None = None,
) -> SpawnResult:
    """Sample a crowd of agents with heterogeneous body dimensions and goals.

    Parameters
    ----------
    rng : np.random.Generator
    spawn_regions : list of Shapely Polygons
        Regions from which agent positions are sampled.
    goal_regions : list of Shapely Polygons
        Regions from which agent goals are sampled.
    config : SpawnConfig
    n_agents : int or None
        If given, overrides config.n_agents_range.
    walkable : Polygon, optional
        The walkable area polygon. When given, all spawned positions and
        goals are guaranteed to lie inside this polygon with a margin
        equal to the agent's body radius (like JuPedSim's InsideGeometry
        + radius check). Spawn/goal regions are clipped to this area.

    Returns
    -------
    SpawnResult
    """
    if n_agents is None:
        n_agents = int(rng.integers(config.n_agents_range[0], config.n_agents_range[1] + 1))

    # Sample body dimensions from truncated normal distributions
    shoulder_widths = np.clip(
        rng.normal(config.shoulder_width_mean, config.shoulder_width_std, n_agents),
        config.min_body_dim,
        None,
    )
    chest_depths = np.clip(
        rng.normal(config.chest_depth_mean, config.chest_depth_std, n_agents),
        config.min_body_dim,
        None,
    )

    # Sample preferred speeds
    preferred_speeds = np.clip(
        rng.normal(config.preferred_speed_mean, config.preferred_speed_std, n_agents),
        config.min_speed,
        config.max_speed,
    )

    # Maximum body radius across all agents
    max_body_radius = float(np.max(np.maximum(shoulder_widths, chest_depths)))

    # Minimum separation must be at least 2× the largest body radius
    # so that no two agent ellipses overlap at spawn
    min_sep = max(config.min_spawn_separation, 2.0 * max_body_radius)

    # Sample positions with minimum separation (rejection sampling)
    positions = _sample_separated_points(
        rng,
        spawn_regions,
        n_agents,
        min_sep,
        config.max_spawn_attempts,
        walkable=walkable,
        margin=max_body_radius if walkable is not None else 0.0,
    )
    actual_n = len(positions)

    # Trim arrays if some agents couldn't be placed
    if actual_n < n_agents:
        shoulder_widths = shoulder_widths[:actual_n]
        chest_depths = chest_depths[:actual_n]
        preferred_speeds = preferred_speeds[:actual_n]

    # Sample goal positions (one per agent, randomly from goal regions)
    goal_kw: dict = {}
    if walkable is not None:
        goal_kw = {"margin": max_body_radius, "walkable": walkable}
    goal_positions = np.array(
        [
            sample_point_in_polygon(goal_regions[rng.integers(len(goal_regions))], rng, **goal_kw)
            for _ in range(actual_n)
        ],
        dtype=np.float64,
    )

    # Initial orientation: face toward goal
    diff = goal_positions - positions
    torso_orientations = np.arctan2(diff[:, 1], diff[:, 0])
    head_orientations = torso_orientations.copy()

    # Start stationary
    velocities = np.zeros((actual_n, 2), dtype=np.float64)

    return SpawnResult(
        positions=positions,
        velocities=velocities,
        torso_orientations=torso_orientations,
        head_orientations=head_orientations,
        shoulder_widths=shoulder_widths,
        chest_depths=chest_depths,
        goal_positions=goal_positions,
        preferred_speeds=preferred_speeds,
    )


def _sample_separated_points(
    rng: np.random.Generator,
    regions: list[Polygon],
    n_points: int,
    min_sep: float,
    max_attempts: int,
    walkable: Polygon | None = None,
    margin: float = 0.0,
) -> NDArray[np.float64]:
    """Sample points with minimum pairwise separation via rejection sampling.

    Returns up to n_points positions (may be fewer if placement is tight).
    """
    placed: list[NDArray[np.float64]] = []

    for _ in range(n_points):
        region = regions[rng.integers(len(regions))]
        success = False
        for _attempt in range(max_attempts):
            candidate = sample_point_in_polygon(region, rng, margin=margin, walkable=walkable)
            if all(np.linalg.norm(candidate - p) >= min_sep for p in placed):
                placed.append(candidate)
                success = True
                break
        if not success:
            # Couldn't place this agent — geometry too tight
            continue

    if not placed:
        raise ValueError("Could not place any agents in the given spawn regions")

    return np.array(placed, dtype=np.float64)
