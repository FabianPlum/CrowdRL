"""Shared test fixtures for crowdrl-core tests.

Hand-built WorldState instances for testing in isolation, as per design principle #4.
"""

import numpy as np
import pytest
from shapely.geometry import Polygon

from crowdrl_core.geometry import build_navmesh, extract_wall_segments
from crowdrl_core.world_state import WorldState


@pytest.fixture
def simple_square_polygon() -> Polygon:
    """10m × 10m square walkable area, no obstacles."""
    return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


@pytest.fixture
def corridor_polygon() -> Polygon:
    """20m × 3m corridor."""
    return Polygon([(0, 0), (20, 0), (20, 3), (0, 3)])


@pytest.fixture
def square_with_obstacle() -> Polygon:
    """10m × 10m square with a 2m × 2m obstacle in the centre."""
    exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(4, 4), (6, 4), (6, 6), (4, 6)]
    return Polygon(exterior, [hole])


@pytest.fixture
def bottleneck_polygon() -> Polygon:
    """10m × 6m area with a 1m-wide bottleneck in the middle.

    Shape: two 4m-wide rooms connected by a 1m-wide, 2m-long passage.
    """
    exterior = [(0, 0), (10, 0), (10, 6), (0, 6)]
    # Two obstacles creating the bottleneck
    top_wall = [(4, 3.5), (6, 3.5), (6, 6), (4, 6)]
    bottom_wall = [(4, 0), (6, 0), (6, 2.5), (4, 2.5)]
    return Polygon(exterior, [top_wall, bottom_wall])


def make_world_state(
    n_agents: int = 2,
    polygon: Polygon | None = None,
    positions: np.ndarray | None = None,
    velocities: np.ndarray | None = None,
    goal_positions: np.ndarray | None = None,
    torso_orientations: np.ndarray | None = None,
    head_orientations: np.ndarray | None = None,
    shoulder_widths: np.ndarray | None = None,
    chest_depths: np.ndarray | None = None,
    build_nav: bool = False,
) -> WorldState:
    """Helper to build WorldState with sensible defaults."""
    if polygon is None:
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    if positions is None:
        positions = np.array([[2.0, 5.0], [8.0, 5.0]])[:n_agents]
    if velocities is None:
        velocities = np.zeros((n_agents, 2), dtype=np.float64)
    if goal_positions is None:
        goal_positions = np.array([[8.0, 5.0], [2.0, 5.0]])[:n_agents]
    if torso_orientations is None:
        torso_orientations = np.zeros(n_agents, dtype=np.float64)
    if head_orientations is None:
        head_orientations = np.zeros(n_agents, dtype=np.float64)
    if shoulder_widths is None:
        shoulder_widths = np.full(n_agents, 0.23, dtype=np.float64)
    if chest_depths is None:
        chest_depths = np.full(n_agents, 0.15, dtype=np.float64)

    wall_segments = extract_wall_segments(polygon)
    navmesh = build_navmesh(polygon) if build_nav else None

    return WorldState(
        positions=positions,
        velocities=velocities,
        torso_orientations=torso_orientations,
        head_orientations=head_orientations,
        shoulder_widths=shoulder_widths,
        chest_depths=chest_depths,
        goal_positions=goal_positions,
        walkable_polygon=polygon,
        wall_segments=wall_segments,
        navmesh=navmesh,
    )
