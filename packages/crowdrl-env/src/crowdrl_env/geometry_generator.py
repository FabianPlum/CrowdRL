"""Procedural geometry generator (Tiers 0–2).

All generators output Shapely Polygons with holes, matching JuPedSim convention.
Walkable area = polygon exterior; obstacles = polygon holes.

Tier 0: Open fields (convex polygons, no obstacles)
Tier 1: Corridors + bottlenecks (width 0.8–5.0m, aperture 0.6–2.0m)
Tier 2: Branching corridors, T-junctions, L-bends, crossroads

Higher tiers compose from lower-tier primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid


class GeometryTier(Enum):
    TIER_0 = 0
    TIER_1 = 1
    TIER_2 = 2


@dataclass(frozen=True)
class GeometryConfig:
    """Configuration for the geometry generator."""

    tier: GeometryTier = GeometryTier.TIER_0

    # Tier 0 params
    min_side: float = 8.0
    max_side: float = 25.0

    # Tier 1 params
    corridor_width_range: tuple[float, float] = (0.8, 5.0)
    corridor_length_range: tuple[float, float] = (8.0, 30.0)
    bottleneck_aperture_range: tuple[float, float] = (0.6, 2.0)
    bottleneck_depth_range: tuple[float, float] = (1.0, 3.0)

    # Tier 2 params
    branch_width_range: tuple[float, float] = (1.5, 4.0)
    branch_length_range: tuple[float, float] = (5.0, 15.0)


@dataclass
class GeneratedGeometry:
    """Output of the geometry generator."""

    polygon: Polygon
    """Walkable area (exterior) with obstacles (holes)."""

    spawn_regions: list[Polygon]
    """Regions where agents may be spawned."""

    goal_regions: list[Polygon]
    """Regions where agent goals may be placed."""

    tier: GeometryTier
    """Which tier generated this geometry."""

    metadata: dict = field(default_factory=dict)
    """Extra info (corridor width, bottleneck aperture, etc.)."""


def _ensure_valid_polygon(poly: Polygon) -> Polygon:
    """Ensure a polygon is valid, fixing if needed."""
    if not poly.is_valid:
        poly = make_valid(poly)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


# ---------------------------------------------------------------------------
# Tier 0: Open fields
# ---------------------------------------------------------------------------

def generate_rectangle(
    rng: np.random.Generator,
    config: GeometryConfig,
) -> GeneratedGeometry:
    """Generate a rectangular open field."""
    w = rng.uniform(config.min_side, config.max_side)
    h = rng.uniform(config.min_side, config.max_side)
    polygon = box(0, 0, w, h)

    margin = min(w, h) * 0.15
    spawn = box(margin, margin, w / 3, h - margin)
    goal = box(2 * w / 3, margin, w - margin, h - margin)

    return GeneratedGeometry(
        polygon=polygon,
        spawn_regions=[spawn],
        goal_regions=[goal],
        tier=GeometryTier.TIER_0,
        metadata={"width": w, "height": h, "shape": "rectangle"},
    )


def generate_convex_polygon(
    rng: np.random.Generator,
    config: GeometryConfig,
    n_vertices: int | None = None,
) -> GeneratedGeometry:
    """Generate a random convex polygon (open field)."""
    if n_vertices is None:
        n_vertices = rng.integers(4, 9)

    # Generate random angles, sort them, map to convex hull
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_vertices))
    radius_x = rng.uniform(config.min_side / 2, config.max_side / 2)
    radius_y = rng.uniform(config.min_side / 2, config.max_side / 2)
    cx = config.max_side / 2
    cy = config.max_side / 2

    coords = [
        (cx + radius_x * np.cos(a), cy + radius_y * np.sin(a))
        for a in angles
    ]
    polygon = Polygon(coords)
    polygon = _ensure_valid_polygon(polygon)

    # Spawn and goal: left and right thirds of bounding box
    minx, miny, maxx, maxy = polygon.bounds
    w = maxx - minx
    spawn = polygon.intersection(box(minx, miny, minx + w / 3, maxy))
    goal = polygon.intersection(box(maxx - w / 3, miny, maxx, maxy))

    # Ensure we have valid polygons
    if spawn.is_empty or not isinstance(spawn, Polygon):
        spawn = polygon
    if goal.is_empty or not isinstance(goal, Polygon):
        goal = polygon

    return GeneratedGeometry(
        polygon=polygon,
        spawn_regions=[spawn if isinstance(spawn, Polygon) else polygon],
        goal_regions=[goal if isinstance(goal, Polygon) else polygon],
        tier=GeometryTier.TIER_0,
        metadata={"shape": "convex", "n_vertices": n_vertices},
    )


def generate_tier0(
    rng: np.random.Generator,
    config: GeometryConfig,
) -> GeneratedGeometry:
    """Generate a random Tier 0 geometry."""
    if rng.random() < 0.5:
        return generate_rectangle(rng, config)
    return generate_convex_polygon(rng, config)


# ---------------------------------------------------------------------------
# Tier 1: Corridors + bottlenecks
# ---------------------------------------------------------------------------

def generate_corridor(
    rng: np.random.Generator,
    config: GeometryConfig,
) -> GeneratedGeometry:
    """Generate a straight corridor."""
    width = rng.uniform(*config.corridor_width_range)
    length = rng.uniform(*config.corridor_length_range)

    polygon = box(0, 0, length, width)

    margin = width * 0.2
    spawn_len = min(2.0, length * 0.2)
    spawn = box(margin, margin, spawn_len, width - margin)
    goal = box(length - spawn_len, margin, length - margin, width - margin)

    return GeneratedGeometry(
        polygon=polygon,
        spawn_regions=[spawn],
        goal_regions=[goal],
        tier=GeometryTier.TIER_1,
        metadata={"width": width, "length": length, "shape": "corridor"},
    )


def generate_bottleneck(
    rng: np.random.Generator,
    config: GeometryConfig,
) -> GeneratedGeometry:
    """Generate a corridor with a bottleneck (constriction) in the middle.

    Two wide rooms connected by a narrow passage.
    """
    corridor_width = rng.uniform(*config.corridor_width_range)
    corridor_width = max(corridor_width, 2.0)  # Rooms need to be wider than bottleneck
    length = rng.uniform(*config.corridor_length_range)
    aperture = rng.uniform(*config.bottleneck_aperture_range)
    bottleneck_depth = rng.uniform(*config.bottleneck_depth_range)

    # Ensure aperture < corridor width
    aperture = min(aperture, corridor_width * 0.8)

    # Exterior: full rectangle
    exterior = box(0, 0, length, corridor_width)

    # Two obstacle blocks creating the bottleneck
    bx_start = (length - bottleneck_depth) / 2
    bx_end = bx_start + bottleneck_depth
    gap_bottom = (corridor_width - aperture) / 2
    gap_top = corridor_width - gap_bottom

    obstacles = []
    # Bottom wall of bottleneck
    if gap_bottom > 0.05:
        obstacles.append(box(bx_start, 0, bx_end, gap_bottom))
    # Top wall of bottleneck
    if corridor_width - gap_top > 0.05:
        obstacles.append(box(bx_start, gap_top, bx_end, corridor_width))

    # Subtract obstacles from exterior
    polygon = exterior
    for obs in obstacles:
        polygon = polygon.difference(obs)
    polygon = _ensure_valid_polygon(polygon)

    # Spawn on left, goal on right
    margin = min(0.5, corridor_width * 0.1)
    spawn = box(margin, margin, bx_start - margin, corridor_width - margin)
    goal = box(bx_end + margin, margin, length - margin, corridor_width - margin)

    return GeneratedGeometry(
        polygon=polygon,
        spawn_regions=[spawn],
        goal_regions=[goal],
        tier=GeometryTier.TIER_1,
        metadata={
            "width": corridor_width,
            "length": length,
            "aperture": aperture,
            "bottleneck_depth": bottleneck_depth,
            "shape": "bottleneck",
        },
    )


def generate_tier1(
    rng: np.random.Generator,
    config: GeometryConfig,
) -> GeneratedGeometry:
    """Generate a random Tier 1 geometry."""
    if rng.random() < 0.4:
        return generate_corridor(rng, config)
    return generate_bottleneck(rng, config)


# ---------------------------------------------------------------------------
# Tier 2: Branching corridors, T-junctions, L-bends, crossroads
# ---------------------------------------------------------------------------

def generate_l_bend(
    rng: np.random.Generator,
    config: GeometryConfig,
) -> GeneratedGeometry:
    """Generate an L-shaped corridor."""
    w = rng.uniform(*config.branch_width_range)
    l1 = rng.uniform(*config.branch_length_range)
    l2 = rng.uniform(*config.branch_length_range)

    # L-shape: horizontal segment + vertical segment at the end
    # Horizontal: (0, 0) to (l1, w)
    # Vertical: (l1-w, 0) to (l1, l2)  (overlaps at the corner)
    horiz = box(0, 0, l1, w)
    vert = box(l1 - w, 0, l1, l2)
    polygon = unary_union([horiz, vert])
    polygon = _ensure_valid_polygon(polygon)

    margin = w * 0.2
    spawn = box(margin, margin, min(2.0, l1 * 0.2), w - margin)
    goal = box(l1 - w + margin, l2 - min(2.0, l2 * 0.2), l1 - margin, l2 - margin)

    return GeneratedGeometry(
        polygon=polygon,
        spawn_regions=[spawn],
        goal_regions=[goal],
        tier=GeometryTier.TIER_2,
        metadata={"width": w, "length_h": l1, "length_v": l2, "shape": "l_bend"},
    )


def generate_t_junction(
    rng: np.random.Generator,
    config: GeometryConfig,
) -> GeneratedGeometry:
    """Generate a T-junction: main corridor with a perpendicular branch."""
    w = rng.uniform(*config.branch_width_range)
    main_length = rng.uniform(*config.branch_length_range)
    branch_length = rng.uniform(config.branch_length_range[0], config.branch_length_range[1] * 0.8)

    # Main corridor: horizontal
    main = box(0, 0, main_length, w)

    # Branch: vertical, centred on main corridor
    branch_x = rng.uniform(main_length * 0.3, main_length * 0.7)
    branch = box(branch_x - w / 2, 0, branch_x + w / 2, branch_length)

    polygon = unary_union([main, branch])
    polygon = _ensure_valid_polygon(polygon)

    margin = w * 0.2
    # Three possible goal/spawn points: left end, right end, branch end
    spawn_left = box(margin, margin, min(2.0, main_length * 0.15), w - margin)
    goal_right = box(
        main_length - min(2.0, main_length * 0.15),
        margin,
        main_length - margin,
        w - margin,
    )
    goal_branch = box(
        branch_x - w / 2 + margin,
        branch_length - min(2.0, branch_length * 0.2),
        branch_x + w / 2 - margin,
        branch_length - margin,
    )

    return GeneratedGeometry(
        polygon=polygon,
        spawn_regions=[spawn_left],
        goal_regions=[goal_right, goal_branch],
        tier=GeometryTier.TIER_2,
        metadata={
            "width": w,
            "main_length": main_length,
            "branch_length": branch_length,
            "branch_x": branch_x,
            "shape": "t_junction",
        },
    )


def generate_crossroads(
    rng: np.random.Generator,
    config: GeometryConfig,
) -> GeneratedGeometry:
    """Generate a crossroads: two perpendicular corridors."""
    w = rng.uniform(*config.branch_width_range)
    l_h = rng.uniform(*config.branch_length_range)
    l_v = rng.uniform(*config.branch_length_range)

    # Centre the cross at the midpoints
    cx, cy = l_h / 2, l_v / 2
    horiz = box(0, cy - w / 2, l_h, cy + w / 2)
    vert = box(cx - w / 2, 0, cx + w / 2, l_v)

    polygon = unary_union([horiz, vert])
    polygon = _ensure_valid_polygon(polygon)

    margin = w * 0.2
    # Four endpoints as spawn/goal regions
    spawn_left = box(margin, cy - w / 2 + margin, min(2.0, l_h * 0.15), cy + w / 2 - margin)
    goal_right = box(
        l_h - min(2.0, l_h * 0.15), cy - w / 2 + margin,
        l_h - margin, cy + w / 2 - margin,
    )
    goal_top = box(
        cx - w / 2 + margin, l_v - min(2.0, l_v * 0.15),
        cx + w / 2 - margin, l_v - margin,
    )
    goal_bottom = box(
        cx - w / 2 + margin, margin,
        cx + w / 2 - margin, min(2.0, l_v * 0.15),
    )

    return GeneratedGeometry(
        polygon=polygon,
        spawn_regions=[spawn_left, goal_bottom],
        goal_regions=[goal_right, goal_top],
        tier=GeometryTier.TIER_2,
        metadata={"width": w, "length_h": l_h, "length_v": l_v, "shape": "crossroads"},
    )


def generate_tier2(
    rng: np.random.Generator,
    config: GeometryConfig,
) -> GeneratedGeometry:
    """Generate a random Tier 2 geometry."""
    choice = rng.random()
    if choice < 0.33:
        return generate_l_bend(rng, config)
    elif choice < 0.66:
        return generate_t_junction(rng, config)
    return generate_crossroads(rng, config)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_geometry(
    rng: np.random.Generator | None = None,
    config: GeometryConfig | None = None,
) -> GeneratedGeometry:
    """Generate a random geometry at the configured tier.

    Parameters
    ----------
    rng : np.random.Generator or None
        Random number generator. If None, creates a new default one.
    config : GeometryConfig or None
        Configuration. If None, uses defaults (Tier 0).

    Returns
    -------
    GeneratedGeometry
    """
    if rng is None:
        rng = np.random.default_rng()
    if config is None:
        config = GeometryConfig()

    generators = {
        GeometryTier.TIER_0: generate_tier0,
        GeometryTier.TIER_1: generate_tier1,
        GeometryTier.TIER_2: generate_tier2,
    }

    return generators[config.tier](rng, config)
