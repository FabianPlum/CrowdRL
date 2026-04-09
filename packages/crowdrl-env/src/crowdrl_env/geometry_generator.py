"""Procedural geometry generator (Tiers 0–3b).

All generators output Shapely Polygons with holes, matching JuPedSim convention.
Walkable area = polygon exterior; obstacles = polygon holes.

Tier 0: Open fields (convex polygons, no obstacles)
Tier 1: Corridors + bottlenecks (width 0.8–5.0m, aperture 0.6–2.0m)
Tier 2: Branching corridors, T-junctions, L-bends, crossroads
Tier 3a: Base room (Tier 0–2) + internal obstacles + doors; optional shared goal area
Tier 3b: 2–3 composed rooms (Tier 0–2) connected by doors + obstacles + evacuation areas

Higher tiers compose from lower-tier primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid


# Minimum wall clearance for spawn/goal regions.  Agent centres placed at the
# region boundary must not have their body ellipse overlap a wall.  This equals
# SpawnConfig.shoulder_width_mean (the half-shoulder-width / worst-case body
# radius for a front-facing agent).
_MIN_SPAWN_WALL_MARGIN = 0.22


class GeometryTier(Enum):
    TIER_0 = 0
    TIER_1 = 1
    TIER_2 = 2
    TIER_3A = 3
    TIER_3B = 4


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

    # Minimum passage width (metres) — any opening or corridor connection
    # must be at least this wide for the geometry to be valid.  Derived from
    # the anthropometric distribution: 2 * (mean_shoulder + 3*std) * 1.2
    # safety factor ~= 0.66m.  Ensures even the widest agents (at their
    # widest orientation + 20% margin) can physically pass through.
    min_passage_width: float = 0.7

    # Tier 3 params (shared by 3a and 3b)
    obstacle_coverage_range: tuple[float, float] = (0.05, 0.20)
    """Fraction of floor area covered by obstacles."""
    obstacle_min_size: float = 0.3
    obstacle_max_size: float = 1.5
    """Side length range for rectangular obstacles (metres)."""
    column_radius_range: tuple[float, float] = (0.15, 0.4)
    """Radius range for circular column obstacles (metres)."""
    column_quad_segs: int = 4
    """Segments per quadrant when discretising circular columns (total vertices = 4 × quad_segs)."""
    door_width_range: tuple[float, float] = (0.8, 2.0)
    """Width range for door openings (metres)."""
    shared_goal_probability: float = 0.4
    """Probability that all agents share a single goal area (Tier 3a)."""
    n_rooms_range: tuple[int, int] = (2, 3)
    """Number of rooms to compose in Tier 3b."""
    max_wall_segments: int = 512
    """Hard cap on wall segments. Polygon is progressively simplified if exceeded."""


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


def _clip_regions(
    regions: list[Polygon],
    walkable: Polygon,
    margin: float = _MIN_SPAWN_WALL_MARGIN,
    min_area: float = 0.1,
) -> list[Polygon]:
    """Clip spawn/goal regions to lie safely within the walkable polygon.

    The walkable polygon is eroded inward by *margin* before clipping so
    that agent centres stay at least *margin* away from walls and obstacle
    edges.  Regions that become empty or too small after clipping are
    discarded.
    """
    eroded = walkable.buffer(-margin)
    if eroded.is_empty:
        return []
    if isinstance(eroded, MultiPolygon):
        eroded = max(eroded.geoms, key=lambda g: g.area)

    clipped: list[Polygon] = []
    for r in regions:
        if r.is_empty:
            continue
        isect = eroded.intersection(r)
        if isect.is_empty:
            continue
        # intersection may return a collection; keep the largest polygon
        if isinstance(isect, MultiPolygon):
            isect = max(isect.geoms, key=lambda g: g.area)
        if not isinstance(isect, Polygon):
            continue
        if isect.area < min_area:
            continue
        clipped.append(isect)
    return clipped


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

    margin = max(min(w, h) * 0.15, _MIN_SPAWN_WALL_MARGIN)
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

    coords = [(cx + radius_x * np.cos(a), cy + radius_y * np.sin(a)) for a in angles]
    polygon = Polygon(coords)
    polygon = _ensure_valid_polygon(polygon)

    # Erode the polygon inward so spawn/goal regions respect wall clearance.
    # This also eliminates sharp corner tips where no agent body would fit.
    eroded = polygon.buffer(-_MIN_SPAWN_WALL_MARGIN)
    if eroded.is_empty or not isinstance(eroded, Polygon):
        eroded = polygon  # fallback for very small polygons

    # Spawn and goal: left and right thirds of eroded bounding box
    minx, miny, maxx, maxy = eroded.bounds
    w = maxx - minx
    spawn = eroded.intersection(box(minx, miny, minx + w / 3, maxy))
    goal = eroded.intersection(box(maxx - w / 3, miny, maxx, maxy))

    # Ensure we have valid polygons
    if spawn.is_empty or not isinstance(spawn, Polygon):
        spawn = eroded
    if goal.is_empty or not isinstance(goal, Polygon):
        goal = eroded

    return GeneratedGeometry(
        polygon=polygon,
        spawn_regions=[spawn if isinstance(spawn, Polygon) else eroded],
        goal_regions=[goal if isinstance(goal, Polygon) else eroded],
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

    margin = _MIN_SPAWN_WALL_MARGIN
    spawn_len = length * 0.25
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

    # Ensure aperture is at least min_passage_width (so agents can fit)
    # and less than corridor width
    aperture = max(aperture, config.min_passage_width)
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
    margin = _MIN_SPAWN_WALL_MARGIN
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

    margin = _MIN_SPAWN_WALL_MARGIN
    spawn = box(margin, margin, l1 * 0.25, w - margin)
    goal = box(l1 - w + margin, l2 - l2 * 0.25, l1 - margin, l2 - margin)

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

    margin = _MIN_SPAWN_WALL_MARGIN
    # Three possible goal/spawn points: left end, right end, branch end
    spawn_left = box(margin, margin, main_length * 0.25, w - margin)
    goal_right = box(
        main_length - main_length * 0.25,
        margin,
        main_length - margin,
        w - margin,
    )
    goal_branch = box(
        branch_x - w / 2 + margin,
        branch_length - branch_length * 0.25,
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

    margin = _MIN_SPAWN_WALL_MARGIN
    # Four endpoints as spawn/goal regions
    spawn_left = box(margin, cy - w / 2 + margin, l_h * 0.25, cy + w / 2 - margin)
    goal_right = box(
        l_h - l_h * 0.25,
        cy - w / 2 + margin,
        l_h - margin,
        cy + w / 2 - margin,
    )
    goal_top = box(
        cx - w / 2 + margin,
        l_v - l_v * 0.25,
        cx + w / 2 - margin,
        l_v - margin,
    )
    goal_bottom = box(
        cx - w / 2 + margin,
        margin,
        cx + w / 2 - margin,
        l_v * 0.25,
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
# Tier 3 helpers: obstacle placement + door cutting
# ---------------------------------------------------------------------------


def _place_obstacles(
    rng: np.random.Generator,
    walkable: Polygon,
    config: GeometryConfig,
    max_attempts: int = 50,
) -> list[Polygon]:
    """Place random obstacles (rectangles + columns) inside a walkable area.

    Obstacles are placed via rejection sampling: each candidate must fit
    entirely inside the walkable polygon (buffered inward) and must leave
    at least ``min_passage_width`` clearance from walls and other obstacles
    so agents can navigate around them.

    Returns a list of obstacle polygons (to become polygon holes).
    """
    target_area = walkable.area * rng.uniform(*config.obstacle_coverage_range)
    placed: list[Polygon] = []
    placed_area = 0.0

    # Inset from walls: at least min_passage_width so agents can walk
    # between any obstacle and the wall.
    wall_margin = max(0.3, config.min_passage_width)
    inner = walkable.buffer(-wall_margin)
    if inner.is_empty or not isinstance(inner, Polygon):
        return []

    minx, miny, maxx, maxy = inner.bounds

    # Minimum gap between obstacles (buffered by this to prevent narrow gaps)
    obs_gap = config.min_passage_width

    for _ in range(max_attempts):
        if placed_area >= target_area:
            break

        if rng.random() < 0.4:
            # Column (circular approximation)
            r = rng.uniform(*config.column_radius_range)
            if maxx - minx < 2 * r or maxy - miny < 2 * r:
                continue
            cx = rng.uniform(minx + r, maxx - r)
            cy = rng.uniform(miny + r, maxy - r)
            obs = Point(cx, cy).buffer(r, quad_segs=config.column_quad_segs)
        else:
            # Rectangular obstacle
            w = rng.uniform(config.obstacle_min_size, config.obstacle_max_size)
            h = rng.uniform(config.obstacle_min_size, config.obstacle_max_size)
            if maxx - minx < w or maxy - miny < h:
                continue
            ox = rng.uniform(minx, maxx - w)
            oy = rng.uniform(miny, maxy - h)
            obs = box(ox, oy, ox + w, oy + h)
            # Random rotation
            angle = rng.uniform(0, 360)
            obs = rotate(obs, angle, origin="centroid")

        # Validate: must fit inside walkable (with wall margin)
        if not inner.contains(obs):
            continue
        # Must maintain min_passage_width gap from existing obstacles
        obs_buffered = obs.buffer(obs_gap)
        if any(obs_buffered.intersects(p) for p in placed):
            continue

        placed.append(obs)
        placed_area += obs.area

    return placed


def _cut_door_in_wall(
    rng: np.random.Generator,
    polygon: Polygon,
    wall_side: str,
    door_width: float,
) -> tuple[Polygon, Polygon]:
    """Cut a door opening in a wall of the polygon's bounding box.

    Returns (modified_polygon, door_region) where door_region is a small
    polygon on the exterior side of the door suitable as a goal/spawn area.

    wall_side: 'left', 'right', 'top', 'bottom'
    """
    minx, miny, maxx, maxy = polygon.bounds
    depth = 0.5  # depth of the door cut-out and goal region

    if wall_side == "left":
        pos = rng.uniform(miny + door_width / 2 + 0.2, maxy - door_width / 2 - 0.2)
        cut = box(minx - depth, pos - door_width / 2, minx + depth, pos + door_width / 2)
        door_region = box(minx - depth, pos - door_width / 2, minx + 0.1, pos + door_width / 2)
    elif wall_side == "right":
        pos = rng.uniform(miny + door_width / 2 + 0.2, maxy - door_width / 2 - 0.2)
        cut = box(maxx - depth, pos - door_width / 2, maxx + depth, pos + door_width / 2)
        door_region = box(maxx - 0.1, pos - door_width / 2, maxx + depth, pos + door_width / 2)
    elif wall_side == "bottom":
        pos = rng.uniform(minx + door_width / 2 + 0.2, maxx - door_width / 2 - 0.2)
        cut = box(pos - door_width / 2, miny - depth, pos + door_width / 2, miny + depth)
        door_region = box(pos - door_width / 2, miny - depth, pos + door_width / 2, miny + 0.1)
    else:  # top
        pos = rng.uniform(minx + door_width / 2 + 0.2, maxx - door_width / 2 - 0.2)
        cut = box(pos - door_width / 2, maxy - depth, pos + door_width / 2, maxy + depth)
        door_region = box(pos - door_width / 2, maxy - 0.1, pos + door_width / 2, maxy + depth)

    merged = unary_union([polygon, cut])
    merged = _ensure_valid_polygon(merged)
    return merged, door_region


# ---------------------------------------------------------------------------
# Tier 3a: Base room + obstacles + doors
# ---------------------------------------------------------------------------


def generate_tier3a(
    rng: np.random.Generator,
    config: GeometryConfig,
) -> GeneratedGeometry:
    """Generate a Tier 3a geometry: base room with internal obstacles and doors.

    1. Generate a base room from Tier 0–2
    2. Place random obstacles (columns, furniture blocks) inside
    3. Cut 1–2 door openings in the walls
    4. Optionally make all agents share one goal area (evacuation-like)
    """
    # Pick a base tier (0, 1, or 2) and generate the room
    base_tier = rng.choice([GeometryTier.TIER_0, GeometryTier.TIER_1, GeometryTier.TIER_2])
    base_config = GeometryConfig(
        tier=base_tier,
        min_side=config.min_side,
        max_side=config.max_side,
        corridor_width_range=config.corridor_width_range,
        corridor_length_range=config.corridor_length_range,
        bottleneck_aperture_range=config.bottleneck_aperture_range,
        bottleneck_depth_range=config.bottleneck_depth_range,
        branch_width_range=config.branch_width_range,
        branch_length_range=config.branch_length_range,
        min_passage_width=config.min_passage_width,
    )
    base = generate_geometry(rng, base_config)
    polygon = base.polygon

    # Place obstacles
    obstacles = _place_obstacles(rng, polygon, config)
    for obs in obstacles:
        result = polygon.difference(obs)
        result = _ensure_valid_polygon(result)
        if result.area > polygon.area * 0.3:  # don't let obstacles eat too much
            polygon = result

    # Cut 1–2 doors
    n_doors = rng.integers(1, 3)
    available_sides = ["left", "right", "top", "bottom"]
    rng.shuffle(available_sides)
    door_regions: list[Polygon] = []

    for i in range(min(n_doors, len(available_sides))):
        door_width = max(rng.uniform(*config.door_width_range), config.min_passage_width)
        side = available_sides[i]
        try:
            polygon, door_region = _cut_door_in_wall(rng, polygon, side, door_width)
            if not door_region.is_empty:
                door_regions.append(door_region)
        except (ValueError, IndexError):
            continue

    polygon = _ensure_valid_polygon(polygon)

    # Decide spawn and goal regions
    shared_goal = rng.random() < config.shared_goal_probability
    if shared_goal and door_regions:
        # All agents share one goal area (the first door = evacuation exit)
        goal_regions = [door_regions[0]]
        # Spawn from base room's spawn regions + any remaining doors
        spawn_regions = list(base.spawn_regions) + door_regions[1:]
        if not spawn_regions:
            spawn_regions = base.spawn_regions
    else:
        # Standard: spawn from base spawn regions, goals include base goals + doors
        spawn_regions = list(base.spawn_regions)
        goal_regions = list(base.goal_regions) + door_regions

    # Clip regions to the walkable polygon so they don't extend beyond
    # the boundary or overlap obstacle holes added after the base room.
    spawn_regions = _clip_regions(spawn_regions, polygon)
    goal_regions = _clip_regions(goal_regions, polygon)

    # Fallback: clip base regions too (obstacles may have changed the polygon)
    if not spawn_regions:
        spawn_regions = _clip_regions(base.spawn_regions, polygon)
    if not goal_regions:
        goal_regions = _clip_regions(base.goal_regions, polygon)

    n_obstacles = len(list(polygon.interiors)) - len(list(base.polygon.interiors))
    return GeneratedGeometry(
        polygon=polygon,
        spawn_regions=spawn_regions,
        goal_regions=goal_regions,
        tier=GeometryTier.TIER_3A,
        metadata={
            "shape": "room_with_obstacles",
            "base_tier": base_tier.value,
            "base_shape": base.metadata.get("shape", "unknown"),
            "n_obstacles": max(n_obstacles, 0),
            "n_doors": len(door_regions),
            "shared_goal": shared_goal and len(door_regions) > 0,
        },
    )


# ---------------------------------------------------------------------------
# Tier 3b: Composed multi-room layouts
# ---------------------------------------------------------------------------


def _generate_room(
    rng: np.random.Generator,
    config: GeometryConfig,
) -> GeneratedGeometry:
    """Generate a compact rectangular room for composition."""
    w = rng.uniform(config.min_side * 0.6, config.min_side)
    h = rng.uniform(config.min_side * 0.6, config.min_side)
    polygon = box(0, 0, w, h)
    margin = min(w, h) * 0.15
    spawn = box(margin, margin, w - margin, h - margin)
    return GeneratedGeometry(
        polygon=polygon,
        spawn_regions=[spawn],
        goal_regions=[spawn],
        tier=GeometryTier.TIER_0,
        metadata={"width": w, "height": h, "shape": "room"},
    )


def _generate_base_room(
    rng: np.random.Generator,
    config: GeometryConfig,
) -> GeneratedGeometry:
    """Generate a room from Tier 0–2 for use in Tier 3b composition."""
    base_tier = rng.choice([GeometryTier.TIER_0, GeometryTier.TIER_1, GeometryTier.TIER_2])
    base_config = GeometryConfig(
        tier=base_tier,
        min_side=config.min_side * 0.6,
        max_side=config.min_side,
        corridor_width_range=config.corridor_width_range,
        corridor_length_range=(
            config.corridor_length_range[0],
            config.corridor_length_range[0] * 1.5,
        ),
        bottleneck_aperture_range=config.bottleneck_aperture_range,
        bottleneck_depth_range=config.bottleneck_depth_range,
        branch_width_range=config.branch_width_range,
        branch_length_range=(config.branch_length_range[0], config.branch_length_range[0] * 1.5),
        min_passage_width=config.min_passage_width,
    )
    return generate_geometry(rng, base_config)


def generate_tier3b(
    rng: np.random.Generator,
    config: GeometryConfig,
) -> GeneratedGeometry:
    """Generate a Tier 3b geometry: 2–3 rooms connected by corridors + doors.

    1. Generate 2–3 base rooms (from Tier 0–2 primitives)
    2. Arrange them spatially (side by side or in an L-shape)
    3. Connect with corridor links (door-width passages)
    4. Place obstacles inside rooms
    5. Add 1–2 evacuation doors on the exterior
    """
    n_rooms = int(rng.integers(config.n_rooms_range[0], config.n_rooms_range[1] + 1))

    # Generate rooms
    rooms: list[GeneratedGeometry] = []
    for _ in range(n_rooms):
        rooms.append(_generate_base_room(rng, config))

    # Arrange rooms: place them side by side with a gap for connecting corridors
    gap = rng.uniform(1.5, 4.0)  # corridor length between rooms
    corridor_width = max(rng.uniform(1.5, 3.0), config.min_passage_width)
    placed_polygons: list[Polygon] = []
    placed_bounds: list[tuple[float, float, float, float]] = []
    connector_polygons: list[Polygon] = []
    connector_regions: list[Polygon] = []
    offset_x = 0.0

    for i, room in enumerate(rooms):
        # Translate room to its position
        poly = translate(room.polygon, xoff=offset_x, yoff=0)
        placed_polygons.append(poly)
        bminx, bminy, bmaxx, bmaxy = poly.bounds
        placed_bounds.append((bminx, bminy, bmaxx, bmaxy))

        if i > 0:
            # Connect to previous room with a corridor
            prev_poly = placed_polygons[i - 1]
            prev_bounds = placed_bounds[i - 1]
            # Corridor between right edge of prev and left edge of current
            prev_right = prev_bounds[2]
            curr_left = bminx

            # Find overlapping y-range for corridor placement
            y_overlap_min = max(prev_bounds[1], bminy) + 0.3
            y_overlap_max = min(prev_bounds[3], bmaxy) - 0.3

            # Use narrower corridor if y-overlap is tight, but still respect
            # min_passage_width as the absolute floor.
            effective_corridor_width = corridor_width
            if y_overlap_max - y_overlap_min < corridor_width:
                effective_corridor_width = max(
                    y_overlap_max - y_overlap_min - 0.1,
                    config.min_passage_width,
                )

            if y_overlap_max - y_overlap_min >= effective_corridor_width:
                ecw = effective_corridor_width
                # Try multiple placements to find one where the effective
                # opening into both rooms is wide enough.
                placed_connector = False
                for _try in range(10):
                    corr_y = rng.uniform(y_overlap_min, y_overlap_max - ecw)
                    # Extend corridor overlap into rooms to ensure connection
                    overlap_depth = 0.3
                    corridor = box(
                        prev_right - overlap_depth,
                        corr_y,
                        curr_left + overlap_depth,
                        corr_y + ecw,
                    )

                    # Validate effective opening: the intersection of the
                    # corridor with each room must be wide enough.  This
                    # catches cases where a convex room wall is angled and
                    # the actual opening is narrower than corridor_width.
                    prev_opening = corridor.intersection(prev_poly)
                    curr_opening = corridor.intersection(poly)
                    if prev_opening.is_empty or curr_opening.is_empty:
                        continue

                    # Measure opening height at the junction boundary
                    prev_junction = box(prev_right - 0.05, corr_y, prev_right + 0.05, corr_y + ecw)
                    curr_junction = box(curr_left - 0.05, corr_y, curr_left + 0.05, corr_y + ecw)
                    prev_junct_area = prev_junction.intersection(prev_poly)
                    curr_junct_area = curr_junction.intersection(poly)

                    # Effective width = height of the intersection at the junction
                    prev_eff_w = (
                        (prev_junct_area.bounds[3] - prev_junct_area.bounds[1])
                        if not prev_junct_area.is_empty
                        else 0.0
                    )
                    curr_eff_w = (
                        (curr_junct_area.bounds[3] - curr_junct_area.bounds[1])
                        if not curr_junct_area.is_empty
                        else 0.0
                    )

                    if (
                        prev_eff_w >= config.min_passage_width
                        and curr_eff_w >= config.min_passage_width
                    ):
                        connector_polygons.append(corridor)
                        margin = ecw * 0.15
                        conn_region = box(
                            prev_right + 0.1,
                            corr_y + margin,
                            curr_left - 0.1,
                            corr_y + ecw - margin,
                        )
                        if not conn_region.is_empty:
                            connector_regions.append(conn_region)
                        placed_connector = True
                        break

                if not placed_connector:
                    # Fallback: extend the corridor deeply into both rooms
                    # to guarantee geometric overlap even for convex shapes.
                    corr_y = (y_overlap_min + y_overlap_max) / 2 - ecw / 2
                    deep_overlap = min(
                        (prev_bounds[2] - prev_bounds[0]) * 0.3,
                        (bmaxx - bminx) * 0.3,
                        2.0,
                    )
                    corridor = box(
                        prev_right - deep_overlap,
                        corr_y,
                        curr_left + deep_overlap,
                        corr_y + ecw,
                    )
                    connector_polygons.append(corridor)

        # Advance offset for next room
        offset_x = bmaxx + gap

    # Merge everything into one walkable polygon
    all_parts = placed_polygons + connector_polygons
    merged = unary_union(all_parts)
    merged = _ensure_valid_polygon(merged)

    # Place obstacles inside the merged area
    obstacles = _place_obstacles(rng, merged, config)
    for obs in obstacles:
        result = merged.difference(obs)
        result = _ensure_valid_polygon(result)
        if result.area > merged.area * 0.3:
            merged = result

    # Add 1–2 evacuation doors on the exterior
    n_evac_doors = rng.integers(1, 3)
    available_sides = ["left", "right", "top", "bottom"]
    rng.shuffle(available_sides)
    evac_regions: list[Polygon] = []

    for i in range(min(n_evac_doors, len(available_sides))):
        door_width = max(rng.uniform(*config.door_width_range), config.min_passage_width)
        side = available_sides[i]
        try:
            merged, door_region = _cut_door_in_wall(rng, merged, side, door_width)
            if not door_region.is_empty:
                evac_regions.append(door_region)
        except (ValueError, IndexError):
            continue

    merged = _ensure_valid_polygon(merged)

    # Spawn regions: interiors of all rooms (translated), clipped to merged polygon
    raw_spawn: list[Polygon] = []
    for i, room in enumerate(rooms):
        for sr in room.spawn_regions:
            translated_sr = translate(sr, xoff=placed_bounds[i][0], yoff=placed_bounds[i][1])
            raw_spawn.append(translated_sr)

    # Goal regions: evacuation doors (primary) + connector regions
    raw_goal: list[Polygon] = list(evac_regions) + connector_regions

    # Clip all regions to the walkable polygon
    spawn_regions = _clip_regions(raw_spawn, merged)
    goal_regions = _clip_regions(raw_goal, merged)

    # Fallbacks (also clipped)
    if not spawn_regions:
        bnd = placed_bounds[0]
        margin = 0.3
        fallback = [box(bnd[0] + margin, bnd[1] + margin, bnd[2] - margin, bnd[3] - margin)]
        spawn_regions = _clip_regions(fallback, merged)
    if not goal_regions:
        bnd = placed_bounds[-1]
        margin = 0.3
        fallback = [box(bnd[0] + margin, bnd[1] + margin, bnd[2] - margin, bnd[3] - margin)]
        goal_regions = _clip_regions(fallback, merged)

    n_obstacles = len(list(merged.interiors))
    return GeneratedGeometry(
        polygon=merged,
        spawn_regions=spawn_regions,
        goal_regions=goal_regions,
        tier=GeometryTier.TIER_3B,
        metadata={
            "shape": "composed_rooms",
            "n_rooms": n_rooms,
            "n_obstacles": n_obstacles,
            "n_connectors": len(connector_polygons),
            "n_evac_doors": len(evac_regions),
            "room_shapes": [r.metadata.get("shape", "unknown") for r in rooms],
        },
    )


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
        GeometryTier.TIER_3A: generate_tier3a,
        GeometryTier.TIER_3B: generate_tier3b,
    }

    gen = generators[config.tier]
    max_attempts = 20
    for _ in range(max_attempts):
        geom = gen(rng, config)
        if geom.spawn_regions and geom.goal_regions:
            return geom
    raise RuntimeError(
        f"Failed to generate a {config.tier.name} geometry with usable "
        f"spawn and goal regions after {max_attempts} attempts"
    )
