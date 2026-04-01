"""Navigation mesh pathfinding: A* on triangle adjacency + funnel algorithm.

A* finds the sequence of triangles. The funnel algorithm (Simple Stupid Funnel)
then computes the true shortest path through the portal edges between those
triangles — producing a taut-string path identical to what JuPedSim computes.

Provides:
- next_waypoint_direction (2D) for the observation builder
- path_deviation scalar for the observation builder
- is_reachable for solvability verification
"""

from __future__ import annotations

import heapq

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import LineString, Point

from crowdrl_core.geometry import find_containing_triangle
from crowdrl_core.world_state import NavMesh


# ---------------------------------------------------------------------------
# A* on triangle adjacency graph
# ---------------------------------------------------------------------------


def _heuristic(a: NDArray, b: NDArray) -> float:
    """Euclidean distance heuristic for A*."""
    return float(np.linalg.norm(a - b))


def astar_triangle_path(
    navmesh: NavMesh,
    start_tri: int,
    goal_tri: int,
) -> list[int] | None:
    """A* shortest path on the triangle adjacency graph.

    Returns
    -------
    path : list[int] or None
        Sequence of triangle indices from start to goal (inclusive),
        or None if no path exists.
    """
    if start_tri == goal_tri:
        return [start_tri]

    centroids = navmesh.centroids
    goal_centroid = centroids[goal_tri]

    counter = 0
    open_set: list[tuple[float, int, int]] = []
    heapq.heappush(open_set, (0.0, counter, start_tri))

    came_from: dict[int, int] = {}
    g_score: dict[int, float] = {start_tri: 0.0}

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == goal_tri:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for neighbour in navmesh.adjacency[current]:
            edge_cost = float(np.linalg.norm(centroids[current] - centroids[neighbour]))
            tentative_g = g_score[current] + edge_cost

            if tentative_g < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g
                f = tentative_g + _heuristic(centroids[neighbour], goal_centroid)
                counter += 1
                heapq.heappush(open_set, (f, counter, neighbour))

    return None


# ---------------------------------------------------------------------------
# Funnel algorithm (Simple Stupid Funnel / Lee's algorithm)
# ---------------------------------------------------------------------------


def _cross2d(o: NDArray, a: NDArray, b: NDArray) -> float:
    """2D cross product of vectors (a - o) and (b - o).

    Positive → b is to the left of the line from o to a.
    """
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _extract_portals(
    navmesh: NavMesh,
    tri_path: list[int],
    start: NDArray[np.float64],
    goal: NDArray[np.float64],
    agent_radius: float = 0.0,
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Extract the sequence of portal edges from a triangle path.

    Returns a list of (left, right) endpoint pairs. The first portal is
    (start, start) and the last is (goal, goal) — degenerate portals that
    anchor the funnel at the endpoints.

    When *agent_radius* > 0, each portal is inset from both ends so that
    the resulting path keeps the agent centre at least *agent_radius* away
    from wall vertices. If a portal is too narrow, it collapses to the
    midpoint (the agent can still squeeze through, just centred).
    """
    portals = [(start.copy(), start.copy())]
    for i in range(len(tri_path) - 1):
        key = (tri_path[i], tri_path[i + 1])
        left, right = navmesh.portals[key]
        if agent_radius > 0:
            left, right = _inset_portal(left, right, agent_radius)
        portals.append((left, right))
    portals.append((goal.copy(), goal.copy()))
    return portals


def _inset_portal(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    radius: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Shrink a portal edge inward by *radius* from both endpoints.

    If the portal is shorter than 2 * radius, collapse both endpoints
    to the midpoint so the path stays centred in the gap.
    """
    direction = right - left
    length = float(np.linalg.norm(direction))
    if length < 1e-12:
        return left.copy(), right.copy()
    unit = direction / length
    if length <= 2.0 * radius:
        mid = (left + right) * 0.5
        return mid.copy(), mid.copy()
    return left + unit * radius, right - unit * radius


def funnel_path(
    navmesh: NavMesh,
    tri_path: list[int],
    start: NDArray[np.float64],
    goal: NDArray[np.float64],
    agent_radius: float = 0.0,
) -> list[NDArray[np.float64]]:
    """Compute the shortest path through a sequence of triangles using the funnel algorithm.

    This is the Simple Stupid Funnel Algorithm (Mikko Mononen).
    It produces a taut-string path — the geometrically shortest path that stays
    within the triangle corridor defined by A*.

    Parameters
    ----------
    navmesh : NavMesh
    tri_path : list[int]
        Triangle index sequence from A*.
    start : (2,) array
    goal : (2,) array

    Returns
    -------
    waypoints : list of (2,) arrays
        The shortest path from start to goal, including both endpoints.
    """
    if len(tri_path) <= 1:
        return [start.copy(), goal.copy()]

    portals = _extract_portals(navmesh, tri_path, start, goal, agent_radius)

    # Simple Stupid Funnel Algorithm
    path_points: list[NDArray[np.float64]] = [start.copy()]
    apex = start.copy()
    left_idx = 0
    right_idx = 0
    portal_left = portals[0][0].copy()
    portal_right = portals[0][1].copy()

    for i in range(1, len(portals)):
        new_left = portals[i][0]
        new_right = portals[i][1]

        # Update right vertex
        if _cross2d(apex, portal_right, new_right) >= 0:
            if np.allclose(apex, portal_right) or _cross2d(apex, portal_left, new_right) < 0:
                # Tighten the funnel
                portal_right = new_right.copy()
                right_idx = i
            else:
                # Right over left — insert left vertex, restart from there
                path_points.append(portal_left.copy())
                apex = portal_left.copy()
                left_idx = left_idx
                right_idx = left_idx
                portal_right = apex.copy()
                # Reset and restart scan from the new apex
                portal_left = apex.copy()
                # We can't actually restart the for-loop in Python, so we use
                # a while-loop implementation below instead.
                # For correctness, fall through — the while-loop version handles this.
                pass

        # Update left vertex
        if _cross2d(apex, portal_left, new_left) <= 0:
            if np.allclose(apex, portal_left) or _cross2d(apex, portal_right, new_left) > 0:
                portal_left = new_left.copy()
                left_idx = i
            else:
                path_points.append(portal_right.copy())
                apex = portal_right.copy()
                right_idx = right_idx
                left_idx = right_idx
                portal_left = apex.copy()
                portal_right = apex.copy()

    path_points.append(goal.copy())

    # Deduplicate consecutive identical points
    deduped = [path_points[0]]
    for p in path_points[1:]:
        if not np.allclose(p, deduped[-1], atol=1e-10):
            deduped.append(p)
    return deduped


def funnel_path_robust(
    navmesh: NavMesh,
    tri_path: list[int],
    start: NDArray[np.float64],
    goal: NDArray[np.float64],
    agent_radius: float = 0.0,
) -> list[NDArray[np.float64]]:
    """Compute the shortest path using the funnel algorithm (while-loop variant).

    This is the canonical implementation that correctly handles funnel restarts
    when one side of the funnel crosses the other.
    """
    if len(tri_path) <= 1:
        return [start.copy(), goal.copy()]

    portals = _extract_portals(navmesh, tri_path, start, goal, agent_radius)
    n_portals = len(portals)

    path_points: list[NDArray[np.float64]] = []
    apex_idx = 0
    left_idx = 0
    right_idx = 0
    apex = start.copy()
    portal_left = start.copy()
    portal_right = start.copy()

    path_points.append(apex.copy())

    i = 1
    while i < n_portals:
        new_left = portals[i][0]
        new_right = portals[i][1]

        # Try to narrow the funnel from the right side.
        # cross >= 0 means new_right is to the left of (or on) portal_right → tighter.
        if _cross2d(apex, portal_right, new_right) >= 0:
            if np.allclose(apex, portal_right) or _cross2d(apex, portal_left, new_right) < 0:
                # Still inside funnel — tighten right
                portal_right = new_right.copy()
                right_idx = i
            else:
                # Right crossed left — left vertex becomes new waypoint
                path_points.append(portal_left.copy())
                apex = portal_left.copy()
                apex_idx = left_idx
                portal_right = apex.copy()
                right_idx = apex_idx
                portal_left = apex.copy()
                left_idx = apex_idx
                i = apex_idx + 1
                continue

        # Try to narrow the funnel from the left side.
        # cross <= 0 means new_left is to the right of (or on) portal_left → tighter.
        if _cross2d(apex, portal_left, new_left) <= 0:
            if np.allclose(apex, portal_left) or _cross2d(apex, portal_right, new_left) > 0:
                # Still inside funnel — tighten left
                portal_left = new_left.copy()
                left_idx = i
            else:
                # Left crossed right — right vertex becomes new waypoint
                path_points.append(portal_right.copy())
                apex = portal_right.copy()
                apex_idx = right_idx
                portal_left = apex.copy()
                left_idx = apex_idx
                portal_right = apex.copy()
                right_idx = apex_idx
                i = apex_idx + 1
                continue

        i += 1

    # Add goal
    path_points.append(goal.copy())

    # Deduplicate consecutive identical points
    deduped = [path_points[0]]
    for p in path_points[1:]:
        if not np.allclose(p, deduped[-1], atol=1e-10):
            deduped.append(p)
    return deduped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_path(
    navmesh: NavMesh,
    start: NDArray[np.float64],
    goal: NDArray[np.float64],
) -> list[int] | None:
    """Find A* triangle path between two world-space points.

    Returns
    -------
    path : list[int] or None
        Triangle index path, or None if either point is outside the navmesh
        or no path exists.
    """
    start_tri = find_containing_triangle(start, navmesh)
    goal_tri = find_containing_triangle(goal, navmesh)
    if start_tri is None or goal_tri is None:
        return None
    return astar_triangle_path(navmesh, start_tri, goal_tri)


def shortest_path(
    navmesh: NavMesh,
    start: NDArray[np.float64],
    goal: NDArray[np.float64],
    agent_radius: float = 0.0,
) -> list[NDArray[np.float64]] | None:
    """Compute the true shortest path between two points through the navmesh.

    Uses A* to find the triangle corridor, then the funnel algorithm to
    compute the geometrically shortest (taut-string) path through the
    portal edges.

    Parameters
    ----------
    agent_radius : float
        Clearance distance from wall vertices. Portal edges are inset by
        this amount so the agent centre stays clear of geometry corners.

    Returns
    -------
    waypoints : list of (2,) arrays or None
        The shortest path including start and goal, or None if unreachable.
    """
    tri_path = find_path(navmesh, start, goal)
    if tri_path is None:
        return None
    return funnel_path_robust(navmesh, tri_path, start, goal, agent_radius)


def next_waypoint_direction(
    navmesh: NavMesh,
    position: NDArray[np.float64],
    goal: NDArray[np.float64],
    agent_radius: float = 0.0,
) -> NDArray[np.float64] | None:
    """Compute the unit direction toward the next waypoint on the shortest path.

    Uses the funnel-smoothed path so the direction points toward the
    actual next turning point — not a triangle centroid.

    Parameters
    ----------
    agent_radius : float
        Clearance distance from wall vertices (see :func:`shortest_path`).

    Returns
    -------
    direction : (2,) array or None
        Unit direction vector, or None if path cannot be computed.
    """
    waypoints = shortest_path(navmesh, position, goal, agent_radius)
    if waypoints is None:
        return None

    # waypoints[0] ≈ position, waypoints[1] is the next turning point
    if len(waypoints) < 2:
        return np.zeros(2, dtype=np.float64)

    diff = waypoints[1] - position
    dist = np.linalg.norm(diff)
    if dist < 1e-10:
        # At the waypoint — look toward the one after that
        if len(waypoints) >= 3:
            diff = waypoints[2] - position
            dist = np.linalg.norm(diff)
        if dist < 1e-10:
            return np.zeros(2, dtype=np.float64)
    return diff / dist


def path_deviation(
    navmesh: NavMesh,
    position: NDArray[np.float64],
    goal: NDArray[np.float64],
    agent_radius: float = 0.0,
) -> float | None:
    """Compute how much longer the shortest walkable path is vs. the straight line.

    Returns the ratio: (shortest_path_length / euclidean_distance) - 1.
    - 0.0 means the shortest path IS the straight line (no obstacles in the way).
    - >0 means obstacles force a detour.

    Parameters
    ----------
    agent_radius : float
        Clearance distance from wall vertices (see :func:`shortest_path`).

    Returns None if no path exists.
    """
    waypoints = shortest_path(navmesh, position, goal, agent_radius)
    if waypoints is None:
        return None

    euclidean = float(np.linalg.norm(goal - position))
    if euclidean < 1e-10:
        return 0.0

    path_length = sum(
        float(np.linalg.norm(waypoints[i + 1] - waypoints[i])) for i in range(len(waypoints) - 1)
    )

    return (path_length / euclidean) - 1.0


def is_reachable(
    navmesh: NavMesh,
    start: NDArray[np.float64],
    goal: NDArray[np.float64],
) -> bool:
    """Check if a path exists between start and goal on the navmesh.

    This is a pure topological check — it does NOT consider agent size.
    Use :func:`is_passable` for clearance-aware reachability.
    """
    return find_path(navmesh, start, goal) is not None


def _min_portal_width(
    navmesh: NavMesh,
    tri_path: list[int],
) -> float:
    """Return the minimum portal (shared-edge) width along a triangle path.

    Portal width is the Euclidean length of the shared edge between
    consecutive triangles — i.e. the physical gap the agent must fit through.
    """
    min_width = float("inf")
    for i in range(len(tri_path) - 1):
        key = (tri_path[i], tri_path[i + 1])
        left, right = navmesh.portals[key]
        width = float(np.linalg.norm(right - left))
        if width < min_width:
            min_width = width
    return min_width


def _validate_path_clearance(
    start: NDArray[np.float64],
    goal: NDArray[np.float64],
    radius: float,
    polygon,
) -> bool:
    """Check that an agent of *radius* can traverse from start to goal within the walkable area.

    Erodes the walkable polygon inward by *radius* (Minkowski difference)
    and checks that start and goal are connected in the eroded polygon.
    This is geometrically exact: if the eroded polygon connects start and
    goal, then a disc of *radius* can traverse between them without
    leaving the original polygon.

    This catches narrow gaps between close obstacles that portal-width
    checks miss (e.g. diagonal portal edges that overestimate the
    perpendicular clearance).
    """
    from shapely.geometry import MultiPolygon

    eroded = polygon.buffer(-radius)
    if eroded.is_empty:
        return False

    start_pt = Point(float(start[0]), float(start[1]))
    goal_pt = Point(float(goal[0]), float(goal[1]))

    if isinstance(eroded, MultiPolygon):
        # Multiple disconnected regions -- start and goal must be in the same one
        for part in eroded.geoms:
            # Use distance < radius to handle points near the eroded boundary
            # (the original point is inside the full polygon, but may be
            # outside the eroded polygon by up to radius)
            if part.distance(start_pt) < radius and part.distance(goal_pt) < radius:
                return True
        return False

    # Single polygon -- both points must be within reach
    return eroded.distance(start_pt) < radius and eroded.distance(goal_pt) < radius


def is_passable(
    navmesh: NavMesh,
    start: NDArray[np.float64],
    goal: NDArray[np.float64],
    agent_radius: float = 0.0,
    clearance_factor: float = 1.0,
) -> bool:
    """Check if an agent of the given radius can traverse from start to goal.

    Three-stage check:

    1. **A* reachability** -- topological path exists on the triangle graph.
    2. **Portal-width filter** -- every shared edge along the corridor is at
       least ``2 * effective_radius`` wide.  This is a fast rejection filter.
    3. **Geometric clearance** -- the funnel-smoothed path, buffered by
       *effective_radius*, lies entirely within the walkable polygon.  This
       catches narrow gaps between close obstacles where the portal edge
       runs diagonally and overestimates the actual perpendicular clearance.

    ``effective_radius = agent_radius * clearance_factor``.  Use
    *clearance_factor* > 1 to add a safety margin (e.g. 1.2 = 20%,
    ensuring the agent at its widest orientation can traverse the path).

    Parameters
    ----------
    navmesh : NavMesh
    start, goal : (2,) arrays
    agent_radius : float
        Half-width of the agent (e.g. max of shoulder_width, chest_depth)).
        If 0, this is equivalent to :func:`is_reachable`.
    clearance_factor : float
        Multiplier applied to *agent_radius* for all clearance checks.
        Default 1.0 (no extra margin).
    """
    tri_path = find_path(navmesh, start, goal)
    if tri_path is None:
        return False

    effective_radius = agent_radius * clearance_factor

    if effective_radius <= 0 or len(tri_path) <= 1:
        return True

    # Stage 2: quick portal-width rejection
    if _min_portal_width(navmesh, tri_path) < 2.0 * effective_radius:
        return False

    # Stage 3: geometric clearance via Minkowski erosion
    if navmesh.polygon is not None:
        if not _validate_path_clearance(start, goal, effective_radius, navmesh.polygon):
            return False

    return True
