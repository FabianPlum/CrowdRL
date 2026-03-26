"""Geometry utilities: polygon handling, triangulation, navmesh construction, wall extraction.

All geometries are Shapely Polygons with holes, matching JuPedSim convention.
Walkable area = polygon exterior; obstacles = polygon holes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import triangulate as shapely_triangulate

from crowdrl_core.world_state import NavMesh


def extract_wall_segments(polygon: Polygon) -> NDArray[np.float64]:
    """Extract all wall segments from a Shapely Polygon (exterior + holes).

    Returns
    -------
    segments : (S, 2, 2) array
        Each segment is [[x1, y1], [x2, y2]].
    """
    segments = []

    # Exterior boundary
    coords = np.array(polygon.exterior.coords)
    for i in range(len(coords) - 1):
        segments.append([coords[i], coords[i + 1]])

    # Hole boundaries (obstacles)
    for hole in polygon.interiors:
        coords = np.array(hole.coords)
        for i in range(len(coords) - 1):
            segments.append([coords[i], coords[i + 1]])

    return np.array(segments, dtype=np.float64)


def _polygon_vertices(polygon: Polygon) -> NDArray[np.float64]:
    """Collect all unique vertices from exterior + holes."""
    coords = list(polygon.exterior.coords[:-1])  # drop closing duplicate
    for hole in polygon.interiors:
        coords.extend(hole.coords[:-1])
    return np.array(coords, dtype=np.float64)


def triangulate_polygon(polygon: Polygon) -> list[NDArray[np.float64]]:
    """Constrained Delaunay triangulation of a polygon with holes.

    Uses Shapely's Delaunay triangulation on polygon vertices, then filters
    to keep only triangles whose centroid lies inside the walkable area.

    Returns
    -------
    triangles : list of (3, 2) arrays
        Vertex coordinates for each valid triangle.
    """
    vertices = _polygon_vertices(polygon)
    if len(vertices) < 3:
        return []

    # Use Shapely's triangulate which handles the constraint edges better
    # for polygons with holes than raw scipy Delaunay
    points = MultiPoint(vertices.tolist())
    raw_triangles = shapely_triangulate(points)

    valid = []
    for tri in raw_triangles:
        if tri.geom_type != "Polygon":
            continue
        centroid = tri.centroid
        if polygon.contains(centroid):
            coords = np.array(tri.exterior.coords[:3], dtype=np.float64)
            valid.append(coords)

    return valid


def _triangles_share_edge(t1: NDArray, t2: NDArray, tol: float = 1e-10) -> bool:
    """Check if two triangles share an edge (two common vertices)."""
    shared = 0
    for v1 in t1:
        for v2 in t2:
            if np.linalg.norm(v1 - v2) < tol:
                shared += 1
                if shared >= 2:
                    return True
    return False


def _orient_portal(
    edge_a: NDArray[np.float64],
    edge_b: NDArray[np.float64],
    from_centroid: NDArray[np.float64],
    to_centroid: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Orient a shared edge as (left, right) relative to the travel direction.

    Standing at `from_centroid` and looking toward `to_centroid`, the portal's
    'left' endpoint is to the left and 'right' is to the right.
    """
    travel = to_centroid - from_centroid
    # Cross product: travel × (edge_a - from_centroid)
    # Positive cross → edge_a is to the left
    to_a = edge_a - from_centroid
    cross = travel[0] * to_a[1] - travel[1] * to_a[0]
    if cross >= 0:
        return edge_a.copy(), edge_b.copy()  # a is left, b is right
    return edge_b.copy(), edge_a.copy()  # b is left, a is right


def build_navmesh(polygon: Polygon) -> NavMesh:
    """Build a navigation mesh from a walkable polygon.

    1. Triangulate the polygon (constrained Delaunay).
    2. Compute triangle centroids.
    3. Build adjacency graph (triangles sharing an edge are neighbours).
    4. Compute portal edges (shared edges between adjacent triangles).

    The portals enable the funnel algorithm for true shortest-path computation.

    Returns
    -------
    NavMesh
        Ready for A* pathfinding + funnel-smoothed shortest paths.
    """
    tri_list = triangulate_polygon(polygon)
    if not tri_list:
        raise ValueError("Polygon produced no valid triangles during triangulation")

    triangles = np.array(tri_list, dtype=np.float64)  # (T, 3, 2)
    centroids = triangles.mean(axis=1)  # (T, 2)

    n_tris = len(triangles)

    # Build vertex → triangle index map for efficient adjacency detection
    # Round vertices to avoid floating-point mismatches
    vertex_to_tris: dict[tuple[float, float], list[int]] = {}
    for i, tri in enumerate(triangles):
        for v in tri:
            key = (round(v[0], 8), round(v[1], 8))
            vertex_to_tris.setdefault(key, []).append(i)

    # Two triangles are adjacent if they share exactly 2 vertices
    adjacency: list[list[int]] = [[] for _ in range(n_tris)]
    portals: dict[tuple[int, int], tuple[NDArray[np.float64], NDArray[np.float64]]] = {}

    for i in range(n_tris):
        verts_i_keys = {(round(v[0], 8), round(v[1], 8)) for v in triangles[i]}
        # Candidate neighbours: triangles that share at least one vertex
        candidates = set()
        for v in verts_i_keys:
            for j in vertex_to_tris.get(v, []):
                if j > i:
                    candidates.add(j)
        for j in candidates:
            verts_j_keys = {(round(v[0], 8), round(v[1], 8)) for v in triangles[j]}
            shared_keys = verts_i_keys & verts_j_keys
            if len(shared_keys) >= 2:
                adjacency[i].append(j)
                adjacency[j].append(i)

                # Extract the actual shared vertex coordinates from triangle i
                shared_verts = []
                for v in triangles[i]:
                    key = (round(v[0], 8), round(v[1], 8))
                    if key in shared_keys:
                        shared_verts.append(v.copy())
                edge_a, edge_b = shared_verts[0], shared_verts[1]

                # Orient portal for both travel directions
                left_ij, right_ij = _orient_portal(
                    edge_a, edge_b, centroids[i], centroids[j]
                )
                portals[(i, j)] = (left_ij, right_ij)

                left_ji, right_ji = _orient_portal(
                    edge_a, edge_b, centroids[j], centroids[i]
                )
                portals[(j, i)] = (left_ji, right_ji)

    return NavMesh(
        triangles=triangles,
        centroids=centroids,
        adjacency=adjacency,
        portals=portals,
    )


def point_in_triangle(point: NDArray, triangle: NDArray) -> bool:
    """Test if a 2D point lies inside a triangle using barycentric coordinates."""
    v0 = triangle[2] - triangle[0]
    v1 = triangle[1] - triangle[0]
    v2 = point - triangle[0]

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return False

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (u >= -1e-10) and (v >= -1e-10) and (u + v <= 1.0 + 1e-10)


def find_containing_triangle(
    point: NDArray[np.float64], navmesh: NavMesh
) -> int | None:
    """Find which triangle in the navmesh contains the given point.

    Returns triangle index, or None if point is outside all triangles.
    """
    for i, tri in enumerate(navmesh.triangles):
        if point_in_triangle(point, tri):
            return i
    return None


def sample_point_in_polygon(
    polygon: Polygon, rng: np.random.Generator | None = None
) -> NDArray[np.float64]:
    """Uniformly sample a random point inside a polygon (rejection sampling)."""
    if rng is None:
        rng = np.random.default_rng()

    minx, miny, maxx, maxy = polygon.bounds
    while True:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if polygon.contains(p):
            return np.array([x, y], dtype=np.float64)
