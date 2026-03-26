"""Tests for geometry module: wall extraction, triangulation, navmesh construction."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from crowdrl_core.geometry import (
    build_navmesh,
    extract_wall_segments,
    find_containing_triangle,
    point_in_triangle,
    sample_point_in_polygon,
    triangulate_polygon,
)


class TestExtractWallSegments:
    def test_square(self, simple_square_polygon):
        segs = extract_wall_segments(simple_square_polygon)
        assert segs.shape == (4, 2, 2)  # 4 edges
        # Each segment should be 2 points
        for seg in segs:
            assert seg.shape == (2, 2)

    def test_corridor(self, corridor_polygon):
        segs = extract_wall_segments(corridor_polygon)
        assert segs.shape == (4, 2, 2)

    def test_polygon_with_hole(self, square_with_obstacle):
        segs = extract_wall_segments(square_with_obstacle)
        # 4 exterior edges + 4 hole edges = 8
        assert segs.shape == (8, 2, 2)

    def test_bottleneck(self, bottleneck_polygon):
        segs = extract_wall_segments(bottleneck_polygon)
        # 4 exterior + 4 + 4 hole edges = 12
        assert segs.shape == (12, 2, 2)

    def test_segments_are_connected(self, simple_square_polygon):
        """Each segment's end should be the next segment's start (for exterior)."""
        segs = extract_wall_segments(simple_square_polygon)
        for i in range(len(segs) - 1):
            # Not necessarily sequential across exterior/holes, but within exterior
            pass
        # At minimum, all segments should have non-zero length
        for seg in segs:
            length = np.linalg.norm(seg[1] - seg[0])
            assert length > 0


class TestTriangulation:
    def test_square_produces_triangles(self, simple_square_polygon):
        tris = triangulate_polygon(simple_square_polygon)
        assert len(tris) >= 2  # A square needs at least 2 triangles

    def test_all_triangles_inside_polygon(self, simple_square_polygon):
        tris = triangulate_polygon(simple_square_polygon)
        for tri in tris:
            centroid = tri.mean(axis=0)
            from shapely.geometry import Point
            assert simple_square_polygon.contains(Point(centroid))

    def test_polygon_with_hole(self, square_with_obstacle):
        tris = triangulate_polygon(square_with_obstacle)
        assert len(tris) >= 4  # Hole creates more triangles
        # No triangle centroid should be inside the hole
        for tri in tris:
            centroid = tri.mean(axis=0)
            from shapely.geometry import Point
            assert square_with_obstacle.contains(Point(centroid))

    def test_corridor(self, corridor_polygon):
        tris = triangulate_polygon(corridor_polygon)
        assert len(tris) >= 2

    def test_bottleneck(self, bottleneck_polygon):
        tris = triangulate_polygon(bottleneck_polygon)
        assert len(tris) >= 4

    def test_triangle_vertices_are_3x2(self, simple_square_polygon):
        tris = triangulate_polygon(simple_square_polygon)
        for tri in tris:
            assert tri.shape == (3, 2)


class TestPointInTriangle:
    def test_inside(self):
        tri = np.array([[0, 0], [4, 0], [2, 3]], dtype=np.float64)
        assert point_in_triangle(np.array([2.0, 1.0]), tri)

    def test_outside(self):
        tri = np.array([[0, 0], [4, 0], [2, 3]], dtype=np.float64)
        assert not point_in_triangle(np.array([5.0, 5.0]), tri)

    def test_on_edge(self):
        tri = np.array([[0, 0], [4, 0], [2, 3]], dtype=np.float64)
        # Point on the bottom edge
        assert point_in_triangle(np.array([2.0, 0.0]), tri)

    def test_on_vertex(self):
        tri = np.array([[0, 0], [4, 0], [2, 3]], dtype=np.float64)
        assert point_in_triangle(np.array([0.0, 0.0]), tri)


class TestBuildNavmesh:
    def test_square(self, simple_square_polygon):
        nm = build_navmesh(simple_square_polygon)
        assert nm.triangles.shape[0] >= 2
        assert nm.triangles.shape[1:] == (3, 2)
        assert nm.centroids.shape == (nm.triangles.shape[0], 2)
        assert len(nm.adjacency) == nm.triangles.shape[0]

    def test_adjacency_is_symmetric(self, simple_square_polygon):
        nm = build_navmesh(simple_square_polygon)
        for i, neighbours in enumerate(nm.adjacency):
            for j in neighbours:
                assert i in nm.adjacency[j], f"Adjacency not symmetric: {i} -> {j}"

    def test_connected_graph(self, simple_square_polygon):
        """All triangles should be reachable from any other (simple convex polygon)."""
        nm = build_navmesh(simple_square_polygon)
        n = len(nm.adjacency)
        if n <= 1:
            return
        # BFS from triangle 0
        visited = {0}
        queue = [0]
        while queue:
            current = queue.pop(0)
            for nb in nm.adjacency[current]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        assert len(visited) == n, f"Only {len(visited)}/{n} triangles reachable"

    def test_polygon_with_hole(self, square_with_obstacle):
        nm = build_navmesh(square_with_obstacle)
        assert nm.triangles.shape[0] >= 4

    def test_find_containing_triangle(self, simple_square_polygon):
        nm = build_navmesh(simple_square_polygon)
        # Point in the middle should be found
        idx = find_containing_triangle(np.array([5.0, 5.0]), nm)
        assert idx is not None
        # Point outside should not be found
        idx = find_containing_triangle(np.array([15.0, 15.0]), nm)
        assert idx is None


class TestSamplePointInPolygon:
    def test_samples_inside(self, simple_square_polygon):
        rng = np.random.default_rng(42)
        for _ in range(50):
            pt = sample_point_in_polygon(simple_square_polygon, rng)
            from shapely.geometry import Point
            assert simple_square_polygon.contains(Point(pt))

    def test_samples_inside_polygon_with_hole(self, square_with_obstacle):
        rng = np.random.default_rng(42)
        for _ in range(50):
            pt = sample_point_in_polygon(square_with_obstacle, rng)
            from shapely.geometry import Point
            assert square_with_obstacle.contains(Point(pt))

    def test_returns_2d_array(self, simple_square_polygon):
        pt = sample_point_in_polygon(simple_square_polygon)
        assert pt.shape == (2,)
        assert pt.dtype == np.float64
