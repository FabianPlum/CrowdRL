"""Tests for the procedural geometry generator (Tiers 0–2)."""

import numpy as np
from shapely.geometry import Polygon

from crowdrl_env.geometry_generator import (
    GeneratedGeometry,
    GeometryConfig,
    GeometryTier,
    generate_bottleneck,
    generate_convex_polygon,
    generate_corridor,
    generate_crossroads,
    generate_geometry,
    generate_l_bend,
    generate_rectangle,
    generate_t_junction,
    generate_tier0,
    generate_tier1,
    generate_tier2,
)


def _validate_generated(geom: GeneratedGeometry):
    """Common validation for any generated geometry."""
    assert isinstance(geom.polygon, Polygon)
    assert geom.polygon.is_valid, f"Invalid polygon: {geom.polygon}"
    assert geom.polygon.area > 0
    assert not geom.polygon.is_empty

    # Spawn regions should overlap with the walkable polygon
    for sr in geom.spawn_regions:
        assert isinstance(sr, Polygon)
        assert not sr.is_empty
        assert geom.polygon.intersects(sr), "Spawn region doesn't overlap walkable area"

    # Goal regions should overlap with the walkable polygon
    for gr in geom.goal_regions:
        assert isinstance(gr, Polygon)
        assert not gr.is_empty
        assert geom.polygon.intersects(gr), "Goal region doesn't overlap walkable area"


class TestTier0:
    def test_rectangle(self, rng):
        config = GeometryConfig(tier=GeometryTier.TIER_0)
        geom = generate_rectangle(rng, config)
        _validate_generated(geom)
        assert geom.tier == GeometryTier.TIER_0
        assert geom.metadata["shape"] == "rectangle"
        # Should have no holes
        assert len(list(geom.polygon.interiors)) == 0

    def test_convex_polygon(self, rng):
        config = GeometryConfig(tier=GeometryTier.TIER_0)
        geom = generate_convex_polygon(rng, config)
        _validate_generated(geom)
        assert geom.tier == GeometryTier.TIER_0
        assert geom.metadata["shape"] == "convex"
        assert len(list(geom.polygon.interiors)) == 0

    def test_tier0_random(self, rng):
        config = GeometryConfig(tier=GeometryTier.TIER_0)
        for _ in range(20):
            geom = generate_tier0(rng, config)
            _validate_generated(geom)
            assert geom.tier == GeometryTier.TIER_0

    def test_size_within_bounds(self, rng):
        config = GeometryConfig(min_side=5.0, max_side=15.0)
        for _ in range(10):
            geom = generate_rectangle(rng, config)
            minx, miny, maxx, maxy = geom.polygon.bounds
            w, h = maxx - minx, maxy - miny
            assert 5.0 <= w <= 15.0
            assert 5.0 <= h <= 15.0


class TestTier1:
    def test_corridor(self, rng):
        config = GeometryConfig(tier=GeometryTier.TIER_1)
        geom = generate_corridor(rng, config)
        _validate_generated(geom)
        assert geom.tier == GeometryTier.TIER_1
        assert geom.metadata["shape"] == "corridor"

    def test_bottleneck(self, rng):
        config = GeometryConfig(tier=GeometryTier.TIER_1)
        geom = generate_bottleneck(rng, config)
        _validate_generated(geom)
        assert geom.tier == GeometryTier.TIER_1
        assert geom.metadata["shape"] == "bottleneck"
        assert "aperture" in geom.metadata
        assert geom.metadata["aperture"] > 0

    def test_bottleneck_aperture_within_bounds(self, rng):
        config = GeometryConfig(
            tier=GeometryTier.TIER_1,
            bottleneck_aperture_range=(0.6, 2.0),
        )
        for _ in range(20):
            geom = generate_bottleneck(rng, config)
            assert 0.0 < geom.metadata["aperture"] <= 2.0

    def test_corridor_width_within_bounds(self, rng):
        config = GeometryConfig(
            tier=GeometryTier.TIER_1,
            corridor_width_range=(1.0, 3.0),
        )
        for _ in range(10):
            geom = generate_corridor(rng, config)
            assert 1.0 <= geom.metadata["width"] <= 3.0

    def test_tier1_random(self, rng):
        config = GeometryConfig(tier=GeometryTier.TIER_1)
        for _ in range(20):
            geom = generate_tier1(rng, config)
            _validate_generated(geom)
            assert geom.tier == GeometryTier.TIER_1


class TestTier2:
    def test_l_bend(self, rng):
        config = GeometryConfig(tier=GeometryTier.TIER_2)
        geom = generate_l_bend(rng, config)
        _validate_generated(geom)
        assert geom.tier == GeometryTier.TIER_2
        assert geom.metadata["shape"] == "l_bend"

    def test_t_junction(self, rng):
        config = GeometryConfig(tier=GeometryTier.TIER_2)
        geom = generate_t_junction(rng, config)
        _validate_generated(geom)
        assert geom.tier == GeometryTier.TIER_2
        assert geom.metadata["shape"] == "t_junction"
        # T-junction should have at least 2 goal regions
        assert len(geom.goal_regions) >= 2

    def test_crossroads(self, rng):
        config = GeometryConfig(tier=GeometryTier.TIER_2)
        geom = generate_crossroads(rng, config)
        _validate_generated(geom)
        assert geom.tier == GeometryTier.TIER_2
        assert geom.metadata["shape"] == "crossroads"

    def test_tier2_random(self, rng):
        config = GeometryConfig(tier=GeometryTier.TIER_2)
        for _ in range(30):
            geom = generate_tier2(rng, config)
            _validate_generated(geom)
            assert geom.tier == GeometryTier.TIER_2


class TestGenerateGeometry:
    def test_default_is_tier0(self):
        geom = generate_geometry()
        assert geom.tier == GeometryTier.TIER_0

    def test_each_tier(self, rng):
        for tier in GeometryTier:
            config = GeometryConfig(tier=tier)
            geom = generate_geometry(rng, config)
            assert geom.tier == tier
            _validate_generated(geom)

    def test_reproducible_with_seed(self):
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        config = GeometryConfig(tier=GeometryTier.TIER_1)
        g1 = generate_geometry(rng1, config)
        g2 = generate_geometry(rng2, config)
        assert g1.polygon.equals(g2.polygon)

    def test_different_seeds_different_results(self):
        config = GeometryConfig(tier=GeometryTier.TIER_0)
        g1 = generate_geometry(np.random.default_rng(1), config)
        g2 = generate_geometry(np.random.default_rng(2), config)
        assert not g1.polygon.equals(g2.polygon)
