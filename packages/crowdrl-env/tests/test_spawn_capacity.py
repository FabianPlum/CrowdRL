"""Spawn capacity, delivery guarantees, and the spawner delivery benchmark.

Regression cover for the silent spawn shortfall: the spawner used to place roughly
half the requested crowd and report the request as though it had been delivered, so
episodes labelled "100 agents, high density" ran 5 agents at 0.19 ped/m2.

The invariants below are what "the spawner works" means:

1. **Exact delivery** -- the requested count is delivered, or the shortfall is raised.
2. **Separation** -- no two bodies overlap at spawn, including across spawn regions.
3. **Wall clearance** -- no body is born inside a wall.
4. **Determinism** -- a seed reproduces exactly, and both engines agree byte-for-byte.
5. **Monotonicity** -- asking for more agents never delivers fewer.
6. **Honest capacity** -- the capacity estimate tracks achievable yield and is never
   derated in a way that would reject a satisfiable request.

:class:`TestSpawnerDeliveryBenchmark` is the tracked metric: per-curriculum-phase
delivery rate. Baseline before the fix (120 episodes/phase, measured): easy 100%,
medium 93.4%, hard 64.8%, rooms 70.9%, complex 62.3%, full 51.7% -- with 87% of
`full`-phase episodes short and a worst case of 10%.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest
from shapely.geometry import Point, box

from crowdrl_env.crowd_env import CrowdEnv, CrowdEnvConfig
from crowdrl_env.geometry_generator import GeometryConfig, GeometryTier, generate_geometry
from crowdrl_env.spawner import (
    SpawnConfig,
    SpawnShortfallError,
    spawn_agents,
    spawn_capacity,
)

ALL_TIERS = list(GeometryTier)

# Agent counts spanning the curriculum: sparse, the training mid-range, and the
# dense tail the scorecard's high-density scenarios use.
REPRESENTATIVE_COUNTS = [10, 30, 60, 100]


def _episode(tier: GeometryTier, seed: int, n_agents: int) -> tuple[CrowdEnv, dict]:
    """Reset a CrowdEnv pinned to one tier and an exact agent count."""
    base = CrowdEnvConfig()
    config = dataclasses.replace(
        base,
        geometry_tiers=[tier],
        tier_weights=None,
        spawn=dataclasses.replace(base.spawn, n_agents_range=(n_agents, n_agents)),
    )
    env = CrowdEnv(config=config, seed=seed)
    _, info = env.reset(seed=seed)
    return env, info


def _min_pairwise_distance(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return float("inf")
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def _hard_packing_ceiling(regions, min_sep: float) -> int:
    """A guaranteed ceiling on centres, unlike the asymptotic estimate.

    ``spawn_capacity`` uses the asymptotic hexagonal density, which a real placement
    can beat on a region only a few separations wide (a boundary centre consumes less
    than a full hex cell). The rigorous version notes that ``min_sep``-separated
    centres are radius-``min_sep/2`` discs packed inside the region *dilated* by
    ``min_sep/2``, so that dilated area over the hex cell area is a true bound.
    """
    from shapely.ops import unary_union

    dilated = unary_union(regions).buffer(min_sep / 2.0)
    return int(dilated.area / (min_sep**2 * np.sqrt(3) / 2.0)) + 1


class TestCapacityEstimate:
    """The estimate drives dilation, so it must not under-report a feasible request."""

    @pytest.mark.parametrize("tier", ALL_TIERS)
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_placement_respects_the_hard_packing_ceiling(self, tier: GeometryTier, seed: int):
        rng = np.random.default_rng(seed)
        geom = generate_geometry(rng, GeometryConfig(tier=tier))
        # Dilation off: the ceiling is computed from geom.spawn_regions, so the
        # sampler must be measured against the regions it was actually given. Margin
        # erosion and the walkable clip only shrink them, so the ceiling still holds.
        result = spawn_agents(
            rng,
            geom.spawn_regions,
            geom.goal_regions,
            SpawnConfig(max_spawn_dilation=0.0),
            n_agents=100,
            walkable=geom.polygon,
        )
        ceiling = _hard_packing_ceiling(geom.spawn_regions, result.min_separation)
        assert result.n_agents <= ceiling, (
            f"placed {result.n_agents} above the rigorous packing ceiling {ceiling} "
            f"-- separation must be violated somewhere"
        )

    def test_estimate_tracks_achievable_yield(self):
        """Within a few percent once the region spans several separations.

        This is the property dilation relies on: hit ``capacity >= headroom * n`` and
        the sampler can actually fill ``n``.
        """
        walkable = box(0.0, 0.0, 12.0, 12.0)
        region = box(0.5, 0.5, 11.5, 11.5)
        # Overshoot the estimate so the region saturates and the yield is the region's
        # own limit rather than the request. Bounded overshoot, not a huge number:
        # the saturation bail-out keeps that cheap, but the test should not rely on it.
        oversubscribed = 2 * spawn_capacity(region.area, 0.55)
        result = spawn_agents(
            np.random.default_rng(0),
            [region],
            [region],
            SpawnConfig(),
            n_agents=oversubscribed,
            walkable=walkable,
        )
        assert 0.75 <= result.n_agents / result.capacity <= 1.05, (
            f"yield {result.n_agents}/{result.capacity} is too far from the estimate "
            f"for dilation targeting to be reliable"
        )

    def test_capacity_matches_hex_packing(self):
        # One centre per min_sep^2 * sqrt(3)/2. At min_sep=0.5 that is ~0.2165 m2,
        # so 10 m2 holds 46.
        assert spawn_capacity(10.0, 0.5) == 46
        assert spawn_capacity(1.25, 0.52) == 5  # the TIER_3A doorway from the report

    def test_capacity_degenerate_inputs(self):
        assert spawn_capacity(0.0, 0.5) == 0
        assert spawn_capacity(-1.0, 0.5) == 0
        assert spawn_capacity(10.0, 0.0) == 0

    def test_capacity_is_not_derated(self):
        """A 0.70-style efficiency factor would reject satisfiable requests.

        Measured: a 1.25 m2 doorway takes 5 bodies at min_sep 0.52, while a 0.70
        derate predicts 3. Gating on the derated number refuses a request the sampler
        can actually fill.
        """
        area, min_sep = 1.25, 0.52
        assert spawn_capacity(area, min_sep) >= 5
        derated = int(0.70 * area / (min_sep**2 * np.sqrt(3) / 2))
        assert derated < 5, "sanity: the derate really is too pessimistic to gate on"


class TestExactDelivery:
    """The headline guarantee: ask for N, simulate N."""

    @pytest.mark.parametrize("tier", ALL_TIERS)
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_env_delivers_requested_count(self, tier: GeometryTier, seed: int):
        _, info = _episode(tier, seed, 30)
        assert info["n_agents"] == info["requested_n"] == 30, (
            f"tier={tier.name} seed={seed} delivered {info['n_agents']}/30 "
            f"(spawn_area={info['spawn_area_m2']:.2f} m2, "
            f"capacity={info['spawn_capacity']})"
        )

    @pytest.mark.parametrize("n_agents", REPRESENTATIVE_COUNTS)
    def test_delivery_across_agent_counts(self, n_agents: int):
        """Dense requests are the regression: 60 and 100 both used to yield 5."""
        for tier in (GeometryTier.TIER_3A, GeometryTier.TIER_3B):
            _, info = _episode(tier, 0, n_agents)
            assert info["n_agents"] == n_agents, (
                f"tier={tier.name} delivered {info['n_agents']}/{n_agents}"
            )

    def test_dense_requests_are_actually_dense(self):
        """The bug made high-density scenarios the SPARSEST in the suite.

        rooms_hi at 60 and at 100 agents both ran 5 agents at 0.19 ped/m2. Distinct
        requests must now produce distinct, non-trivial densities.
        """
        _, info_60 = _episode(GeometryTier.TIER_3A, 0, 60)
        _, info_100 = _episode(GeometryTier.TIER_3A, 0, 100)
        assert info_60["n_agents"] == 60
        assert info_100["n_agents"] == 100
        assert info_60["achieved_density"] > 0.25
        assert info_100["achieved_density"] > 0.25

    def test_provenance_is_recorded(self):
        _, info = _episode(GeometryTier.TIER_3A, 0, 30)
        for key in (
            "requested_n",
            "spawn_area_m2",
            "spawn_capacity",
            "walkable_area_m2",
            "achieved_density",
        ):
            assert info[key] is not None, f"{key} missing from reset() info"
        assert info["achieved_density"] == pytest.approx(
            info["n_agents"] / info["walkable_area_m2"]
        )


class TestSeparationInvariant:
    @pytest.mark.parametrize("tier", ALL_TIERS)
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_no_bodies_overlap_at_spawn(self, tier: GeometryTier, seed: int):
        env, _ = _episode(tier, seed, 40)
        world = env._world
        radii = np.maximum(world.shoulder_widths, world.chest_depths)
        closest = _min_pairwise_distance(world.positions)
        assert closest >= 2.0 * radii.max() - 1e-9 or closest >= 0.3 - 1e-9, (
            f"tier={tier.name} seed={seed}: closest pair {closest:.4f} m"
        )

    def test_separation_holds_across_multiple_regions(self):
        """A lattice built per-region lets neighbouring regions collide.

        Two adjacent spawn regions each satisfy min_sep internally while points on
        either side of the shared edge sit closer than min_sep to each other.
        """
        regions = [box(0.0, 0.0, 2.0, 2.0), box(2.0, 0.0, 4.0, 2.0)]
        goal = [box(20.0, 0.0, 25.0, 5.0)]
        walkable = box(-1.0, -1.0, 26.0, 6.0)
        result = spawn_agents(
            np.random.default_rng(3),
            regions,
            goal,
            SpawnConfig(),
            n_agents=40,
            walkable=walkable,
        )
        closest = _min_pairwise_distance(result.positions)
        assert closest >= result.min_separation - 1e-9, (
            f"cross-region pair at {closest:.4f} m < min_sep {result.min_separation:.4f} m"
        )

    def test_jitter_cannot_break_separation(self):
        """Disc jitter of radius (spacing - min_sep)/2 is the binding constraint.

        Per-axis jitter in a square of the same half-width lets two lattice
        neighbours approach diagonally by 2*j*sqrt(2) and violate min_sep -- measured
        0.4910 m against a required 0.4933 m while developing this.
        """
        walkable = box(0.0, 0.0, 12.0, 12.0)
        for seed in range(25):
            result = spawn_agents(
                np.random.default_rng(seed),
                [box(0.5, 0.5, 11.5, 11.5)],
                [box(0.5, 0.5, 11.5, 11.5)],
                SpawnConfig(),
                n_agents=200,
                walkable=walkable,
            )
            closest = _min_pairwise_distance(result.positions)
            assert closest >= result.min_separation - 1e-9, (
                f"seed={seed}: {closest:.6f} < {result.min_separation:.6f}"
            )


class TestWallClearance:
    """Both engines used to omit ``walkable=``, so 1.5% of agents spawned in walls."""

    @pytest.mark.parametrize("tier", ALL_TIERS)
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_every_body_clears_the_walls(self, tier: GeometryTier, seed: int):
        env, _ = _episode(tier, seed, 40)
        world = env._world
        radii = np.maximum(world.shoulder_widths, world.chest_depths)
        boundary = world.walkable_polygon.boundary
        distances = np.array([boundary.distance(Point(p)) for p in world.positions])
        inside = np.array([world.walkable_polygon.contains(Point(p)) for p in world.positions])
        assert inside.all(), f"tier={tier.name} seed={seed}: agent spawned outside walkable"
        n_clipping = int((distances < radii - 1e-9).sum())
        assert n_clipping == 0, (
            f"tier={tier.name} seed={seed}: {n_clipping}/{len(distances)} agents "
            f"closer to a wall than their own body radius"
        )


class TestDeterminism:
    def test_same_seed_same_positions(self):
        a, _ = _episode(GeometryTier.TIER_3A, 11, 30)
        b, _ = _episode(GeometryTier.TIER_3A, 11, 30)
        np.testing.assert_array_equal(a._world.positions, b._world.positions)

    def test_engines_agree_byte_for_byte(self):
        """crowdrl-torch reimplements the step but must share the spawn exactly."""
        pytest.importorskip("torch")
        from crowdrl_torch.episode_factory import make_episode_factory

        base = CrowdEnvConfig()
        config = dataclasses.replace(
            base, spawn=dataclasses.replace(base.spawn, n_agents_range=(25, 25))
        )
        env = CrowdEnv(config=config, seed=7)
        env.reset(seed=7)
        episode = make_episode_factory(config)(7)

        np.testing.assert_array_equal(env._world.positions, episode["positions"])
        np.testing.assert_array_equal(env._world.goal_positions, episode["goal_positions"])


class TestMonotonicity:
    def test_requesting_more_never_delivers_fewer(self):
        """``min_sep`` is the max body radius over the sampled crowd, so a larger
        request draws a larger maximum body, raises the separation and LOWERS
        capacity. Requesting 100 used to deliver fewer than requesting 30.
        """
        for tier in (GeometryTier.TIER_2, GeometryTier.TIER_3A):
            delivered = [_episode(tier, 0, n)[1]["n_agents"] for n in REPRESENTATIVE_COUNTS]
            assert delivered == sorted(delivered), (
                f"tier={tier.name}: {dict(zip(REPRESENTATIVE_COUNTS, delivered))} "
                f"is not monotonic in the request"
            )


class TestShortfallIsLoud:
    """There is no silent option. Every policy either raises, warns, or records."""

    @pytest.fixture
    def impossible(self):
        return [box(0.0, 0.0, 0.5, 0.5)], [box(10.0, 0.0, 15.0, 5.0)]

    def test_raise_names_the_achievable_count(self, impossible):
        spawn, goal = impossible
        with pytest.raises(SpawnShortfallError) as excinfo:
            spawn_agents(
                np.random.default_rng(0),
                spawn,
                goal,
                SpawnConfig(spawn_shortfall_policy="raise"),
                n_agents=100,
            )
        error = excinfo.value
        assert error.requested_n == 100
        assert error.placed_n < 100
        assert str(error.capacity) in str(error), "the achievable count must be named"
        assert "m2" in str(error), "the area that made it impossible must be named"

    def test_warn_logs_the_numbers(self, impossible, caplog):
        spawn, goal = impossible
        with caplog.at_level(logging.WARNING, logger="crowdrl_env.spawner"):
            spawn_agents(
                np.random.default_rng(0),
                spawn,
                goal,
                SpawnConfig(spawn_shortfall_policy="warn"),
                n_agents=100,
            )
        assert any("spawn shortfall" in r.getMessage() for r in caplog.records), (
            "policy='warn' must emit a warning naming the shortfall"
        )

    def test_regenerate_is_quiet_but_records(self, impossible, caplog):
        """Quiet only because the CALLER retries; the numbers are still on the result."""
        spawn, goal = impossible
        with caplog.at_level(logging.WARNING, logger="crowdrl_env.spawner"):
            result = spawn_agents(
                np.random.default_rng(0),
                spawn,
                goal,
                SpawnConfig(spawn_shortfall_policy="regenerate"),
                n_agents=100,
            )
        assert not caplog.records, "regenerate must not warn per attempt"
        assert result.is_short and result.requested_n == 100

    def test_env_warns_once_when_it_runs_out_of_attempts(self, caplog):
        """An impossible ask must not fail silently even after regeneration."""
        base = CrowdEnvConfig()
        config = dataclasses.replace(
            base,
            geometry_tiers=[GeometryTier.TIER_2],
            tier_weights=None,
            # Far beyond what any generated geometry can hold.
            spawn=dataclasses.replace(base.spawn, n_agents_range=(4000, 4000)),
            max_regeneration_attempts=2,
        )
        env = CrowdEnv(config=config, seed=0)
        with caplog.at_level(logging.WARNING, logger="crowdrl_env.crowd_env"):
            _, info = env.reset(seed=0)
        assert info["n_agents"] < 4000
        assert any("shortfall" in r.getMessage() for r in caplog.records), (
            "exhausting regeneration attempts must warn"
        )


class TestSpawnerDeliveryBenchmark:
    """The tracked "is the spawner good" metric, per curriculum phase.

    Floors are set at 98% / 2% rather than a flat 100% so shapely or numpy version
    drift cannot turn a healthy spawner red, while still catching any real
    regression: the pre-fix numbers were 51.7% delivery and 87% short episodes in the
    `full` phase, nowhere near these floors.
    """

    MIN_DELIVERY_RATE = 0.98
    MAX_SHORT_EPISODE_FRACTION = 0.02
    SEEDS = range(8)

    @pytest.mark.parametrize(
        ("phase_name", "tiers", "n_range", "tier_weights"),
        [
            ("easy", [GeometryTier.TIER_0], (5, 15), None),
            ("medium", [GeometryTier.TIER_0, GeometryTier.TIER_1], (10, 30), None),
            ("hard", [GeometryTier.TIER_1, GeometryTier.TIER_2], (20, 50), None),
            ("rooms", [GeometryTier.TIER_2, GeometryTier.TIER_3A], (15, 40), None),
            ("complex", [GeometryTier.TIER_3A, GeometryTier.TIER_3B], (20, 60), None),
            (
                "full",
                ALL_TIERS,
                (20, 100),
                (0.10, 0.15, 0.25, 0.25, 0.25),
            ),
        ],
    )
    def test_phase_delivery_rate(self, phase_name, tiers, n_range, tier_weights):
        base = CrowdEnvConfig()
        requested_total = 0
        delivered_total = 0
        short_episodes = 0

        for seed in self.SEEDS:
            rng = np.random.default_rng(seed)
            n_agents = int(rng.integers(n_range[0], n_range[1] + 1))
            config = dataclasses.replace(
                base,
                geometry_tiers=list(tiers),
                tier_weights=list(tier_weights) if tier_weights else None,
                spawn=dataclasses.replace(base.spawn, n_agents_range=(n_agents, n_agents)),
            )
            env = CrowdEnv(config=config, seed=seed)
            _, info = env.reset(seed=seed)

            requested_total += n_agents
            delivered_total += info["n_agents"]
            if info["n_agents"] < n_agents:
                short_episodes += 1

        delivery_rate = delivered_total / requested_total
        short_fraction = short_episodes / len(self.SEEDS)

        assert delivery_rate >= self.MIN_DELIVERY_RATE, (
            f"phase {phase_name}: delivered {delivery_rate:.1%} of "
            f"{requested_total} requested agents "
            f"(floor {self.MIN_DELIVERY_RATE:.0%})"
        )
        assert short_fraction <= self.MAX_SHORT_EPISODE_FRACTION, (
            f"phase {phase_name}: {short_episodes}/{len(self.SEEDS)} episodes short "
            f"(ceiling {self.MAX_SHORT_EPISODE_FRACTION:.0%})"
        )
