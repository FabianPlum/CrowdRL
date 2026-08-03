"""Crowd composition sampler and agent spawner.

Samples agent count, body dimensions (from anthropometric distributions),
desired speeds, spawn positions, and goal positions for each episode.
Returns arrays ready to populate a WorldState.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

from crowdrl_core.geometry import sample_point_in_polygon

logger = logging.getLogger(__name__)

ShortfallPolicy = Literal["warn", "regenerate", "raise"]
SamplerName = Literal["lattice", "rejection"]

_SATURATION_FAILURE_STREAK = 3
"""Consecutive placement failures after which a region is treated as jammed.

Rejection sampling that has just failed ``max_attempts`` times in a row three agents
running is not going to succeed on the fourth; stopping bounds the cost of a request
that exceeds capacity instead of burning the full budget per remaining agent."""

_LATTICE_SPACING_STEPS = 8
"""Maximum spacing reductions when a coarse lattice yields too few cells."""

_LATTICE_SPACING_DECAY = 0.85
"""Factor the lattice spacing shrinks by per retry (yields ~1.4x more cells)."""

_LATTICE_SPACING_SLACK = 0.05
"""Floor for the lattice spacing, as ``min_sep * (1 + this)``.

Only the floor: the spacing actually used is scaled to the crowd (see
:func:`_sample_hex_lattice`) and drops here only when the region is full. The slack is
what jitter is drawn from -- it breaks up the visually obvious crystalline grid without
ever letting two neighbours close to within ``min_sep``. Larger values look more natural
but cost capacity (spacing grows, so fewer cells fit)."""


class SpawnShortfallError(RuntimeError):
    """Raised when fewer agents can be placed than were requested.

    Carries the numbers needed to act on it: how many were asked for, how many
    fit, and the area / analytic capacity that made it impossible.
    """

    def __init__(
        self,
        requested_n: int,
        placed_n: int,
        spawn_area_m2: float,
        capacity: int,
    ) -> None:
        self.requested_n = requested_n
        self.placed_n = placed_n
        self.spawn_area_m2 = spawn_area_m2
        self.capacity = capacity
        super().__init__(
            f"spawn shortfall: requested {requested_n} agents, only {placed_n} fit "
            f"in {spawn_area_m2:.2f} m2 of spawn region "
            f"(estimated capacity {capacity}). "
            f"Enlarge the spawn region (SpawnConfig.max_spawn_dilation), lower the "
            f"agent count to {placed_n}, or set "
            f"spawn_shortfall_policy='regenerate' to try another geometry."
        )


def format_shortfall(
    requested_n: int,
    placed_n: int,
    spawn_area_m2: float,
    capacity: int,
    min_separation: float,
) -> str:
    """The one place a shortfall is turned into words.

    Shared by :attr:`SpawnResult.shortfall_summary`, the spawner's own warning, and
    both engines' give-up warnings, so a reader never has to reconcile three slightly
    different renderings of the same event.
    """
    return (
        f"requested={requested_n} placed={placed_n} "
        f"spawn_area={spawn_area_m2:.2f}m2 capacity={capacity} "
        f"min_sep={min_separation:.3f}m"
    )


def spawn_capacity(area_m2: float, min_sep: float) -> int:
    """Estimate how many agents fit in ``area_m2`` at ``min_sep`` separation.

    Hexagonal (densest) circle packing puts one centre per
    ``min_sep**2 * sqrt(3)/2`` of area. Deliberately NOT derated for polygon edges
    or holes: this drives the dilation target, and under-reporting it would grow
    spawn regions further than needed (dissolving the directional-flow semantics)
    or reject requests a sampler can in fact satisfy. A 0.70-style efficiency factor
    predicts 3 for a 1.25 m2 doorway that measurably takes 5.

    This is the *asymptotic* density bound, not a hard ceiling. On a region only a
    few separations wide, boundary effects let real placements exceed it slightly --
    two bodies fit in a 0.5 m box at 0.53 m apart across the diagonal while this
    returns 1 -- because a centre on the boundary consumes less than a full
    hexagonal cell. Accurate to within a few percent once the region spans several
    ``min_sep``, which is the regime that matters for capacity decisions. For a
    guaranteed ceiling, bound the region dilated by ``min_sep / 2`` instead.
    """
    if area_m2 <= 0.0 or min_sep <= 0.0:
        return 0
    return int(area_m2 / (min_sep**2 * math.sqrt(3.0) / 2.0))


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

    # Agent mass distribution
    mass_mean: float = 80.0
    """Mean agent mass (kg). Literature: ~80 kg for mixed-gender adults."""
    mass_std: float = 15.0
    mass_min: float = 40.0
    """Minimum agent mass (kg)."""

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

    min_spawn_separation: float = 0.3
    """Minimum distance between spawned agents (metres).

    Note: effective separation is max(this, 2 * max_body_radius), so this
    only matters when agents are unusually small.
    """

    max_spawn_attempts: int = 50
    """Maximum rejection-sampling attempts per agent before giving up.

    Used by ``spawn_sampler="rejection"`` and by the lattice sampler's top-up pass.
    Raising it does not help a tight region: the shortfall there is capacity-bound,
    not budget-bound (measured: 50 -> 5000 attempts buys 0-3 agents).
    """

    spawn_sampler: SamplerName = "lattice"
    """How spawn positions are placed.

    - ``"lattice"``: jittered hexagonal lattice. Reaches the region's packing
      capacity, is deterministic given a seed, and has no ``O(n^2)`` inner loop.
    - ``"rejection"``: the original per-agent rejection sampling. Kept as the
      benchmark baseline (see ``test_spawn_capacity.py``); jams below capacity on
      tight regions.
    """

    max_spawn_dilation: float = 3.0
    """How far (metres) a spawn region may be grown to fit the requested crowd.

    Spawn regions inherit the base tier's entry zones, which are only ~5-40% of the
    walkable area -- a doorway or corridor end-cap can be 1.25 m2 and hold about 5
    bodies. Rather than silently placing 5 of the 30 agents asked for, the region is
    buffered outward (clipped to the walkable area, and never over the exit) until it
    satisfies :attr:`target_initial_density` and :attr:`spawn_dilation_headroom`.
    Growth is the smallest that suffices, so agents normally stay anchored to the entry
    side and directional-flow scenarios keep their meaning. If this cap is still not
    enough, :func:`dilate_spawn_regions` falls back to the whole walkable area minus the
    exit -- reported via :attr:`SpawnResult.dilation_applied`.

    0.0 disables dilation entirely. Requires ``walkable`` to be passed to
    :func:`spawn_agents` -- without it there is nothing to clip the growth against, so
    dilation is skipped.
    """

    separate_spawn_from_exit: bool = True
    """Keep the spawn zone off the goal zone -- see :func:`separate_from_exit`.

    Overlapping zones spawn agents on their own goals, so the task starts solved and
    any flow measurement through the exit is meaningless. Disable only to reproduce
    older episode distributions.
    """

    target_initial_density: float = 1.0
    """Spawn density to dilate towards, in agents per m2 of spawn area.

    Sizing the spawn zone to merely *fit* the crowd starts every body at the minimum
    separation -- i.e. in contact with its neighbours -- which produces contact forces
    and collision penalties from the first step purely as a spawn artefact. Measured on
    a TIER_0 layout: 8 agents in the bare 3.28 m2 entry zone (2.4 ped/m2) begin at
    0.549 m spacing against a 0.549 m minimum and log 321 collisions in 200 steps;
    dilating to ~1 ped/m2 gives 0.776 m spacing and 20 collisions.

    1.0 ped/m2 is a comfortable standing density, well clear of the 2+ ped/m2 jamming
    regime. This is a *target*, not a guarantee: it is capped by ``max_spawn_dilation``
    and by the walkable area, so genuinely dense requests (100 agents in a small room)
    still pack tight -- which is what those scenarios are for.

    Set to 0.0 to size the spawn zone for bare feasibility only.
    """

    spawn_dilation_headroom: float = 1.15
    """Feasibility floor when dilating, as a multiple of the requested count.

    Applied on top of :attr:`target_initial_density`: the analytic capacity is a
    hexagonal-packing bound that a real sampler undershoots on concave regions, so
    target ``headroom * n`` capacity to make the request actually reachable rather than
    merely theoretically feasible. Only binds when the density target is disabled or
    unreachable.
    """

    spawn_shortfall_policy: ShortfallPolicy = "regenerate"
    """What to do when fewer agents fit than were requested.

    - ``"raise"``: raise :class:`SpawnShortfallError` immediately.
    - ``"regenerate"``: return the short result quietly and let the caller discard
      the geometry and try another (what :class:`~crowdrl_env.crowd_env.CrowdEnv`
      does); the caller warns once if it runs out of attempts.
    - ``"warn"``: accept the short crowd and log a warning naming the numbers.

    There is deliberately no silent option. A shortfall changes the density an
    episode actually trains or evaluates at, so it must be visible somewhere.
    """

    min_spawn_goal_distance: float = 3.0
    """Minimum distance (metres) between an agent's spawn and its goal.

    Goals are sampled per agent independently of spawns, and for several tiers
    the spawn and goal regions are the SAME walkable polygon, so without this a
    goal can land on top of its spawn (a trivial/degenerate task). Enforced via
    rejection sampling with a farthest-of-attempts fallback for geometries too
    small to satisfy it.
    """

    max_goal_attempts: int = 32
    """Maximum rejection-sampling attempts per goal to satisfy
    ``min_spawn_goal_distance`` before falling back to the farthest candidate."""


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
    masses: NDArray[np.float64]
    """(n_agents,) — agent mass in kg."""
    goal_positions: NDArray[np.float64]
    """(n_agents, 2)"""
    preferred_speeds: NDArray[np.float64]
    """(n_agents,) — per-agent preferred speed for reward computation."""

    # --- Provenance: what was asked for vs. what the geometry could hold. ---
    requested_n: int = 0
    """Agent count requested. Differs from :attr:`n_agents` on a shortfall."""

    spawn_area_m2: float = 0.0
    """Area of the (deduplicated) spawn regions actually sampled from."""

    capacity: int = 0
    """Analytic upper bound for :attr:`spawn_area_m2` -- see :func:`spawn_capacity`."""

    min_separation: float = 0.0
    """Effective separation enforced: ``max(config value, 2 * max body radius)``."""

    dilation_applied: float = 0.0
    """Metres the spawn regions were grown to fit the request (0.0 = untouched)."""

    @property
    def n_agents(self) -> int:
        return self.positions.shape[0]

    @property
    def is_short(self) -> bool:
        """True when fewer agents were placed than requested."""
        return self.n_agents < self.requested_n

    @property
    def shortfall_summary(self) -> str:
        """One-line description of the shortfall, for logs and error messages."""
        return format_shortfall(
            self.requested_n,
            self.n_agents,
            self.spawn_area_m2,
            self.capacity,
            self.min_separation,
        )


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

    # Sample agent masses
    masses = np.clip(
        rng.normal(config.mass_mean, config.mass_std, n_agents),
        config.mass_min,
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
    # so that no two agent ellipses overlap at spawn.
    # NOTE this is the max over the *sampled* crowd, so a larger n_agents draws a
    # larger maximum body and raises min_sep, which LOWERS capacity. Requesting
    # more agents can therefore place fewer of them.
    min_sep = max(config.min_spawn_separation, 2.0 * max_body_radius)

    spawn_margin = max_body_radius if walkable is not None else 0.0

    # Keep the spawn zone off the exit before sizing anything, so capacity and
    # dilation are computed on the area agents may actually start in.
    if config.separate_spawn_from_exit:
        spawn_regions = separate_from_exit(spawn_regions, goal_regions, clearance=max_body_radius)

    dilation_applied = 0.0
    if walkable is not None and config.max_spawn_dilation > 0.0:
        spawn_regions, dilation_applied = dilate_spawn_regions(
            spawn_regions,
            walkable,
            n_agents,
            min_sep,
            config.max_spawn_dilation,
            margin=spawn_margin,
            headroom=config.spawn_dilation_headroom,
            # Never grow the spawn zone over the exit.
            exclude=goal_regions,
            target_density=config.target_initial_density,
        )

    effective_spawn_area = _effective_area(
        spawn_regions,
        walkable=walkable,
        margin=spawn_margin,
    )
    capacity = spawn_capacity(effective_spawn_area, min_sep)

    # Sample positions with minimum separation. Both samplers work off regions that
    # are ALREADY clipped to walkable and eroded by the body radius, so neither pays
    # to re-buffer the walkable polygon per candidate.
    sampling_regions = _as_polygon_list(_sampling_geometry(spawn_regions, walkable, spawn_margin))
    if config.spawn_sampler == "lattice":
        positions = _sample_hex_lattice(rng, sampling_regions, n_agents, min_sep)
        # A hex lattice has fixed spacing, so a thin or awkwardly-angled region can
        # leave usable gaps between its rows. Top the crowd up by rejection sampling
        # against the points already placed -- never fewer agents than either
        # sampler alone would have managed.
        if len(positions) < n_agents and sampling_regions:
            positions = _sample_separated_points(
                rng,
                sampling_regions,
                n_agents - len(positions),
                min_sep,
                config.max_spawn_attempts,
                initial=positions,
            )
    else:
        positions = _sample_separated_points(
            rng,
            sampling_regions,
            n_agents,
            min_sep,
            config.max_spawn_attempts,
        )
    actual_n = len(positions)

    # Trim arrays if some agents couldn't be placed
    if actual_n < n_agents:
        shoulder_widths = shoulder_widths[:actual_n]
        chest_depths = chest_depths[:actual_n]
        masses = masses[:actual_n]
        preferred_speeds = preferred_speeds[:actual_n]

        summary = format_shortfall(n_agents, actual_n, effective_spawn_area, capacity, min_sep)
        if config.spawn_shortfall_policy == "raise":
            raise SpawnShortfallError(n_agents, actual_n, effective_spawn_area, capacity)
        if config.spawn_shortfall_policy == "warn":
            logger.warning("spawn shortfall: %s", summary)
        else:
            # "regenerate": the caller discards this geometry and retries, so a
            # warning here would fire once per attempt. It warns if it gives up.
            logger.debug("spawn shortfall (regenerating): %s", summary)

    # Sample goal positions (one per agent), enforcing a minimum spawn->goal distance.
    # separate_from_exit above keeps the spawn ZONE off the goal zone, but goals are
    # still drawn independently of spawns, and a few tiers deliberately share one
    # polygon for both (where separation is declined), so an individual goal can still
    # land on its own spawn (init_goal_distance ~= 0 -> trivial task + huge
    # temporal-memory obs ratios). This per-agent check is the remaining guard:
    # reject-sample, and if the geometry is too small to satisfy the distance within
    # max_goal_attempts, keep the FARTHEST candidate seen.
    #
    # Clip the goal regions to walkable ONCE. Passing walkable per-sample would
    # re-erode the walkable polygon on every one of the up-to
    # max_goal_attempts * n_agents candidates, and would raise outright on a goal
    # region too narrow to survive the erosion. Prefer full body clearance; fall back
    # to bare walkable containment for regions that cannot afford it (a narrow
    # doorway is still a legitimate goal -- reaching it is a proximity test).
    goal_sampling_regions = goal_regions
    if walkable is not None:
        goal_sampling_regions = _as_polygon_list(
            _sampling_geometry(goal_regions, walkable, max_body_radius)
        )
        if not goal_sampling_regions:
            goal_sampling_regions = _as_polygon_list(
                _sampling_geometry(goal_regions, walkable, 0.0)
            )
        if not goal_sampling_regions:
            goal_sampling_regions = goal_regions

    min_sg = config.min_spawn_goal_distance
    goal_positions = np.empty((actual_n, 2), dtype=np.float64)
    for i in range(actual_n):
        spawn_i = positions[i]
        chosen = None
        best, best_d = None, -1.0
        for _ in range(config.max_goal_attempts):
            region = goal_sampling_regions[rng.integers(len(goal_sampling_regions))]
            cand = sample_point_in_polygon(region, rng)
            d = float(np.hypot(cand[0] - spawn_i[0], cand[1] - spawn_i[1]))
            if d >= min_sg:
                chosen = cand
                break
            if d > best_d:
                best_d, best = d, cand
        goal_positions[i] = chosen if chosen is not None else best

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
        masses=masses,
        goal_positions=goal_positions,
        preferred_speeds=preferred_speeds,
        requested_n=n_agents,
        spawn_area_m2=effective_spawn_area,
        capacity=capacity,
        min_separation=min_sep,
        dilation_applied=dilation_applied,
    )


def separate_from_exit(
    regions: list[Polygon],
    exclude: list[Polygon] | None,
    clearance: float = 0.0,
    min_retained_fraction: float = 0.02,
) -> list[Polygon]:
    """Remove the exit (goal) zone from the spawn zone.

    A spawn region overlapping its own goal region spawns agents on top of their
    goals: the task is already solved at t=0, the episode contributes a trivial
    learning signal, and any flow measurement through the exit is meaningless. The
    TIER_3B generator overlaps them by ~9 m2 unprompted, so 28 of 30 agents could
    start inside the goal zone.

    Trimmed with ``clearance`` first (so spawns are not flush against the exit), then
    without it. The retained fraction only has to be small: what is left is a seed for
    dilation, which grows it back into non-exit walkable area. The threshold exists
    solely for tiers that deliberately use one polygon for both, where trimming leaves
    nothing and having nowhere to spawn is worse than overlapping.
    """
    if not exclude or not regions:
        return regions

    base = unary_union(regions)
    if base.is_empty:
        return regions
    forbidden = unary_union(exclude)
    if forbidden.is_empty or not base.intersects(forbidden):
        return regions

    for pad in (clearance, 0.0):
        blocked = forbidden.buffer(pad) if pad > 0.0 else forbidden
        polygons = _as_polygon_list(base.difference(blocked))
        retained = sum(p.area for p in polygons)
        if polygons and retained >= min_retained_fraction * base.area:
            return polygons

    return regions


def _as_polygon_list(geom) -> list[Polygon]:
    """Normalise a Shapely result to a list of non-empty Polygons.

    ``buffer``/``intersection`` may return a Polygon, a MultiPolygon, or a
    GeometryCollection containing degenerate lines/points when a region is clipped
    to a thin passage. Only genuine Polygons are usable as spawn regions.
    """
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and not g.is_empty]


def dilate_spawn_regions(
    regions: list[Polygon],
    walkable: Polygon,
    n_agents: int,
    min_sep: float,
    max_dilation: float,
    margin: float = 0.0,
    headroom: float = 1.15,
    exclude: list[Polygon] | None = None,
    target_density: float = 0.0,
) -> tuple[list[Polygon], float]:
    """Grow ``regions`` outward until they can hold ``n_agents``.

    Buffers the union of the spawn regions by an increasing distance, clipping each
    time to ``walkable`` so growth follows the geometry (down a corridor, into a
    room) instead of leaking through walls. Stops at the first distance whose
    analytic capacity covers ``headroom * n_agents``.

    Returns ``(regions, dilation_applied)``. When even ``max_dilation`` is not
    enough, returns the fully-dilated regions -- the caller reports the residual
    shortfall rather than pretending it fit.

    Growth is isotropic, so a corridor entry zone grows along the corridor as well as
    across it. That keeps agents clustered near the entry (unlike sampling the whole
    walkable area), which is what directional-flow and bottleneck scenarios need.

    ``exclude`` (the goal regions) is kept out of the GROWN area: unrestrained growth
    swallowed 100% of the exit zone on tiers 1-3A, which spawns agents on top of their
    own goals and destroys the directional structure the scenario exists to test. Only
    the added area is trimmed -- the original ``regions`` are always preserved, so
    tiers where the spawn and goal zone are deliberately the same polygon behave as
    they always did.
    """
    if max_dilation <= 0.0 or not regions:
        return regions, 0.0

    target_capacity = max(1, int(math.ceil(headroom * n_agents)))
    # Comfort target: enough AREA that the crowd starts at target_density rather than
    # shoulder to shoulder. Expressed as an equivalent capacity so one comparison
    # covers both criteria.
    target_area = n_agents / target_density if target_density > 0.0 else 0.0
    if target_area > 0.0:
        target_capacity = max(target_capacity, spawn_capacity(target_area, min_sep))

    def _sufficient(candidate: list[Polygon]) -> bool:
        return spawn_capacity(_effective_area(candidate, walkable, margin), min_sep) >= (
            target_capacity
        )

    if _sufficient(regions):
        return regions, 0.0

    base = unary_union(regions)
    # Clearance so a spawn does not sit flush against the exit either.
    forbidden = None
    if exclude:
        forbidden = unary_union(exclude)
        if margin > 0.0:
            forbidden = forbidden.buffer(margin)

    def _protect_exit(grown_geom):
        """Trim the exit zone out of newly-gained area, never out of ``base``."""
        if forbidden is None:
            return grown_geom
        added = grown_geom.difference(base).difference(forbidden)
        return base.union(added)

    # Linear scan: a handful of buffer ops is cheap next to the triangulation the
    # episode already pays for, and it finds the SMALLEST sufficient growth --
    # important, because over-dilating dissolves the directional-flow semantics.
    # The final candidate is max_dilation itself, so a coarse step never leaves
    # usable area unclaimed (with step == min_sep, the last multiple below the cap
    # can otherwise fall well short of it).
    step = max(min_sep, 0.25)
    n_steps = max(1, int(math.ceil(max_dilation / step)))
    distances = [min(step * (i + 1), max_dilation) for i in range(n_steps)]

    best_regions = regions
    best_distance = 0.0
    for distance in distances:
        grown = _as_polygon_list(_protect_exit(base.buffer(distance).intersection(walkable)))
        if grown:
            best_regions, best_distance = grown, distance
            if _sufficient(grown):
                return grown, distance

    # Last resort: the entry zone grown to its cap still cannot hold the crowd, so
    # fall back to the whole walkable area MINUS the exit zone. This trades the
    # directional-flow framing (agents end up spread through the geometry rather than
    # clustered at the entry) for delivering the requested count, which is the explicit
    # priority -- and it is reported via SpawnResult.dilation_applied / spawn_area_m2
    # either way.
    whole = _as_polygon_list(_protect_exit(walkable))
    if whole and spawn_capacity(
        _effective_area(whole, walkable, margin), min_sep
    ) > spawn_capacity(_effective_area(best_regions, walkable, margin), min_sep):
        return whole, max_dilation

    return best_regions, best_distance


def _sampling_geometry(
    regions: list[Polygon],
    walkable: Polygon | None = None,
    margin: float = 0.0,
):
    """The geometry a sampler may place centres in.

    Regions are unioned (so overlapping spawn regions are not double-counted),
    clipped to ``walkable``, then eroded by ``margin`` so every placed body keeps a
    full radius of wall clearance. May come back empty when the erosion closes a
    thin passage -- callers treat that as zero capacity, not an error.
    """
    if not regions:
        return unary_union([])
    area = unary_union(regions)
    if walkable is not None:
        area = area.intersection(walkable)
    if margin > 0.0:
        area = area.buffer(-margin)
    return area


def _effective_area(
    regions: list[Polygon],
    walkable: Polygon | None = None,
    margin: float = 0.0,
) -> float:
    """Area actually available to the sampler, in m2.

    Thin wrapper over :func:`_sampling_geometry` so the area a capacity decision is
    based on can never drift from the geometry the sampler is handed. Returns 0.0
    rather than raising when the erosion empties the region: an area of zero is itself
    the diagnosis.
    """
    geom = _sampling_geometry(regions, walkable, margin)
    return 0.0 if geom.is_empty else float(geom.area)


def _sample_hex_lattice(
    rng: np.random.Generator,
    regions: list[Polygon],
    n_points: int,
    min_sep: float,
) -> NDArray[np.float64]:
    """Place points on a jittered hexagonal lattice.

    ``regions`` must already be clipped to the walkable area and eroded by the body
    radius -- see :func:`_sampling_geometry`. Returns up to ``n_points`` positions,
    fewer only when the region genuinely cannot hold them.

    Two invariants that are easy to get wrong (both are pinned by tests):

    * Jitter is sampled in a **disc** of radius ``(spacing - min_sep)/2``, not
      per-axis in a square. Two lattice neighbours sitting ``spacing`` apart can each
      move by the jitter radius, so a square jitter lets them approach diagonally by
      ``2*j*sqrt(2)`` and break ``min_sep``.
    * The lattice is built over the **union** of the regions, not each region
      separately, or points from two adjacent regions can land closer than
      ``min_sep`` to each other.
    """
    # Union (not per-region) is what makes the cross-region separation hold.
    area = _sampling_geometry(regions)
    if area.is_empty or n_points <= 0:
        return np.empty((0, 2), dtype=np.float64)

    # Spacing is scaled to the crowd, not pinned to min_sep. Packing 8 agents at the
    # minimum separation into a region with room for 15 starts every body in contact
    # with a neighbour, which turns into contact forces and collision penalties on the
    # first step (measured: 66 collisions/episode against 12 for the region-filling
    # spread it replaced). Spread the requested crowd over the WHOLE region instead:
    # the spacing at which n hexagonal cells exactly tile the available area, floored
    # at min_sep so genuinely dense requests still pack tight.
    floor_spacing = min_sep * (1.0 + _LATTICE_SPACING_SLACK)
    ideal_spacing = math.sqrt(area.area / (n_points * math.sqrt(3.0) / 2.0))
    spacing = max(floor_spacing, ideal_spacing)

    # A coarse lattice loses cells to boundary effects, so step down towards the floor
    # until enough cells fit. Ends at the floor, which is the densest legal packing.
    cells = np.empty((0, 2))
    jitter_radius = 0.0
    for _ in range(_LATTICE_SPACING_STEPS):
        jitter_radius = (spacing - min_sep) / 2.0
        # Erode by the jitter radius too, so a jittered point cannot escape the region
        # (and therefore keeps its full body-radius wall clearance).
        cell_area = area.buffer(-jitter_radius) if jitter_radius > 0.0 else area
        if cell_area.is_empty:
            cell_area = area

        cells = _lattice_cells(cell_area, spacing)
        if len(cells) >= n_points or spacing <= floor_spacing:
            break
        spacing = max(floor_spacing, spacing * _LATTICE_SPACING_DECAY)

    if len(cells) == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Shuffle so a partial fill is spread over the region rather than bottom-up.
    chosen = cells[rng.permutation(len(cells))[:n_points]]

    if jitter_radius > 0.0:
        theta = rng.uniform(0.0, 2.0 * math.pi, len(chosen))
        # sqrt for a uniform areal distribution within the disc
        radius = jitter_radius * np.sqrt(rng.uniform(0.0, 1.0, len(chosen)))
        chosen = chosen + np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])

    return np.ascontiguousarray(chosen, dtype=np.float64)


def _lattice_cells(region, spacing: float) -> NDArray[np.float64]:
    """Hexagonal lattice points at ``spacing`` that fall inside ``region``."""
    dx = spacing
    dy = spacing * math.sqrt(3.0) / 2.0
    min_x, min_y, max_x, max_y = region.bounds

    n_rows = max(1, int((max_y - min_y) / dy) + 1)
    rows = []
    for row in range(n_rows):
        y = min_y + row * dy
        x_start = min_x + (dx / 2.0 if row % 2 else 0.0)
        n_cols = max(1, int((max_x - x_start) / dx) + 1)
        xs = x_start + np.arange(n_cols) * dx
        rows.append(np.column_stack([xs, np.full(n_cols, y)]))
    candidates = np.vstack(rows) if rows else np.empty((0, 2))

    prepared = prep(region)
    inside = np.fromiter(
        (prepared.contains(Point(x, y)) for x, y in candidates),
        dtype=bool,
        count=len(candidates),
    )
    return candidates[inside]


def _sample_separated_points(
    rng: np.random.Generator,
    regions: list[Polygon],
    n_points: int,
    min_sep: float,
    max_attempts: int,
    initial: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Sample ``n_points`` more points with minimum pairwise separation.

    ``regions`` must already be clipped to the walkable area and eroded by the body
    radius -- see :func:`_sampling_geometry`. Clipping here instead would re-buffer the
    walkable polygon once per candidate.

    Used two ways: as the ``spawn_sampler="rejection"`` baseline the lattice is
    benchmarked against, and to top up a lattice that left usable gaps (pass the
    lattice points as ``initial`` -- new points are separated from those too, and
    they are included in the return value).

    Returns up to ``len(initial) + n_points`` positions (fewer if placement is tight).
    """
    placed: list[NDArray[np.float64]] = [] if initial is None else [p for p in initial]

    if not regions:
        return np.empty((0, 2), dtype=np.float64)

    consecutive_failures = 0
    for _ in range(n_points):
        region = regions[rng.integers(len(regions))]
        # Vectorised separation test. The original per-point Python generator made
        # this O(n^2) in interpreted code and dominated episode generation.
        existing = np.asarray(placed, dtype=np.float64) if placed else None
        for _attempt in range(max_attempts):
            candidate = sample_point_in_polygon(region, rng)
            if existing is None or np.all(np.linalg.norm(existing - candidate, axis=1) >= min_sep):
                placed.append(candidate)
                consecutive_failures = 0
                break
        else:
            # Falling through the attempt budget means this agent could not be placed.
            consecutive_failures += 1
            if consecutive_failures >= _SATURATION_FAILURE_STREAK:
                # The region is jammed; every further agent would burn the full
                # attempt budget to fail as well. Requesting far more agents than fit
                # would otherwise cost O(n_points * max_attempts * n_placed).
                break

    if not placed:
        # Shaped (0, 2) so downstream vector maths still broadcasts. Placing nothing
        # is reported through the normal shortfall path in spawn_agents, so the
        # "regenerate" policy can discard this geometry rather than crash on it.
        return np.empty((0, 2), dtype=np.float64)

    return np.array(placed, dtype=np.float64)
