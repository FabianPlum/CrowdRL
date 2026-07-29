"""Byte-exact CrowdRL deployment inside JuPedSim.

``LearnedPolicyModel`` is the interactive adapter: per-agent callbacks, the
router's waypoint, per-agent physics. Faithful at route level, but not
bit-exact against CrowdRL's native pipeline -- the waypoint source alone
disagrees by tens of degrees near corners and drops the ``path_deviation``
signal entirely (measured up to +4 sigma under the baked normalizer stats).

``LockstepPolicyModel`` exists for validation runs where the JuPedSim-hosted
simulation must reproduce the CrowdRL-native simulation **byte-identically**.
It runs the entire native batched step (same observation builder, one batched
ONNX call, same interpreters, same contact physics, same wall projection,
same array ordering, navmesh funnel waypoints with true path deviation) once
per JuPedSim iteration and serves each agent its precomputed row. JuPedSim's
compute-then-apply pass reads pre-step state for every agent, so the first
callback of a pass sees exactly the synchronous snapshot the native loop
uses.

Native removal semantics are applied inside the model: a row landing inside
an exit polygon is frozen and excluded from every subsequent batch
immediately (JuPedSim's own exit stage lags removal by 2 iterations, which
would otherwise leak the lingering agent into neighbours' observations).

Verified on the corner scenario (1082 steps, every position of every agent
``np.array_equal`` to the native reference, identical exit steps) and the
bottleneck (12 agents, 400 contact-heavy steps, byte-identical).

Requirements and caveats:

* ``walkable_geometry`` and ``exit_geometries`` are required -- the model
  builds the navmesh (needs the ``crowdrl-core[geometry]`` extra) and owns
  removal semantics.
* Dynamics parameters (``desired_velocity_weight``, contact constants) must
  match the reference run by hand -- they do not travel in the ONNX metadata
  (schema v1 covers obs/action configs only).
* Byte-identity is machine-scoped: it additionally relies on ONNX Runtime
  producing identical results for the same session on the same hardware
  (measured exact here, including batch-vs-single-row).
* Fixed final goals per agent are assumed (single exit stage journeys).
* Roughly 2 navmesh path queries per agent per step -- markedly slower than
  ``LearnedPolicyModel``; intended for validation, not interactive use.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from crowdrl_core.action import ActionConfig, interpret_actions_batch
from crowdrl_core.collision import (
    compute_contact_forces,
    detect_collisions,
    enforce_wall_boundaries,
)
from crowdrl_core.geometry import extract_wall_segments
from crowdrl_core.observation import ObsConfig, build_observations_batch
from crowdrl_core.world_state import WorldState

from .model import CrowdRLAgentState, CustomOperationalModel
from .policy import Policy, resolve_configs, resolve_dynamics

if TYPE_CHECKING:  # pragma: no cover
    from shapely import Polygon


def native_batch_step(
    world: WorldState,
    policy_batch,
    obs_config: ObsConfig,
    action_config: ActionConfig,
    *,
    desired_velocity_weight: float,
    max_velocity_magnitude: float,
    contact_stiffness: float,
    contact_damping: float,
    dt: float,
):
    """One CrowdRL-native step over ``world``, in place. THE shared step.

    Both the lockstep model and any native reference loop must call this
    function -- sharing the code is what makes byte-identity a structural
    property instead of a maintenance promise. Mirrors ``CrowdEnv.step``
    order exactly: observations -> batched policy -> interpreters -> velocity
    filter -> contact accelerations -> magnitude clamp -> integration ->
    body-clearance wall projection.

    ``policy_batch`` maps an (N, obs_dim) float64 array to (N, action_dim)
    raw actions. Returns the interpreter batch result (new headings live
    there; ``world`` carries everything else).
    """
    obs = build_observations_batch(world, obs_config)
    actions = np.asarray(policy_batch(obs), dtype=np.float64)
    batch = interpret_actions_batch(
        actions,
        world.torso_orientations,
        world.torso_orientations,
        world.head_orientations,
        action_config,
        current_speeds=np.linalg.norm(world.velocities, axis=1),
    )
    w = desired_velocity_weight
    world.velocities[:] = w * batch.desired_velocities + (1.0 - w) * world.velocities
    world.torso_orientations[:] = batch.new_torso_orientations
    world.head_orientations[:] = batch.new_head_orientations

    accel = compute_contact_forces(
        world,
        stiffness=contact_stiffness,
        damping=contact_damping,
        collisions=detect_collisions(world),
    )
    world.velocities[:] += accel * dt
    speeds = np.linalg.norm(world.velocities, axis=1)
    fast = speeds > max_velocity_magnitude
    if np.any(fast):
        world.velocities[:] *= np.where(
            fast, max_velocity_magnitude / np.maximum(speeds, 1e-10), 1.0
        )[:, None]
    world.positions[:] += world.velocities * dt
    enforce_wall_boundaries(world)
    return batch


class _Row:
    """Internal per-agent record. Arrays are copy-on-write like TemporalMemory."""

    __slots__ = (
        "position",
        "velocity",
        "heading",
        "torso",
        "head",
        "shoulder",
        "chest",
        "mass",
        "preferred_speed",
        "goal",
        "spawn",
        "init_gdist",
        "cum_path",
        "pos_hist",
        "gdist_hist",
    )

    def __init__(self, state, goal, buf_size: int) -> None:
        self.position = np.asarray(state.position, dtype=np.float64)
        self.velocity = np.asarray(getattr(state, "velocity", (0.0, 0.0)), dtype=np.float64)
        self.heading = float(getattr(state, "heading", 0.0))
        self.torso = float(getattr(state, "torso_angle", 0.0))
        self.head = float(getattr(state, "head_angle", 0.0))
        self.shoulder = float(getattr(state, "shoulder_width", 0.225))
        self.chest = float(getattr(state, "chest_depth", 0.15))
        self.mass = float(getattr(state, "mass", 80.0))
        self.preferred_speed = float(getattr(state, "preferred_speed", 1.34))
        self.goal = np.asarray(goal, dtype=np.float64)
        self.spawn = self.position.copy()
        self.init_gdist = float(np.linalg.norm(self.goal - self.position))
        self.cum_path = 0.0
        self.pos_hist = np.tile(self.position, (buf_size, 1))
        self.gdist_hist = np.full(buf_size, self.init_gdist)

    def as_state(self) -> CrowdRLAgentState:
        return CrowdRLAgentState(
            position=(float(self.position[0]), float(self.position[1])),
            velocity=(float(self.velocity[0]), float(self.velocity[1])),
            heading=self.heading,
            torso_angle=self.torso,
            head_angle=self.head,
            shoulder_width=self.shoulder,
            chest_depth=self.chest,
            mass=self.mass,
            preferred_speed=self.preferred_speed,
        )


class LockstepPolicyModel(CustomOperationalModel):
    """Byte-exact validation deployment: the native batched step per pass."""

    def __init__(
        self,
        policy: Policy,
        *,
        walkable_geometry: Polygon,
        exit_geometries,
        obs_config: ObsConfig | None = None,
        action_config: ActionConfig | None = None,
        desired_velocity_weight: float | None = None,
        max_velocity_magnitude: float | None = None,
        contact_stiffness: float | None = None,
        contact_damping: float | None = None,
    ) -> None:
        """
        Parameters
        ----------
        policy
            Batched-capable policy; ``OnnxPolicy`` artefacts self-configure
            exactly as in ``LearnedPolicyModel``.
        walkable_geometry
            The same shapely Polygon given to ``jps.Simulation(geometry=...)``.
            Sources the wall-segment array, the navmesh (funnel waypoints +
            true path deviation) and the contact physics.
        exit_geometries
            The exit polygons registered as exit stages, in any order. Rows
            landing inside any of them are removed from the batch immediately
            (native semantics); JuPedSim's own removal lags 2 iterations.
        desired_velocity_weight, max_velocity_magnitude, contact_stiffness,
        contact_damping
            Must match the native reference run's values. Schema-v2 artefacts
            embed them (self-configured, mismatching explicit values raise);
            for v1 artefacts pass them by hand.
        """
        super().__init__()
        import shapely

        self.policy = policy
        self.obs_config, self.action_config = resolve_configs(policy, obs_config, action_config)
        dynamics = resolve_dynamics(
            policy,
            {
                "desired_velocity_weight": desired_velocity_weight,
                "max_velocity_magnitude": max_velocity_magnitude,
                "contact_stiffness": contact_stiffness,
                "contact_damping": contact_damping,
            },
        )
        self.desired_velocity_weight = dynamics["desired_velocity_weight"]
        self.max_velocity_magnitude = dynamics["max_velocity_magnitude"]
        self.contact_stiffness = dynamics["contact_stiffness"]
        self.contact_damping = dynamics["contact_damping"]

        self.walkable_geometry = walkable_geometry
        self.wall_segments = extract_wall_segments(walkable_geometry)
        try:
            from crowdrl_core.geometry import build_navmesh

            self.navmesh = build_navmesh(walkable_geometry)
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError(
                "LockstepPolicyModel needs the navmesh (funnel waypoints + "
                "path deviation). Install the crowdrl-core[geometry] extra."
            ) from exc
        self.exit_polygons = [
            geom if isinstance(geom, shapely.Polygon) else shapely.Polygon(geom)
            for geom in exit_geometries
        ]
        minx, miny, maxx, maxy = walkable_geometry.bounds
        self._roster_radius = 2.0 * float(np.hypot(maxx - minx, maxy - miny))

        self._buf = self.obs_config.temporal_memory_window + 1
        self._rows: dict[int, _Row] = {}
        self._frozen: dict[int, _Row] = {}
        self.exit_steps: dict[int, int] = {}
        """JuPedSim agent id -> native-semantics exit step (landing step)."""
        self._served: set[int] = set()
        self._pass_count = 0
        self._dt_checked = False

    # -- batch pass -----------------------------------------------------------

    def _world_from(self, rows: list[_Row]) -> WorldState:
        n = len(rows)
        world = WorldState(
            positions=np.stack([r.position for r in rows]),
            velocities=np.stack([r.velocity for r in rows]),
            torso_orientations=np.array([r.torso for r in rows]),
            head_orientations=np.array([r.head for r in rows]),
            shoulder_widths=np.array([r.shoulder for r in rows]),
            chest_depths=np.array([r.chest for r in rows]),
            masses=np.array([r.mass for r in rows]),
            goal_positions=np.stack([r.goal for r in rows]),
            walkable_polygon=self.walkable_geometry,
            wall_segments=self.wall_segments,
            navmesh=self.navmesh,
            active_mask=np.ones(n, dtype=np.bool_),
        )
        world.preferred_speeds = np.array([r.preferred_speed for r in rows])
        if self.obs_config.use_temporal_memory:
            world.spawn_positions = np.stack([r.spawn for r in rows])
            world.initial_goal_distances = np.array([r.init_gdist for r in rows])
            world.cumulative_path_length = np.array([r.cum_path for r in rows])
            world.pos_history = np.stack([r.pos_hist for r in rows])
            world.gdist_history = np.stack([r.gdist_hist for r in rows])
            world.step_count = self._pass_count
        return world

    def _policy_batch(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        session = getattr(self.policy, "_session", None)
        if session is not None:
            out = session.run(None, {"observations": obs.astype(np.float32)})[0]
            return np.asarray(out, dtype=np.float64)
        return np.stack(
            [np.asarray(self.policy(obs[i]), dtype=np.float64) for i in range(len(obs))]
        )

    def _run_pass(self, ped, env_query, dt: float) -> None:
        neighbors = env_query.other_agents_in_range(ped, self._roster_radius)
        transients = {ped.id: ped, **{n.id: n for n in neighbors}}
        roster = sorted(aid for aid in transients if aid not in self._frozen)

        for gone in set(self._rows) - set(roster):
            del self._rows[gone]
        for aid in roster:
            if aid not in self._rows:
                t = transients[aid]
                self._rows[aid] = _Row(t.model, t.final_target, self._buf)
        if not roster:
            return

        rows = [self._rows[aid] for aid in roster]
        world = self._world_from(rows)
        prev_positions = world.positions.copy()

        batch = native_batch_step(
            world,
            self._policy_batch,
            self.obs_config,
            self.action_config,
            desired_velocity_weight=self.desired_velocity_weight,
            max_velocity_magnitude=self.max_velocity_magnitude,
            contact_stiffness=self.contact_stiffness,
            contact_damping=self.contact_damping,
            dt=dt,
        )

        self._pass_count += 1
        write_idx = (self._pass_count - 1) % self._buf
        deltas = np.linalg.norm(world.positions - prev_positions, axis=1)
        gdists = np.linalg.norm(world.goal_positions - world.positions, axis=1)
        import shapely

        for i, (aid, row) in enumerate(zip(roster, rows)):
            row.position = world.positions[i].copy()
            row.velocity = world.velocities[i].copy()
            row.heading = float(batch.new_headings[i])
            row.torso = float(world.torso_orientations[i])
            row.head = float(world.head_orientations[i])
            row.cum_path += float(deltas[i])
            row.pos_hist = row.pos_hist.copy()
            row.pos_hist[write_idx] = world.positions[i]
            row.gdist_hist = row.gdist_hist.copy()
            row.gdist_hist[write_idx] = gdists[i]

            point = shapely.Point(row.position)
            if any(poly.contains(point) for poly in self.exit_polygons):
                self._frozen[aid] = row
                self.exit_steps[aid] = self._pass_count
                del self._rows[aid]

    # -- JuPedSim operational-model interface ---------------------------------

    def compute_next_state(self, dt, ped, env_query) -> CrowdRLAgentState:
        if not self._dt_checked:
            self._dt_checked = True
            if abs(float(dt) - self.action_config.dt) > 1e-12:
                warnings.warn(
                    f"simulation dt={dt} differs from the trained "
                    f"ActionConfig.dt={self.action_config.dt}: per-step action "
                    "limits rescale the whole motion envelope.",
                    UserWarning,
                    stacklevel=2,
                )
        if ped.id in self._served:
            self._served.clear()
        if not self._served:
            self._run_pass(ped, env_query, float(dt))
        self._served.add(ped.id)
        row = self._rows.get(ped.id) or self._frozen[ped.id]
        return row.as_state()
