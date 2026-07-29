"""LearnedPolicyModel -- a CrowdRL policy as a JuPedSim 2.0 operational model.

JuPedSim 2.0 lets an operational model be written in pure Python by subclassing
``CustomOperationalModel``. The framework calls ``compute_next_state`` once per
agent per iteration, in a compute-then-apply pass, and applies the position on
the returned state **verbatim**.

That makes the division of labour sharp (see Project Plan v8, Section 3.6):

* JuPedSim supplies the walkable geometry and wall segments (through the
  per-step ``EnvironmentQuery``), the agent's final goal ``ped.final_target``
  and its routed next waypoint ``ped.next_target`` (the strategical + tactical
  systems run *before* the operational step), neighbour queries, and agent
  lifecycle.
* JuPedSim performs **no** velocity integration, **no** boundary clamping and
  **no** collision resolution for the operational layer. ``GenericAgent`` has no
  velocity or orientation field at all.

So this class owns the entire state transition: sensing, WorldState assembly,
observation construction (via the *same* crowdrl-core builder used in training),
policy inference, action interpretation, and integration to a new position.
Everything an agent can do lives here.

Per-agent state is an arbitrary immutable Python object, which is what lets the
policy's torso/head orientation and body dimensions be first-class -- no
side-channel bookkeeping and no JuPedSim core changes.

Raycasting is wired, and matters -- every policy trained so far uses it. Wall
segments come from the JuPedSim geometry and neighbouring agents are present in
the assembled WorldState, so rays intersect both, exactly as in training. The
query radii are derived from ``ObsConfig.raycast.max_range`` so nothing inside
ray range is ever missing from the world the policy sees.

Navmesh (route-waypoint) and temporal-memory observation blocks are wired, and
contact physics (agent-agent contact forces + body-clearance wall projection,
the same crowdrl-core functions the training env runs) is active whenever
``walkable_geometry`` is passed -- without it the model falls back to a coarse
centre-point containment check and warns.

Scope note: the neighbour-memory observation blocks (A+/A++,
``use_neighbor_vel_history`` / ``use_neighbor_trajectory_features``) are not
yet wired; observation parity is only guaranteed for configs that leave those
disabled. See the obs-parity harness.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from crowdrl_core.action import ActionConfig, interpret_action
from crowdrl_core.collision import compute_contact_forces, enforce_wall_boundaries
from crowdrl_core.observation import ObsConfig, build_observation
from crowdrl_core.world_state import WorldState

from .policy import Policy, resolve_configs

if TYPE_CHECKING:  # pragma: no cover
    from shapely import Polygon

try:  # pragma: no cover - import shape depends on how jupedsim 2.0 is provided
    from jupedsim.models.custom_model import CustomOperationalModel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "crowdrl-jupedsim requires JuPedSim 2.0 (the CustomOperationalModel layer). "
        "It is not published to PyPI; build it from source and put the built "
        "extension and python_modules/jupedsim on sys.path. See this package's "
        "pyproject.toml for the wiring."
    ) from exc


@dataclass(frozen=True)
class TemporalMemory:
    """Per-agent trajectory memory backing the temporal observation block.

    Mirrors the state CrowdEnv keeps per agent (spawn, initial goal distance,
    cumulative path, position/goal-distance ring buffers, step count) with the
    exact same ring-buffer contract: all slots initialised to the spawn
    values, the post-step position written at index ``step_count % buf_size``
    (pre-step count), then the count incremented.

    The ndarray fields are treated as immutable by convention: every update
    copies before writing (see ``LearnedPolicyModel._advance_memory``),
    because this object is shared live with the running simulation.
    """

    spawn_position: tuple[float, float]
    initial_goal_distance: float
    cumulative_path_length: float
    step_count: int
    pos_history: NDArray[np.float64]
    """(buf_size, 2) ring buffer of post-step positions."""
    gdist_history: NDArray[np.float64]
    """(buf_size,) ring buffer of post-step goal distances."""


@dataclass(frozen=True, kw_only=True)
class CrowdRLAgentState:
    """Per-agent state for a CrowdRL-driven JuPedSim agent.

    Satisfies JuPedSim's ``CustomModelAgentState`` protocol by exposing
    ``position``. Everything else is state JuPedSim does not model but the
    learned policy requires.

    Frozen on purpose: JuPedSim shares this object live with the running
    simulation during the compute phase, so in-place mutation would corrupt the
    compute-then-apply ordering. Updates must go through
    ``dataclasses.replace``.
    """

    position: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)

    heading: float = 0.0
    """Direction of travel (radians). Distinct from torso_angle."""

    torso_angle: float = 0.0
    """Torso orientation (radians); sets the collision-ellipse axis."""

    head_angle: float = 0.0
    """Absolute head orientation (radians). Raycasts follow the head."""

    shoulder_width: float = 0.225
    """Half-width of the collision ellipse, torso-perpendicular (m)."""

    chest_depth: float = 0.15
    """Half-depth of the collision ellipse, torso-forward (m)."""

    mass: float = 80.0
    preferred_speed: float = 1.34

    memory: TemporalMemory | None = None
    """Trajectory memory for the temporal observation block. Leave None at
    ``add_agent``: the spawn position and routed goal are only knowable after
    the first routing pass, so the model initialises it lazily on the agent's
    first ``compute_next_state`` (when ``ObsConfig.use_temporal_memory``)."""


class LearnedPolicyModel(CustomOperationalModel):
    """Drives JuPedSim agents with an exported CrowdRL policy."""

    def __init__(
        self,
        policy: Policy,
        *,
        obs_config: ObsConfig | None = None,
        action_config: ActionConfig | None = None,
        walkable_geometry: Polygon | None = None,
        desired_velocity_weight: float = 0.05,
        max_velocity_magnitude: float = 5.0,
        contact_stiffness: float = 30000.0,
        contact_damping: float = 500.0,
        neighbor_radius: float | None = None,
        wall_query_radius: float | None = None,
        keep_inside_geometry: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        policy
            Maps one observation vector to a raw 4D action in [-1, 1].
        obs_config
            Normally omitted: artefacts exported since issue #7 embed the
            training config, and the model self-configures from it. When
            given, it is cross-checked field-for-field against the embedded
            record and any disagreement raises. Required (with a warning)
            only for legacy artefacts without metadata -- rebuild it from the
            run's ``config_resolved.yaml``.
        action_config
            Action limits used during training; same resolution rules as
            ``obs_config``.
        walkable_geometry
            The same shapely Polygon passed to ``jps.Simulation(geometry=...)``.
            When given, the model runs the training env's contact physics each
            step: agent-agent contact forces and body-clearance wall
            projection (``enforce_wall_boundaries``; clearance =
            max(shoulder_width, chest_depth) per agent) -- the exact
            crowdrl-core functions and ordering used in training. Without it,
            only a coarse centre-point containment check runs, agent bodies
            can clip walls and interpenetrate, and construction warns.
        desired_velocity_weight
            First-order velocity filter weight, mirroring
            ``CrowdEnvConfig.desired_velocity_weight``:
            ``v_new = w * v_desired + (1 - w) * v_old``.
        neighbor_radius
            Radius for the neighbour query. Defaults to the larger of
            ``obs_config.neighbor_sensing_radius`` and the raycast range, because
            rays intersect agents as well as walls. Must be >= the horizon used
            in training, or the K-nearest set differs.
        wall_query_radius
            Radius for the wall-segment query. Defaults to the raycast range.
        keep_inside_geometry
            Fallback containment when ``walkable_geometry`` is NOT given: if
            the integrated position leaves the walkable area, hold position
            instead. JuPedSim applies the returned position verbatim and a
            point outside the walkable area crashes the next iteration, so the
            model is responsible for containment. Superseded by the
            body-clearance projection when ``walkable_geometry`` is provided.
        """
        super().__init__()
        self.policy = policy
        # Issue #7: no silent ObsConfig() fallback. Self-configure from the
        # artefact's embedded metadata, cross-check anything explicit, refuse
        # legacy artefacts without explicit configs.
        self.obs_config, self.action_config = resolve_configs(policy, obs_config, action_config)
        self.walkable_geometry = walkable_geometry
        if walkable_geometry is None:
            warnings.warn(
                "LearnedPolicyModel constructed without walkable_geometry: "
                "contact forces and body-clearance wall projection are "
                "DISABLED, so agent bodies can clip walls and interpenetrate "
                "(the training env enforces both). Pass the same shapely "
                "Polygon given to jps.Simulation(geometry=...).",
                UserWarning,
                stacklevel=2,
            )
        self.desired_velocity_weight = float(desired_velocity_weight)
        self.max_velocity_magnitude = float(max_velocity_magnitude)
        self.contact_stiffness = float(contact_stiffness)
        self.contact_damping = float(contact_damping)
        # Rays intersect walls *and* other agents, so the neighbour query has to
        # cover the raycast horizon as well as the social one. If it did not,
        # an agent sitting beyond the social radius but inside ray range would
        # be missing from the WorldState and its rays would report phantom-clear
        # space.
        ray_range = float(self.obs_config.raycast.max_range)
        self.neighbor_radius = (
            float(neighbor_radius)
            if neighbor_radius is not None
            else max(float(self.obs_config.neighbor_sensing_radius), ray_range)
        )
        # A wall segment farther from the agent than the ray length cannot be
        # hit by any ray, so the ray range is exactly the right query radius.
        self.wall_query_radius = (
            float(wall_query_radius) if wall_query_radius is not None else ray_range
        )
        self.keep_inside_geometry = keep_inside_geometry

    # -- temporal memory ------------------------------------------------------

    def _ensure_memory(self, state: CrowdRLAgentState, goal) -> CrowdRLAgentState:
        """Initialise trajectory memory on first contact, mirroring CrowdEnv
        reset: every ring-buffer slot pre-filled with the spawn value."""
        if not self.obs_config.use_temporal_memory or state.memory is not None:
            return state
        buf_size = self.obs_config.temporal_memory_window + 1
        pos = np.asarray(state.position, dtype=np.float64)
        gdist = float(np.linalg.norm(np.asarray(goal, dtype=np.float64) - pos))
        memory = TemporalMemory(
            spawn_position=(float(pos[0]), float(pos[1])),
            initial_goal_distance=gdist,
            cumulative_path_length=0.0,
            step_count=0,
            pos_history=np.tile(pos, (buf_size, 1)),
            gdist_history=np.full(buf_size, gdist, dtype=np.float64),
        )
        return replace(state, memory=memory)

    def _advance_memory(
        self,
        memory: TemporalMemory,
        old_position: NDArray[np.float64],
        new_position: NDArray[np.float64],
        goal,
    ) -> TemporalMemory:
        """One step of the CrowdEnv ring-buffer contract: write the post-step
        position/goal-distance at the pre-step count's slot, then increment."""
        buf_size = self.obs_config.temporal_memory_window + 1
        write_idx = memory.step_count % buf_size
        pos_history = memory.pos_history.copy()
        pos_history[write_idx] = new_position
        gdist_history = memory.gdist_history.copy()
        gdist_history[write_idx] = np.linalg.norm(
            np.asarray(goal, dtype=np.float64) - new_position
        )
        return TemporalMemory(
            spawn_position=memory.spawn_position,
            initial_goal_distance=memory.initial_goal_distance,
            cumulative_path_length=memory.cumulative_path_length
            + float(np.linalg.norm(new_position - old_position)),
            step_count=memory.step_count + 1,
            pos_history=pos_history,
            gdist_history=gdist_history,
        )

    # -- observation assembly -------------------------------------------------

    def _walls(self, env_query, position: tuple[float, float]) -> NDArray[np.float64]:
        """Wall segments near ``position`` as a crowdrl-core (S, 2, 2) array."""
        segments = env_query.line_segments_in_range(position, self.wall_query_radius)
        if not segments:
            return np.zeros((0, 2, 2), dtype=np.float64)
        return np.array([[ls.p1, ls.p2] for ls in segments], dtype=np.float64)

    def build_world_state(self, ped, env_query) -> WorldState:
        """Assemble a WorldState with the ego agent at index 0.

        This is the deployment half of the transfer guarantee: the observation
        builder consumes only WorldState and never learns which engine filled
        it in.
        """
        state = ped.model
        # other_agents_in_range excludes the querying agent by contract.
        neighbors = env_query.other_agents_in_range(ped, self.neighbor_radius)

        agents = [state] + [n.model for n in neighbors]
        # goal_positions carries each agent's FINAL goal, matching the training
        # env's episode goal. The routed next waypoint (ped.next_target) is a
        # separate signal and feeds the navmesh observation block instead.
        goals = [ped.final_target] + [n.final_target for n in neighbors]

        extra: dict = {}
        if self.obs_config.use_temporal_memory:
            ego_memory = self._ensure_memory(state, ped.final_target).memory
            n = len(agents)
            buf_size = self.obs_config.temporal_memory_window + 1
            spawn = np.zeros((n, 2), dtype=np.float64)
            init_g = np.zeros(n, dtype=np.float64)
            cum = np.zeros(n, dtype=np.float64)
            pos_h = np.zeros((n, buf_size, 2), dtype=np.float64)
            g_h = np.zeros((n, buf_size), dtype=np.float64)
            # The ego row (index 0) is the one the observation reads. Neighbour
            # rows are copied from their own memory when initialised; note the
            # world-level step_count below is the EGO's, so neighbour buffers
            # written at different agent ages would misindex -- acceptable while
            # only the ego's temporal block reads them, revisit before wiring
            # use_neighbor_trajectory_features.
            rows = [ego_memory] + [nb.model.memory for nb in neighbors]
            for i, mem in enumerate(rows):
                if mem is None:
                    continue
                spawn[i] = mem.spawn_position
                init_g[i] = mem.initial_goal_distance
                cum[i] = mem.cumulative_path_length
                pos_h[i] = mem.pos_history
                g_h[i] = mem.gdist_history
            extra = {
                "spawn_positions": spawn,
                "initial_goal_distances": init_g,
                "cumulative_path_length": cum,
                "pos_history": pos_h,
                "gdist_history": g_h,
                "step_count": ego_memory.step_count if ego_memory is not None else 0,
            }

        return WorldState(
            positions=np.array([a.position for a in agents], dtype=np.float64),
            velocities=np.array([a.velocity for a in agents], dtype=np.float64),
            torso_orientations=np.array([a.torso_angle for a in agents], dtype=np.float64),
            head_orientations=np.array([a.head_angle for a in agents], dtype=np.float64),
            shoulder_widths=np.array([a.shoulder_width for a in agents], dtype=np.float64),
            chest_depths=np.array([a.chest_depth for a in agents], dtype=np.float64),
            masses=np.array([a.mass for a in agents], dtype=np.float64),
            goal_positions=np.array(goals, dtype=np.float64),
            preferred_speeds=np.array([a.preferred_speed for a in agents], dtype=np.float64),
            wall_segments=self._walls(env_query, ped.position),
            # The router's next waypoint drives the navmesh observation block
            # (waypoint direction + path_deviation=0.0, the single-waypoint
            # contract). Populated for every agent so the block is available
            # regardless of which index is queried.
            route_next_waypoints=np.array(
                [ped.next_target] + [n.next_target for n in neighbors], dtype=np.float64
            ),
            **extra,
        )

    # -- JuPedSim operational-model interface ---------------------------------

    def compute_next_state(self, dt, ped, env_query) -> CrowdRLAgentState:
        # Lazy-init trajectory memory (no-op unless use_temporal_memory); the
        # observation this step reads the PRE-step memory, the returned state
        # carries the advanced one.
        state = self._ensure_memory(ped.model, ped.final_target)

        world = self.build_world_state(ped, env_query)
        obs = build_observation(world, 0, self.obs_config)
        raw_action = self.policy(obs)

        velocity = np.asarray(state.velocity, dtype=np.float64)
        result = interpret_action(
            np.asarray(raw_action, dtype=np.float64),
            current_heading=state.heading,
            current_torso=state.torso_angle,
            current_head=state.head_angle,
            config=self.action_config,
            current_speed=float(np.linalg.norm(velocity)),
        )

        # Integration mirrors CrowdEnv.step, in the exact training order:
        # velocity filter -> contact accelerations (pre-integration positions,
        # post-filter velocities) -> magnitude clamp -> semi-implicit Euler ->
        # body-clearance wall projection. The assembled ``world`` is transient
        # and adapter-owned, so mutating the ego row is safe. One deviation is
        # inherent to the per-agent compute-then-apply contract: neighbours are
        # at their PRE-step positions when the ego's contact forces are
        # computed, whereas training computes all forces from one synchronized
        # snapshot -- a single-step staleness, not a systematic bias.
        w = self.desired_velocity_weight
        new_velocity = w * result.desired_velocity + (1.0 - w) * velocity

        physics = self.walkable_geometry is not None
        if physics:
            world.velocities[0] = new_velocity
            accel = compute_contact_forces(
                world, stiffness=self.contact_stiffness, damping=self.contact_damping
            )[0]
            new_velocity = new_velocity + accel * dt

        speed = float(np.linalg.norm(new_velocity))
        if speed > self.max_velocity_magnitude:
            new_velocity = new_velocity * (self.max_velocity_magnitude / max(speed, 1e-10))

        position = np.asarray(state.position, dtype=np.float64)
        new_position = position + new_velocity * dt

        if physics:
            world.positions[0] = new_position
            world.velocities[0] = new_velocity
            world.walkable_polygon = self.walkable_geometry
            enforce_wall_boundaries(world)  # corrects the ego row in place
            new_position = world.positions[0]
            new_velocity = world.velocities[0]
        elif self.keep_inside_geometry and not env_query.inside_geometry(
            (float(new_position[0]), float(new_position[1]))
        ):
            new_position = position
            new_velocity = np.zeros(2, dtype=np.float64)

        # Advance memory on the FINAL position (post-containment), matching
        # the training env, which records post-step, post-collision positions.
        memory = state.memory
        if memory is not None:
            memory = self._advance_memory(memory, position, new_position, ped.final_target)

        return replace(
            state,
            position=(float(new_position[0]), float(new_position[1])),
            velocity=(float(new_velocity[0]), float(new_velocity[1])),
            heading=result.new_heading,
            torso_angle=result.new_torso_orientation,
            head_angle=result.new_head_orientation,
            memory=memory,
        )
