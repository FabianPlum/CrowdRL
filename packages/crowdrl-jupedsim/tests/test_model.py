"""Tests for the LearnedPolicyModel JuPedSim adapter.

These require a JuPedSim 2.0 source build (the CustomOperationalModel layer),
which is not published to PyPI and is therefore absent in CI. The whole module
skips cleanly when it is unavailable.
"""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

pytest.importorskip("jupedsim", reason="requires a JuPedSim 2.0 source build")
pytest.importorskip(
    "jupedsim.models.custom_model",
    reason="requires the JuPedSim 2.0 CustomOperationalModel layer",
)

import shapely  # noqa: E402

import jupedsim as jps  # noqa: E402
from crowdrl_core.action import ActionConfig  # noqa: E402
from crowdrl_core.observation import ObsConfig, build_observation  # noqa: E402
from crowdrl_core.sensing import RaycastConfig  # noqa: E402
from crowdrl_jupedsim import (  # noqa: E402
    ConstantPolicy,
    CrowdRLAgentState,
    LearnedPolicyModel,
)

# Drive straight ahead at max forward speed, no turning.
FORWARD = [1.0, 0.0, 0.0, 0.0]


def _model(**kwargs) -> LearnedPolicyModel:
    """LearnedPolicyModel over a ConstantPolicy with explicit default configs.

    ConstantPolicy carries no embedded config metadata and issue #7 removed
    the silent ObsConfig()/ActionConfig() fallback, so the configs these tests
    always ran with are now stated explicitly. walkable_geometry defaults to
    the same room the sim fixture uses, turning on training-parity contact
    physics (pass walkable_geometry=None explicitly to exercise the fallback).
    """
    kwargs.setdefault("obs_config", ObsConfig())
    kwargs.setdefault("action_config", ActionConfig())
    kwargs.setdefault("walkable_geometry", _room())
    return LearnedPolicyModel(ConstantPolicy(FORWARD), **kwargs)


def _room() -> shapely.Polygon:
    return shapely.Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])


def _sim(model, dt: float = 0.01):
    # dt must match CrowdEnvConfig.dt (0.01), which is also JuPedSim's default.
    # The action limits are applied per *step*, not per second -- e.g.
    # max_heading_change is 0.020 rad/step, which is 115 deg/s at dt=0.01 but
    # only 23 deg/s at dt=0.05. Running the adapter at a different dt therefore
    # silently rescales the whole motion envelope away from what the policy was
    # trained under.
    sim = jps.Simulation(model=model, geometry=_room(), dt=dt)
    exit_id = sim.add_exit_stage([(19, 9), (19, 11), (20, 11), (20, 9)])
    journey_id = sim.add_journey(jps.JourneyDescription([exit_id]))
    return sim, exit_id, journey_id


class _StubAgent:
    """Minimal stand-in for a JuPedSim transient agent view."""

    def __init__(self, agent_id, state, target):
        self.id = agent_id
        self.model = state
        self.final_target = target
        self.next_target = target  # open room: the routed waypoint IS the goal

    @property
    def position(self):
        return self.model.position


class _StubEnvQuery:
    """Stand-in for jupedsim's per-step EnvironmentQuery.

    Mirrors the 2.0 contract: ``other_agents_in_range`` excludes the querying
    agent itself.
    """

    def __init__(self, agents=(), walls=()):
        self._agents = list(agents)
        self._walls = list(walls)

    def other_agents_in_range(self, agent, radius):
        return [a for a in self._agents if a.id != agent.id]

    def line_segments_in_range(self, position, distance):
        return self._walls

    def inside_geometry(self, position):
        return True


class TestCrowdRLAgentState:
    def test_satisfies_position_protocol(self):
        state = CrowdRLAgentState(position=(1.0, 2.0))
        assert state.position == (1.0, 2.0)

    def test_is_frozen(self):
        """Frozen matters: JuPedSim shares the state live during the compute
        phase, so in-place mutation would break compute-then-apply ordering."""
        state = CrowdRLAgentState(position=(1.0, 2.0))
        with pytest.raises(FrozenInstanceError):
            state.position = (3.0, 4.0)

    def test_replace_produces_a_new_instance(self):
        state = CrowdRLAgentState(position=(1.0, 2.0))
        updated = replace(state, position=(3.0, 4.0))
        assert updated is not state
        assert state.position == (1.0, 2.0)


class TestWorldStateAssembly:
    def test_ego_is_index_zero_and_self_is_excluded(self):
        model = _model()

        ego_state = CrowdRLAgentState(position=(5.0, 5.0), velocity=(1.0, 0.0))
        ego = _StubAgent(0, ego_state, target=(19.0, 10.0))
        neighbor = _StubAgent(1, CrowdRLAgentState(position=(6.0, 5.0)), target=(19.0, 10.0))
        # other_agents_in_range excludes the querying agent by contract; the
        # stub mirrors that, so listing the ego here must not duplicate it.
        env_query = _StubEnvQuery(agents=[ego, neighbor])

        world = model.build_world_state(ego, env_query)

        assert world.n_agents == 2, "ego + 1 neighbour, self-match removed"
        np.testing.assert_allclose(world.positions[0], [5.0, 5.0])
        np.testing.assert_allclose(world.positions[1], [6.0, 5.0])
        np.testing.assert_allclose(world.goal_positions[0], [19.0, 10.0])
        world.validate()

    def test_wall_segments_are_shaped_for_crowdrl_core(self):
        model = _model()
        ego = _StubAgent(0, CrowdRLAgentState(position=(5.0, 5.0)), target=(19.0, 10.0))
        world = model.build_world_state(ego, _StubEnvQuery())
        assert world.wall_segments.shape == (0, 2, 2)


class TestSimulationIntegration:
    def test_agent_advances_toward_the_routing_target(self):
        model = _model()
        sim, exit_id, journey_id = _sim(model)
        start = (2.0, 10.0)
        agent_id = sim.add_agent(
            journey_id=journey_id,
            stage_id=exit_id,
            state=CrowdRLAgentState(position=start),
        )

        for _ in range(100):
            sim.iterate()

        assert sim.agent(agent_id).position[0] > start[0], (
            "agent should advance in +x toward the exit"
        )

    def test_agent_reaches_the_exit(self):
        model = _model()
        sim, exit_id, journey_id = _sim(model)
        sim.add_agent(
            journey_id=journey_id,
            stage_id=exit_id,
            state=CrowdRLAgentState(position=(2.0, 10.0)),
        )

        for _ in range(2000):
            if sim.agent_count() == 0:
                break
            sim.iterate()

        assert sim.agent_count() == 0, "agent should reach the exit and be removed"

    def test_custom_state_fields_survive_iterations(self):
        """Guards the dataclasses.replace contract: compute_next_state must
        carry state forward, not rebuild it from defaults. A silent reset here
        would corrupt the policy's observations without ever crashing."""
        model = _model()
        sim, exit_id, journey_id = _sim(model)
        agent_id = sim.add_agent(
            journey_id=journey_id,
            stage_id=exit_id,
            state=CrowdRLAgentState(
                position=(2.0, 10.0),
                shoulder_width=0.31,
                chest_depth=0.19,
                preferred_speed=1.71,
                mass=93.0,
            ),
        )

        for _ in range(25):
            sim.iterate()

        state = sim.agent(agent_id).model
        assert isinstance(state, CrowdRLAgentState)
        assert state.shoulder_width == 0.31
        assert state.chest_depth == 0.19
        assert state.preferred_speed == 1.71
        assert state.mass == 93.0
        assert state.position == pytest.approx(sim.agent(agent_id).position)

    def test_velocity_filter_ramps_rather_than_jumping(self):
        """v_new = w*v_desired + (1-w)*v_old, mirroring CrowdEnv.step -- so the
        first step must not reach max forward speed immediately."""
        model = _model(desired_velocity_weight=0.05)
        sim, exit_id, journey_id = _sim(model)
        agent_id = sim.add_agent(
            journey_id=journey_id,
            stage_id=exit_id,
            state=CrowdRLAgentState(position=(2.0, 10.0)),
        )

        sim.iterate()
        speed_after_one = float(np.linalg.norm(sim.agent(agent_id).model.velocity))

        assert 0.0 < speed_after_one < 0.5, (
            f"expected a filtered ramp, got {speed_after_one:.3f} m/s"
        )

    def test_agent_is_kept_inside_the_walkable_area(self):
        """JuPedSim applies the returned position verbatim and crashes on the
        next iteration if it is outside the walkable area, so containment is
        the model's responsibility."""
        model = _model()
        sim, exit_id, journey_id = _sim(model)
        # Head straight at the right wall, away from the exit slot.
        agent_id = sim.add_agent(
            journey_id=journey_id,
            stage_id=exit_id,
            state=CrowdRLAgentState(position=(18.0, 3.0)),
        )

        for _ in range(400):
            if sim.agent_count() == 0:
                break
            sim.iterate()
            if sim.agent_count() > 0:
                x, y = sim.agent(agent_id).position
                assert 0.0 <= x <= 20.0 and 0.0 <= y <= 20.0, (
                    f"agent left the walkable area at ({x}, {y})"
                )


class TestContactPhysics:
    """Training-parity contact physics: the crowd_env step's contact forces +
    body-clearance wall projection, active when walkable_geometry is given."""

    def test_construction_without_geometry_warns(self):
        with pytest.warns(UserWarning, match="walkable_geometry"):
            _model(walkable_geometry=None)

    def test_wall_clearance_is_enforced(self):
        """An agent driving straight at a wall must be held one body radius
        (max(shoulder, chest) = 0.225 m default) off the boundary -- not just
        its centre point inside, which was the pre-physics behaviour."""
        room = _room()
        model = _model()
        sim, exit_id, journey_id = _sim(model)
        # Spawn near the right wall, heading +x, away from the exit slot.
        agent_id = sim.add_agent(
            journey_id=journey_id,
            stage_id=exit_id,
            state=CrowdRLAgentState(position=(18.0, 3.0)),
        )

        min_boundary_dist = np.inf
        for _ in range(400):
            if sim.agent_count() == 0:
                break
            sim.iterate()
            if sim.agent_count() > 0:
                point = shapely.Point(sim.agent(agent_id).position)
                min_boundary_dist = min(min_boundary_dist, room.boundary.distance(point))

        assert min_boundary_dist >= 0.225 - 1e-6, (
            f"agent centre came within {min_boundary_dist:.3f} m of the wall; "
            "body clearance (0.225 m) must hold"
        )

    def test_head_on_agents_do_not_pass_through(self):
        """Two agents driven straight at each other. Without contact forces
        they ghost through one another; the spring-damper contact must keep
        their centres apart at roughly body scale."""
        model = _model()
        sim, exit_id, journey_id = _sim(model)
        a = sim.add_agent(
            journey_id=journey_id,
            stage_id=exit_id,
            state=CrowdRLAgentState(position=(8.0, 10.0), heading=0.0, torso_angle=0.0),
        )
        b = sim.add_agent(
            journey_id=journey_id,
            stage_id=exit_id,
            state=CrowdRLAgentState(position=(12.0, 10.0), heading=np.pi, torso_angle=np.pi),
        )

        min_dist = np.inf
        for _ in range(600):
            sim.iterate()
            pa = np.asarray(sim.agent(a).position)
            pb = np.asarray(sim.agent(b).position)
            min_dist = min(min_dist, float(np.linalg.norm(pa - pb)))

        # Head-on ellipse contact is at 2 x chest_depth = 0.30 m; the spring
        # is soft, so allow transient compression but nothing like the
        # pre-physics pass-through (which reached ~0).
        assert min_dist > 0.2, f"agents interpenetrated: min centre distance {min_dist:.3f} m"


class TestTemporalMemory:
    """The adapter must replicate CrowdEnv's ring-buffer contract exactly --
    an off-by-one here corrupts the policy's time-pressure and progress
    signals without ever crashing."""

    CFG = ObsConfig(use_temporal_memory=True, temporal_memory_window=4)  # buf_size 5

    def test_lazy_init_mirrors_crowdenv_reset(self):
        model = _model(obs_config=self.CFG)
        state = CrowdRLAgentState(position=(2.0, 10.0))
        mem = model._ensure_memory(state, (19.0, 10.0)).memory

        assert mem.step_count == 0
        assert mem.spawn_position == (2.0, 10.0)
        assert mem.initial_goal_distance == pytest.approx(17.0)
        assert mem.cumulative_path_length == 0.0
        np.testing.assert_allclose(mem.pos_history, np.tile([2.0, 10.0], (5, 1)))
        np.testing.assert_allclose(mem.gdist_history, np.full(5, 17.0))

    def test_advance_writes_post_step_position_at_pre_step_slot(self):
        model = _model(obs_config=self.CFG)
        state = model._ensure_memory(CrowdRLAgentState(position=(2.0, 10.0)), (19.0, 10.0))
        mem = model._advance_memory(
            state.memory, np.array([2.0, 10.0]), np.array([2.5, 10.0]), (19.0, 10.0)
        )

        assert mem.step_count == 1
        np.testing.assert_allclose(mem.pos_history[0], [2.5, 10.0])  # write_idx = 0
        assert mem.gdist_history[0] == pytest.approx(16.5)
        assert mem.cumulative_path_length == pytest.approx(0.5)
        # Untouched slots keep the spawn fill; the source memory is unchanged
        # (copy-on-write -- it is shared live with the simulation).
        np.testing.assert_allclose(mem.pos_history[1], [2.0, 10.0])
        np.testing.assert_allclose(state.memory.pos_history[0], [2.0, 10.0])

    def test_simulation_accumulates_memory(self):
        model = _model(obs_config=self.CFG)
        sim, exit_id, journey_id = _sim(model)
        agent_id = sim.add_agent(
            journey_id=journey_id,
            stage_id=exit_id,
            state=CrowdRLAgentState(position=(2.0, 10.0)),
        )

        positions = []
        for _ in range(3):
            sim.iterate()
            positions.append(np.asarray(sim.agent(agent_id).model.position))

        mem = sim.agent(agent_id).model.memory
        assert mem is not None and mem.step_count == 3
        for k in range(3):
            np.testing.assert_allclose(mem.pos_history[k], positions[k], atol=1e-12)
        assert mem.spawn_position == (2.0, 10.0)
        assert mem.cumulative_path_length == pytest.approx(
            sum(
                float(np.linalg.norm(b - a))
                for a, b in zip([np.array([2.0, 10.0])] + positions[:-1], positions)
            )
        )

    def test_deployment_obs_width_is_89d(self):
        """The current best checkpoints' shape: nogoaldir + navmesh + temporal.
        All three optional blocks must materialise from adapter-supplied state."""
        cfg = ObsConfig(use_navmesh=True, use_goal_direction=False, use_temporal_memory=True)
        model = _model(obs_config=cfg)
        ego = _StubAgent(0, CrowdRLAgentState(position=(5.0, 5.0)), target=(19.0, 5.0))

        world = model.build_world_state(ego, _StubEnvQuery())
        obs = build_observation(world, 0, cfg)

        assert cfg.obs_dim == 89
        assert obs.shape == (89,)


class _StubWall:
    """Stand-in for a JuPedSim LineSegment (exposes p1 / p2 endpoints)."""

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2


class TestRaycasting:
    """Every trained policy uses raycasts, so a silently-blind sensor would
    produce plausible-looking but wrong actions. These assert rays actually
    respond to geometry and to other agents."""

    @staticmethod
    def _rays(model, world):
        obs = build_observation(world, 0, model.obs_config)
        return obs[-model.obs_config.raycast.n_rays :]

    def test_open_space_reads_clear(self):
        model = _model()
        ego = _StubAgent(
            0, CrowdRLAgentState(position=(5.0, 5.0), head_angle=0.0), target=(19.0, 5.0)
        )
        world = model.build_world_state(ego, _StubEnvQuery())
        assert np.allclose(self._rays(model, world), 1.0), "no geometry -> all rays clear"

    def test_rays_detect_a_wall_ahead(self):
        model = _model()
        ego = _StubAgent(
            0, CrowdRLAgentState(position=(5.0, 5.0), head_angle=0.0), target=(19.0, 5.0)
        )
        # Vertical wall 1 m ahead of an agent facing +x, ray range 5 m.
        env_query = _StubEnvQuery(walls=[_StubWall((6.0, 0.0), (6.0, 10.0))])
        world = model.build_world_state(ego, env_query)
        rays = self._rays(model, world)

        assert rays.min() < 1.0, "a wall 1 m ahead must register a hit"
        assert rays.min() == pytest.approx(1.0 / 5.0, abs=0.05), (
            "forward ray should read ~1m/5m of its normalised range"
        )

    def test_rays_detect_a_neighbouring_agent(self):
        """Rays intersect agent collision boundaries, not just walls."""
        model = _model()
        ego = _StubAgent(
            0, CrowdRLAgentState(position=(5.0, 5.0), head_angle=0.0), target=(19.0, 5.0)
        )
        blocker = _StubAgent(1, CrowdRLAgentState(position=(6.0, 5.0)), target=(19.0, 5.0))

        alone = self._rays(model, model.build_world_state(ego, _StubEnvQuery()))
        blocked = self._rays(model, model.build_world_state(ego, _StubEnvQuery(agents=[blocker])))

        assert np.allclose(alone, 1.0)
        assert blocked.min() < 1.0, "an agent 1 m ahead must occlude at least one ray"

    def test_next_target_feeds_the_nav_block_not_the_final_goal(self):
        """The nav block must follow the ROUTED waypoint. Pointing at the
        final goal instead sends agents through walls -- the exact failure
        we filed as jupedsim#1625."""
        cfg = ObsConfig(use_navmesh=True)
        model = _model(obs_config=cfg)
        ego = _StubAgent(0, CrowdRLAgentState(position=(5.0, 5.0)), target=(19.0, 5.0))
        ego.next_target = (5.0, 9.0)  # router says: around the corner, +y

        world = model.build_world_state(ego, _StubEnvQuery())
        obs = build_observation(world, 0, cfg)

        # Ego heading 0 (+x): waypoint due +y must read (0, 1) in ego frame,
        # with path_deviation 0.0 (single-waypoint contract).
        np.testing.assert_allclose(obs[-3:], [0.0, 1.0, 0.0], atol=1e-12)

    def test_query_radii_follow_the_raycast_range(self):
        """Regression guard: hardcoding the radii would make walls and agents
        beyond 5 m invisible for a policy trained with longer rays."""
        cfg = ObsConfig(raycast=RaycastConfig(max_range=12.0))
        model = _model(obs_config=cfg)

        assert model.wall_query_radius == pytest.approx(12.0)
        assert model.neighbor_radius >= 12.0, (
            "neighbour query must cover the ray horizon, since rays hit agents"
        )

    def test_explicit_radii_override_the_defaults(self):
        model = _model(neighbor_radius=3.0, wall_query_radius=4.0)
        assert model.neighbor_radius == pytest.approx(3.0)
        assert model.wall_query_radius == pytest.approx(4.0)
