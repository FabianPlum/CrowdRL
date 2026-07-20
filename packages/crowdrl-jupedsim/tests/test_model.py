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
from crowdrl_core.observation import ObsConfig, build_observation  # noqa: E402
from crowdrl_core.sensing import RaycastConfig  # noqa: E402
from crowdrl_jupedsim import (  # noqa: E402
    ConstantPolicy,
    CrowdRLAgentState,
    LearnedPolicyModel,
)

# Drive straight ahead at max forward speed, no turning.
FORWARD = [1.0, 0.0, 0.0, 0.0]


def _room() -> shapely.Polygon:
    return shapely.Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])


def _sim(model, dt: float = 0.05):
    sim = jps.Simulation(model=model, geometry=_room(), dt=dt)
    exit_id = sim.add_exit_stage([(19, 9), (19, 11), (20, 11), (20, 9)])
    journey_id = sim.add_journey(jps.JourneyDescription([exit_id]))
    return sim, exit_id, journey_id


class _StubAgent:
    """Minimal stand-in for a JuPedSim transient agent view."""

    def __init__(self, agent_id, state, target):
        self.id = agent_id
        self.model = state
        self.target = target

    @property
    def position(self):
        return self.model.position


class _StubNeighborhood:
    def __init__(self, agents):
        self._agents = agents

    def get_neighboring_agents(self, position, radius):
        return self._agents


class _StubGeometry:
    def get_walls_in_distance_to(self, point, distance):
        return []


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
        model = LearnedPolicyModel(ConstantPolicy(FORWARD))

        ego_state = CrowdRLAgentState(position=(5.0, 5.0), velocity=(1.0, 0.0))
        ego = _StubAgent(0, ego_state, target=(19.0, 10.0))
        neighbor = _StubAgent(1, CrowdRLAgentState(position=(6.0, 5.0)), target=(19.0, 10.0))
        # The neighbourhood query returns the querying agent too; it must be filtered.
        neighborhood = _StubNeighborhood([ego, neighbor])

        world = model.build_world_state(ego, _StubGeometry(), neighborhood)

        assert world.n_agents == 2, "ego + 1 neighbour, self-match removed"
        np.testing.assert_allclose(world.positions[0], [5.0, 5.0])
        np.testing.assert_allclose(world.positions[1], [6.0, 5.0])
        np.testing.assert_allclose(world.goal_positions[0], [19.0, 10.0])
        world.validate()

    def test_wall_segments_are_shaped_for_crowdrl_core(self):
        model = LearnedPolicyModel(ConstantPolicy(FORWARD))
        ego = _StubAgent(0, CrowdRLAgentState(position=(5.0, 5.0)), target=(19.0, 10.0))
        world = model.build_world_state(ego, _StubGeometry(), _StubNeighborhood([]))
        assert world.wall_segments.shape == (0, 2, 2)


class TestSimulationIntegration:
    def test_agent_advances_toward_the_routing_target(self):
        model = LearnedPolicyModel(ConstantPolicy(FORWARD))
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
        model = LearnedPolicyModel(ConstantPolicy(FORWARD))
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
        model = LearnedPolicyModel(ConstantPolicy(FORWARD))
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
        model = LearnedPolicyModel(ConstantPolicy(FORWARD), desired_velocity_weight=0.05)
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
        model = LearnedPolicyModel(ConstantPolicy(FORWARD))
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


class _StubWall:
    """Stand-in for a JuPedSim LineSegment (exposes p1 / p2 endpoints)."""

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2


class _WalledGeometry:
    def __init__(self, walls):
        self._walls = walls

    def get_walls_in_distance_to(self, point, distance):
        return self._walls


class TestRaycasting:
    """Every trained policy uses raycasts, so a silently-blind sensor would
    produce plausible-looking but wrong actions. These assert rays actually
    respond to geometry and to other agents."""

    @staticmethod
    def _rays(model, world):
        obs = build_observation(world, 0, model.obs_config)
        return obs[-model.obs_config.raycast.n_rays :]

    def test_open_space_reads_clear(self):
        model = LearnedPolicyModel(ConstantPolicy(FORWARD))
        ego = _StubAgent(
            0, CrowdRLAgentState(position=(5.0, 5.0), head_angle=0.0), target=(19.0, 5.0)
        )
        world = model.build_world_state(ego, _StubGeometry(), _StubNeighborhood([]))
        assert np.allclose(self._rays(model, world), 1.0), "no geometry -> all rays clear"

    def test_rays_detect_a_wall_ahead(self):
        model = LearnedPolicyModel(ConstantPolicy(FORWARD))
        ego = _StubAgent(
            0, CrowdRLAgentState(position=(5.0, 5.0), head_angle=0.0), target=(19.0, 5.0)
        )
        # Vertical wall 1 m ahead of an agent facing +x, ray range 5 m.
        geometry = _WalledGeometry([_StubWall((6.0, 0.0), (6.0, 10.0))])
        world = model.build_world_state(ego, geometry, _StubNeighborhood([]))
        rays = self._rays(model, world)

        assert rays.min() < 1.0, "a wall 1 m ahead must register a hit"
        assert rays.min() == pytest.approx(1.0 / 5.0, abs=0.05), (
            "forward ray should read ~1m/5m of its normalised range"
        )

    def test_rays_detect_a_neighbouring_agent(self):
        """Rays intersect agent collision boundaries, not just walls."""
        model = LearnedPolicyModel(ConstantPolicy(FORWARD))
        ego = _StubAgent(
            0, CrowdRLAgentState(position=(5.0, 5.0), head_angle=0.0), target=(19.0, 5.0)
        )
        blocker = _StubAgent(1, CrowdRLAgentState(position=(6.0, 5.0)), target=(19.0, 5.0))

        alone = self._rays(
            model, model.build_world_state(ego, _StubGeometry(), _StubNeighborhood([]))
        )
        blocked = self._rays(
            model, model.build_world_state(ego, _StubGeometry(), _StubNeighborhood([blocker]))
        )

        assert np.allclose(alone, 1.0)
        assert blocked.min() < 1.0, "an agent 1 m ahead must occlude at least one ray"

    def test_query_radii_follow_the_raycast_range(self):
        """Regression guard: hardcoding the radii would make walls and agents
        beyond 5 m invisible for a policy trained with longer rays."""
        cfg = ObsConfig(raycast=RaycastConfig(max_range=12.0))
        model = LearnedPolicyModel(ConstantPolicy(FORWARD), obs_config=cfg)

        assert model.wall_query_radius == pytest.approx(12.0)
        assert model.neighbor_radius >= 12.0, (
            "neighbour query must cover the ray horizon, since rays hit agents"
        )

    def test_explicit_radii_override_the_defaults(self):
        model = LearnedPolicyModel(
            ConstantPolicy(FORWARD), neighbor_radius=3.0, wall_query_radius=4.0
        )
        assert model.neighbor_radius == pytest.approx(3.0)
        assert model.wall_query_radius == pytest.approx(4.0)
