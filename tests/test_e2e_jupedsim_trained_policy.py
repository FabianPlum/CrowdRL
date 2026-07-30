"""End-to-end: trained CrowdRL policies drive JuPedSim agents.

Scenarios:

* **Corner** -- the jupedsim#1625 reproduction geometry: an L-shaped corridor
  where an agent must walk +x along the lower corridor, then turn +y to the
  exit. Under the old adapter (final goal as the only navigation signal)
  every agent walked into the corner wall and pinned there -- 0/N exited.
  With jupedsim 2.0 exposing ``ped.next_target`` and the adapter feeding it
  into the navmesh observation block, a nogoaldir policy has its full
  navigation signal and must make the turn.
* **Bottleneck** -- 12 agents through a 1.4 m aperture (mid-range of the
  tier-1 training distribution): does crowd behaviour translate, not just
  solo navigation?

The self-configured scenarios run against the shipped, metadata-carrying
``example_model/policy_r0400.onnx`` and need only a JuPedSim 2.0 source
build on ``sys.path`` -- no configuration of any kind: the adapter
reconstructs the training configs from the artefact (issue #7).

The legacy-path scenario additionally needs ``CROWDRL_E2E_RESULTS_DIR``
pointing at a results dir with ``config_resolved.yaml`` + the policy named
by ``CROWDRL_E2E_POLICY`` (default ``policy_r0800.onnx``); it exercises the
pre-#7 deployment route (explicit configs rebuilt from the YAML).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jupedsim", reason="requires a JuPedSim 2.0 source build")
pytest.importorskip(
    "jupedsim.models.custom_model",
    reason="requires the JuPedSim 2.0 CustomOperationalModel layer",
)

import shapely  # noqa: E402

import jupedsim as jps  # noqa: E402
from crowdrl_jupedsim import CrowdRLAgentState, LearnedPolicyModel, OnnxPolicy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MODEL = ROOT / "example_model" / "policy_r0400.onnx"
RESULTS_DIR = os.environ.get("CROWDRL_E2E_RESULTS_DIR")

# The jupedsim#1625 reproduction geometry: lower corridor -> corner -> exit.
CORNER_AREA = shapely.Polygon([(0, 0), (12, 0), (12, 12), (10, 12), (10, 2), (0, 2)])
CORNER_EXIT = [(10, 11), (12, 11), (12, 12), (10, 12)]
CORNER_SPAWNS = [(1.5, 1.0), (3.0, 1.2), (4.5, 0.8), (6.0, 1.0)]

# Hourglass room: 14x10, pinched to a 1.4 m aperture (y in [4.3, 5.7]) at x~7.
BOTTLENECK_AREA = shapely.Polygon(
    [
        (0, 0),
        (6.8, 0),
        (6.8, 4.3),
        (7.2, 4.3),
        (7.2, 0),
        (14, 0),
        (14, 10),
        (7.2, 10),
        (7.2, 5.7),
        (6.8, 5.7),
        (6.8, 10),
        (0, 10),
    ]
)
BOTTLENECK_EXIT = [(13.0, 4.0), (14.0, 4.0), (14.0, 6.0), (13.0, 6.0)]
# Deterministic spread over the left chamber (rng seed pinned).
_rng = np.random.default_rng(7)
BOTTLENECK_SPAWNS = [
    (float(x), float(y)) for x, y in zip(_rng.uniform(1.0, 5.5, 12), _rng.uniform(1.5, 8.5, 12))
]


def run_scenario(model, area, exit_poly, spawns, max_steps):
    """Drive one simulation to completion.

    Returns (sim, steps, ids, trajectories, min_pairwise) where min_pairwise
    is the smallest centre-to-centre agent distance observed at any step.
    """
    sim = jps.Simulation(model=model, geometry=area, dt=0.01)
    exit_id = sim.add_exit_stage(exit_poly)
    journey_id = sim.add_journey(jps.JourneyDescription([exit_id]))
    ids = [
        sim.add_agent(journey_id=journey_id, stage_id=exit_id, state=CrowdRLAgentState(position=p))
        for p in spawns
    ]

    trajectories: dict[int, list[tuple[float, float]]] = {i: [] for i in ids}
    min_pairwise = np.inf
    steps = 0
    while sim.agent_count() > 0 and steps < max_steps:
        sim.iterate()
        steps += 1
        positions = []
        for agent in sim.agents():
            trajectories[agent.id].append(tuple(agent.position))
            positions.append(agent.position)
        if len(positions) >= 2:
            pos = np.asarray(positions)
            dists = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
            np.fill_diagonal(dists, np.inf)
            min_pairwise = min(min_pairwise, float(dists.min()))

    return sim, steps, ids, trajectories, min_pairwise


def assert_all_exited(sim, steps, ids, trajectories, *_):
    # Report only the agents actually still in the simulation, not every agent.
    stuck = {
        a.id: tuple(round(c, 2) for c in trajectories[a.id][-1])
        for a in sim.agents()
        if trajectories.get(a.id)
    }
    assert sim.agent_count() == 0, (
        f"{sim.agent_count()}/{len(ids)} agents still in the simulation "
        f"after {steps} steps; last positions: {stuck}"
    )


def assert_inside(area, trajectories):
    walkable = area.buffer(1e-6)
    for agent_id, traj in trajectories.items():
        for x, y in traj[:: max(1, len(traj) // 200)]:
            assert walkable.contains(shapely.Point(x, y)), (
                f"agent {agent_id} left the walkable area at ({x:.2f}, {y:.2f})"
            )


@pytest.mark.skipif(not EXAMPLE_MODEL.is_file(), reason="shipped example model missing")
class TestSelfConfiguredCorner:
    """The shipped schema-v2 artefact: configs AND dynamics self-configure,
    so this runs at the TRAINED dynamics (w=0.8, clamp 3.0).

    Baseline: 4/4 exit in ~9.8 s sim (exit steps 738/821/903/979). Until
    2026-07-30 this scenario lost one agent, which was attributed to a policy
    absorbing state; it was in fact divergence channel 8 -- the adapter
    free-integrated a heading instead of re-anchoring it to the torso the way
    both training engines do (see plan/lockstep_parity_analysis.md). With that
    closed, no agent is lost.

    Machine scope: exit steps and the exact completion count rest on ONNX
    Runtime being run-to-run deterministic for this session/hardware. Expect
    ULP-level drift cross-machine, which can shift an exit step by one and, in
    principle, flip a marginal agent -- read a near-miss failure here as
    numerical scope, not a broken adapter.
    """

    @pytest.fixture(scope="class")
    def result(self):
        model = LearnedPolicyModel(OnnxPolicy(EXAMPLE_MODEL), walkable_geometry=CORNER_AREA)
        assert model.desired_velocity_weight == 0.8  # self-configured from metadata
        assert model.max_velocity_magnitude == 3.0
        return run_scenario(model, CORNER_AREA, CORNER_EXIT, CORNER_SPAWNS, max_steps=4000)

    def test_agents_reach_the_exit_at_trained_dynamics(self, result):
        """All 4 exit at the trained dynamics."""
        assert_all_exited(*result)

    def test_exiting_agents_actually_rounded_the_corner(self, result):
        """The route must pass through the vertical corridor -- the pre-#1626
        failure mode was every agent pinned at the lower wall (y=2.0).

        Every trajectory is examined (no straggler exemption): the scenario
        completes, so an unexamined trajectory would mean a regression.
        """
        _, _, ids, trajectories, _ = result
        assert len(trajectories) == len(ids)
        for agent_id, traj in trajectories.items():
            ys = np.array([p[1] for p in traj])
            assert ys.max() > 9.0, (
                f"agent {agent_id} never entered the vertical corridor "
                f"(max y {ys.max():.2f}) -- corner navigation failed"
            )

    def test_no_agent_left_the_walkable_area(self, result):
        assert_inside(CORNER_AREA, result[3])

    def test_bodies_keep_clear_of_walls(self, result):
        """Agent centres must stay one body radius (0.225 m) off the walls --
        the centre-at-the-wall clipping is exactly the regression this pins."""
        trajectories = result[3]
        boundary = CORNER_AREA.boundary
        for agent_id, traj in trajectories.items():
            for x, y in traj[:: max(1, len(traj) // 200)]:
                dist = boundary.distance(shapely.Point(x, y))
                assert dist >= 0.225 - 1e-6, (
                    f"agent {agent_id} centre came within {dist:.3f} m of a wall"
                )


@pytest.mark.skipif(not EXAMPLE_MODEL.is_file(), reason="shipped example model missing")
class TestSelfConfiguredBottleneck:
    """Crowd behaviour, not just solo navigation: 12 agents, 1.4 m aperture,
    at the trained dynamics (self-configured from the schema-v2 artefact).

    Baseline: 12/12 exit in ~7.4 s sim (first 4.21 s, median 6.06 s, last
    7.38 s). Until 2026-07-30 one agent was lost here too -- same cause as in
    the corner scenario, divergence channel 8 (heading anchoring), not a
    policy absorbing state.

    Spacing is a real criterion with contact physics active: pre-physics this
    scenario bottomed out at ~0.04 m centre distance (ghosting); observed
    floor at trained dynamics is 0.41 m.

    Machine scope: as in the corner scenario, exit steps are ONNX-Runtime
    determinism-scoped to this session/hardware.
    """

    @pytest.fixture(scope="class")
    def result(self):
        model = LearnedPolicyModel(OnnxPolicy(EXAMPLE_MODEL), walkable_geometry=BOTTLENECK_AREA)
        return run_scenario(
            model, BOTTLENECK_AREA, BOTTLENECK_EXIT, BOTTLENECK_SPAWNS, max_steps=6000
        )

    def test_crowd_flows_through_at_trained_dynamics(self, result):
        """All 12 clear the neck and exit."""
        assert_all_exited(*result)

    def test_exiting_agents_passed_the_aperture(self, result):
        _, _, ids, trajectories, _ = result
        assert len(trajectories) == len(ids)
        for agent_id, traj in trajectories.items():
            xs = np.array([p[0] for p in traj])
            assert xs.max() > 7.2, f"agent {agent_id} never cleared the neck"

    def test_no_agent_left_the_walkable_area(self, result):
        assert_inside(BOTTLENECK_AREA, result[3])

    def test_agents_keep_body_scale_separation(self, result):
        """The spring-damper contact must keep centres at body scale even in
        the squeeze -- the pre-physics floor was ~0.04 m."""
        min_pairwise = result[4]
        assert min_pairwise > 0.2, (
            f"min centre-to-centre distance {min_pairwise:.3f} m -- contact "
            "forces failed to hold body-scale separation"
        )


@pytest.mark.skipif(
    not RESULTS_DIR,
    reason="set CROWDRL_E2E_RESULTS_DIR to a results dir to run the legacy-path scenario",
)
class TestLegacyPathCorner:
    """Pre-#7 artefact: explicit configs rebuilt from config_resolved.yaml."""

    @pytest.fixture(scope="class")
    def result(self):
        # Make the repo root importable so `import train_mappo` works.
        sys.path.insert(0, str(ROOT))
        from train_mappo import build_env_config, load_config

        results = Path(RESULTS_DIR)
        policy_name = os.environ.get("CROWDRL_E2E_POLICY", "policy_r0800.onnx")
        env_config = build_env_config(load_config(results / "config_resolved.yaml"))

        policy = OnnxPolicy(results / policy_name)
        with warnings.catch_warnings():
            # The documented legacy path emits the "cannot be verified" warning.
            warnings.simplefilter("ignore", UserWarning)
            model = LearnedPolicyModel(
                policy,
                obs_config=env_config.obs,
                action_config=env_config.action,
                walkable_geometry=CORNER_AREA,
                desired_velocity_weight=env_config.desired_velocity_weight,
            )
        return run_scenario(model, CORNER_AREA, CORNER_EXIT, CORNER_SPAWNS, max_steps=4000)

    def test_every_agent_reaches_the_exit(self, result):
        assert_all_exited(*result)

    def test_no_agent_left_the_walkable_area(self, result):
        assert_inside(CORNER_AREA, result[3])
