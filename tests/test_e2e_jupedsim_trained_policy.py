"""End-to-end: a trained CrowdRL policy drives JuPedSim agents around a corner.

The scenario is the jupedsim#1625 reproduction geometry: an L-shaped corridor
where an agent must walk +x along the lower corridor, then turn +y to the
exit. Under the old adapter (final goal as the only navigation signal) every
agent walked into the corner wall and pinned there -- 0/N exited. With
jupedsim 2.0 exposing ``ped.next_target`` and the adapter feeding it into the
navmesh observation block, a nogoaldir policy has its full navigation signal
and must make the turn.

Requires artefacts that do not ship with the repo, so it runs only when
``CROWDRL_E2E_RESULTS_DIR`` points at a results dir carrying
``config_resolved.yaml`` + ``policy_r0800.onnx`` (or the rollout named by
``CROWDRL_E2E_POLICY``), plus a JuPedSim 2.0 source build on sys.path.
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

RESULTS_DIR = os.environ.get("CROWDRL_E2E_RESULTS_DIR")
if not RESULTS_DIR:
    pytest.skip(
        "set CROWDRL_E2E_RESULTS_DIR to a results dir with config_resolved.yaml "
        "+ policy ONNX to run the end-to-end scenario",
        allow_module_level=True,
    )

import shapely  # noqa: E402

import jupedsim as jps  # noqa: E402
from crowdrl_jupedsim import CrowdRLAgentState, LearnedPolicyModel, OnnxPolicy  # noqa: E402

# Make the repo root importable so `import train_mappo` works.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_mappo import build_env_config, load_config  # noqa: E402

# The jupedsim#1625 reproduction geometry: lower corridor -> corner -> exit.
AREA = shapely.Polygon([(0, 0), (12, 0), (12, 12), (10, 12), (10, 2), (0, 2)])
EXIT = [(10, 11), (12, 11), (12, 12), (10, 12)]
SPAWNS = [(1.5, 1.0), (3.0, 1.2), (4.5, 0.8), (6.0, 1.0)]

MAX_STEPS = 4000  # 40 s sim time at dt=0.01; the route is ~20 m


@pytest.fixture(scope="module")
def sim_result():
    results = Path(RESULTS_DIR)
    policy_name = os.environ.get("CROWDRL_E2E_POLICY", "policy_r0800.onnx")
    env_config = build_env_config(load_config(results / "config_resolved.yaml"))

    policy = OnnxPolicy(results / policy_name)
    with warnings.catch_warnings():
        # Pre-#7 artefact: explicit configs rebuilt from config_resolved.yaml,
        # the documented legacy path (emits the "cannot be verified" warning).
        warnings.simplefilter("ignore", UserWarning)
        model = LearnedPolicyModel(
            policy,
            obs_config=env_config.obs,
            action_config=env_config.action,
            desired_velocity_weight=env_config.desired_velocity_weight,
        )

    sim = jps.Simulation(model=model, geometry=AREA, dt=0.01)
    exit_id = sim.add_exit_stage(EXIT)
    journey_id = sim.add_journey(jps.JourneyDescription([exit_id]))
    ids = [
        sim.add_agent(journey_id=journey_id, stage_id=exit_id, state=CrowdRLAgentState(position=p))
        for p in SPAWNS
    ]

    trajectories: dict[int, list[tuple[float, float]]] = {i: [] for i in ids}
    steps = 0
    while sim.agent_count() > 0 and steps < MAX_STEPS:
        sim.iterate()
        steps += 1
        for agent in sim.agents():
            trajectories[agent.id].append(tuple(agent.position))

    return sim, steps, ids, trajectories


class TestCornerScenario:
    def test_every_agent_reaches_the_exit(self, sim_result):
        sim, steps, ids, trajectories = sim_result
        stuck = {i: t[-1] for i, t in trajectories.items() if t and sim.agent_count() > 0}
        assert sim.agent_count() == 0, (
            f"{sim.agent_count()}/{len(ids)} agents still in the simulation "
            f"after {steps} steps; last positions: {stuck}"
        )

    def test_agents_actually_rounded_the_corner(self, sim_result):
        """Guard against pathological exits: the route must pass through the
        corner region (x>9, y rising through the vertical corridor), not
        terminate at the lower wall -- the pre-#1626 failure mode was every
        agent pinned at y=2.0."""
        _, _, _, trajectories = sim_result
        for agent_id, traj in trajectories.items():
            xs = np.array([p[0] for p in traj])
            ys = np.array([p[1] for p in traj])
            assert ys.max() > 9.0, (
                f"agent {agent_id} never entered the vertical corridor "
                f"(max y {ys.max():.2f}) -- corner navigation failed"
            )
            assert xs.max() > 10.0, f"agent {agent_id} never reached the exit column"

    def test_no_agent_left_the_walkable_area(self, sim_result):
        _, _, _, trajectories = sim_result
        for agent_id, traj in trajectories.items():
            for x, y in traj[:: max(1, len(traj) // 200)]:
                assert AREA.buffer(1e-6).contains(shapely.Point(x, y)), (
                    f"agent {agent_id} left the walkable area at ({x:.2f}, {y:.2f})"
                )
