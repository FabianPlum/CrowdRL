"""Byte-identity: LockstepPolicyModel vs the CrowdRL-native reference loop.

The reference loop below calls the SAME ``native_batch_step`` the model runs
internally -- byte-identity is a structural property of sharing the step
function, and these tests pin the remaining glue: pass detection, roster
ordering, temporal bookkeeping, and native removal semantics inside JuPedSim.

Positions are compared with ``np.array_equal`` -- exact, no tolerance.

Needs a JuPedSim 2.0 source build on sys.path (skips in CI) and the shipped
``example_model/policy_r0400.onnx``.
"""

from __future__ import annotations

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
from crowdrl_core.geometry import build_navmesh, extract_wall_segments  # noqa: E402
from crowdrl_core.world_state import WorldState  # noqa: E402
from crowdrl_jupedsim import (  # noqa: E402
    CrowdRLAgentState,
    LockstepPolicyModel,
    OnnxPolicy,
    native_batch_step,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MODEL = ROOT / "example_model" / "policy_r0400.onnx"
pytestmark = pytest.mark.skipif(
    not EXAMPLE_MODEL.is_file(), reason="shipped example model missing"
)

W = 0.8  # the run's trained desired_velocity_weight
DT = 0.01


def run_native_reference(area, exit_polygon, spawns, goal, max_steps):
    """CrowdRL-native loop on the shared step function; native removal."""
    policy = OnnxPolicy(EXAMPLE_MODEL)
    oc, ac = policy.metadata.obs_config, policy.metadata.action_config
    buf = oc.temporal_memory_window + 1
    session = policy._session
    exit_poly = shapely.Polygon(exit_polygon)
    goal = np.asarray(goal, dtype=np.float64)

    n = len(spawns)
    alive = list(range(n))
    positions = np.array(spawns, dtype=np.float64)
    velocities = np.zeros((n, 2))
    torso = np.zeros(n)
    head = np.zeros(n)
    heading = np.zeros(n)
    spawn = positions.copy()
    init_g = np.linalg.norm(goal - positions, axis=1)
    cum = np.zeros(n)
    pos_h = np.repeat(positions[:, None, :], buf, axis=1)
    g_h = np.repeat(init_g[:, None], buf, axis=1)

    wall_segments = extract_wall_segments(area)
    navmesh = build_navmesh(area)

    def policy_batch(obs):
        out = session.run(None, {"observations": obs.astype(np.float32)})[0]
        return np.asarray(out, dtype=np.float64)

    history, exits = [], {}
    for step in range(1, max_steps + 1):
        idx = np.array(alive, dtype=np.intp)
        world = WorldState(
            positions=positions[idx].copy(),
            velocities=velocities[idx].copy(),
            torso_orientations=torso[idx].copy(),
            head_orientations=head[idx].copy(),
            shoulder_widths=np.full(len(idx), 0.225),
            chest_depths=np.full(len(idx), 0.15),
            masses=np.full(len(idx), 80.0),
            goal_positions=np.tile(goal, (len(idx), 1)),
            walkable_polygon=area,
            wall_segments=wall_segments,
            navmesh=navmesh,
            active_mask=np.ones(len(idx), dtype=np.bool_),
        )
        world.preferred_speeds = np.full(len(idx), 1.34)
        world.spawn_positions = spawn[idx].copy()
        world.initial_goal_distances = init_g[idx].copy()
        world.cumulative_path_length = cum[idx].copy()
        world.pos_history = pos_h[idx].copy()
        world.gdist_history = g_h[idx].copy()
        world.step_count = step - 1

        prev = world.positions.copy()
        batch = native_batch_step(
            world,
            policy_batch,
            oc,
            ac,
            desired_velocity_weight=W,
            max_velocity_magnitude=5.0,
            contact_stiffness=30000.0,
            contact_damping=500.0,
            dt=DT,
        )

        wi = (step - 1) % buf
        deltas = np.linalg.norm(world.positions - prev, axis=1)
        gdists = np.linalg.norm(world.goal_positions - world.positions, axis=1)
        still = []
        for k, i in enumerate(alive):
            positions[i] = world.positions[k]
            velocities[i] = world.velocities[k]
            heading[i] = batch.new_headings[k]
            torso[i] = world.torso_orientations[k]
            head[i] = world.head_orientations[k]
            cum[i] += deltas[k]
            pos_h[i, wi] = world.positions[k]
            g_h[i, wi] = gdists[k]
            if exit_poly.contains(shapely.Point(positions[i])):
                exits[i] = step
            else:
                still.append(i)
        alive = still
        history.append({i: positions[i].copy() for i in alive})
        if not alive:
            break
    return history, exits


def run_lockstep_jupedsim(area, exit_polygon, spawns, max_steps):
    policy = OnnxPolicy(EXAMPLE_MODEL)
    model = LockstepPolicyModel(
        policy,
        walkable_geometry=area,
        exit_geometries=[exit_polygon],
        desired_velocity_weight=W,
    )
    sim = jps.Simulation(model=model, geometry=area, dt=DT)
    exit_id = sim.add_exit_stage(exit_polygon)
    journey = sim.add_journey(jps.JourneyDescription([exit_id]))
    ids = [
        sim.add_agent(journey_id=journey, stage_id=exit_id, state=CrowdRLAgentState(position=p))
        for p in spawns
    ]
    id2idx = {jid: k for k, jid in enumerate(ids)}

    history, steps = [], 0
    while sim.agent_count() > 0 and steps < max_steps:
        sim.iterate()
        steps += 1
        history.append(
            {
                id2idx[a.id]: np.asarray(a.position)
                for a in sim.agents()
                if a.id not in model._frozen
            }
        )
    exits = {id2idx[jid]: s for jid, s in model.exit_steps.items()}
    return history, exits


def assert_byte_identical(native_history, jps_history):
    steps = min(len(native_history), len(jps_history))
    for t in range(steps):
        assert set(native_history[t]) == set(jps_history[t]), (
            f"step {t + 1}: roster differs "
            f"({sorted(native_history[t])} vs {sorted(jps_history[t])})"
        )
        for i in native_history[t]:
            assert np.array_equal(native_history[t][i], jps_history[t][i]), (
                f"step {t + 1}, agent {i}: positions differ by "
                f"{np.abs(native_history[t][i] - jps_history[t][i]).max():.3e}"
            )


class TestCorridorWithExits:
    """Short corridor run that completes: exercises removal semantics too."""

    AREA = shapely.Polygon([(0, 0), (20, 0), (20, 6), (0, 6)])
    EXIT = [(19, 2), (20, 2), (20, 4), (19, 4)]
    SPAWNS = [(14.0, 3.0), (13.0, 2.4), (13.5, 3.6)]
    GOAL = shapely.Polygon(EXIT).centroid.coords[0]

    def test_full_run_is_byte_identical_including_exits(self):
        native_history, native_exits = run_native_reference(
            self.AREA, self.EXIT, self.SPAWNS, self.GOAL, max_steps=1500
        )
        jps_history, jps_exits = run_lockstep_jupedsim(
            self.AREA, self.EXIT, self.SPAWNS, max_steps=1500
        )
        assert native_exits and native_exits == jps_exits
        assert_byte_identical(native_history, jps_history)


class TestCornerSegment:
    """First 300 steps of the #1625 corner: funnel corners + wall projection."""

    AREA = shapely.Polygon([(0, 0), (12, 0), (12, 12), (10, 12), (10, 2), (0, 2)])
    EXIT = [(10, 11), (12, 11), (12, 12), (10, 12)]
    SPAWNS = [(1.5, 1.0), (3.0, 1.2), (4.5, 0.8), (6.0, 1.0)]
    GOAL = (11.0, 11.5)

    def test_trajectories_are_byte_identical(self):
        native_history, _ = run_native_reference(
            self.AREA, self.EXIT, self.SPAWNS, self.GOAL, max_steps=300
        )
        jps_history, _ = run_lockstep_jupedsim(self.AREA, self.EXIT, self.SPAWNS, max_steps=300)
        assert_byte_identical(native_history, jps_history)
