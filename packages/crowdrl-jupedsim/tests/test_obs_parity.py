"""Observation parity between training and deployment -- the transfer guarantee.

The architectural invariant from Project Plan v8, Section 3.6: if WorldState is
populated correctly from JuPedSim's agent states, the observation vector must be
numerically identical to what the policy saw during training for the same
physical configuration. Drift between the two population paths is a bug that
produces subtle policy failures rather than a crash, so it needs a direct test.

Both paths below consume the *same* crowdrl-core ``build_observation``. What is
under test is the WorldState each side hands it.

Note ``build_observation`` runs its own K-nearest query internally, so the order
of agents within the arrays is irrelevant -- only that the same set of agents,
with the same properties, is present.
"""

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
from crowdrl_core.geometry import extract_wall_segments  # noqa: E402
from crowdrl_core.observation import ObsConfig, build_observation  # noqa: E402
from crowdrl_core.world_state import WorldState  # noqa: E402
from crowdrl_jupedsim import ConstantPolicy, CrowdRLAgentState, LearnedPolicyModel  # noqa: E402

HOLD = [0.0, 0.0, 0.0, 0.0]

ROOM = shapely.Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])

# A small heterogeneous crowd: varied orientations and body dimensions so the
# ego, social and raycast blocks all carry non-trivial values.
AGENTS = [
    dict(
        position=(5.0, 10.0),
        velocity=(0.8, 0.1),
        heading=0.10,
        torso_angle=0.15,
        head_angle=0.35,
        shoulder_width=0.24,
        chest_depth=0.16,
        mass=82.0,
        preferred_speed=1.40,
    ),
    dict(
        position=(6.2, 10.4),
        velocity=(-0.3, 0.5),
        heading=2.10,
        torso_angle=2.00,
        head_angle=1.70,
        shoulder_width=0.21,
        chest_depth=0.14,
        mass=71.0,
        preferred_speed=1.25,
    ),
    dict(
        position=(5.4, 8.9),
        velocity=(0.2, -0.6),
        heading=-1.20,
        torso_angle=-1.10,
        head_angle=-0.70,
        shoulder_width=0.27,
        chest_depth=0.18,
        mass=95.0,
        preferred_speed=1.55,
    ),
]


class _RecordingModel(LearnedPolicyModel):
    """Captures the observation and routing target the adapter produced per agent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured: dict[int, tuple[np.ndarray, tuple[float, float]]] = {}

    def compute_next_state(self, dt, ped, env_query):
        world = self.build_world_state(ped, env_query)
        obs = build_observation(world, 0, self.obs_config)
        self.captured[ped.id] = (
            np.asarray(obs, dtype=np.float64).copy(),
            tuple(ped.final_target),
        )
        return super().compute_next_state(dt, ped, env_query)


def _training_world(targets: list[tuple[float, float]]) -> WorldState:
    """WorldState as crowdrl-env would populate it, for the same configuration."""
    return WorldState(
        positions=np.array([a["position"] for a in AGENTS], dtype=np.float64),
        velocities=np.array([a["velocity"] for a in AGENTS], dtype=np.float64),
        torso_orientations=np.array([a["torso_angle"] for a in AGENTS], dtype=np.float64),
        head_orientations=np.array([a["head_angle"] for a in AGENTS], dtype=np.float64),
        shoulder_widths=np.array([a["shoulder_width"] for a in AGENTS], dtype=np.float64),
        chest_depths=np.array([a["chest_depth"] for a in AGENTS], dtype=np.float64),
        masses=np.array([a["mass"] for a in AGENTS], dtype=np.float64),
        goal_positions=np.array(targets, dtype=np.float64),
        preferred_speeds=np.array([a["preferred_speed"] for a in AGENTS], dtype=np.float64),
        walkable_polygon=ROOM,
        wall_segments=extract_wall_segments(ROOM),
    )


@pytest.fixture
def parity(obs_config: ObsConfig | None = None):
    """Run one JuPedSim step, capturing each agent's adapter-side observation."""
    config = obs_config or ObsConfig()
    # ConstantPolicy carries no embedded metadata, so both configs are explicit.
    model = _RecordingModel(ConstantPolicy(HOLD), obs_config=config, action_config=ActionConfig())

    # dt=0.01 matches CrowdEnvConfig.dt; see the note in test_model.py::_sim.
    sim = jps.Simulation(model=model, geometry=ROOM, dt=0.01)
    exit_id = sim.add_exit_stage([(19, 9), (19, 11), (20, 11), (20, 9)])
    journey_id = sim.add_journey(jps.JourneyDescription([exit_id]))

    ids = [
        sim.add_agent(
            journey_id=journey_id,
            stage_id=exit_id,
            state=CrowdRLAgentState(**spec),
        )
        for spec in AGENTS
    ]

    # One iteration: routing sets each agent's final_target/next_target, then
    # the operational step runs and the model records the observation it built
    # from the pre-step state.
    sim.iterate()

    assert len(model.captured) == len(AGENTS), "every agent should have been visited"
    return config, ids, model.captured


class TestObservationParity:
    def test_observations_match_between_training_and_deployment(self, parity):
        config, ids, captured = parity
        targets = [captured[i][1] for i in ids]
        world = _training_world(targets)

        for idx, agent_id in enumerate(ids):
            deployed, _ = captured[agent_id]
            trained = build_observation(world, idx, config)

            assert deployed.shape == trained.shape, (
                f"agent {idx}: obs width {deployed.shape} vs {trained.shape} -- "
                "the policy would be fed a differently-shaped world"
            )
            np.testing.assert_allclose(
                deployed,
                trained,
                rtol=1e-9,
                atol=1e-9,
                err_msg=(
                    f"agent {idx}: deployment observation diverged from training. "
                    "This breaks the transfer guarantee."
                ),
            )

    def test_ego_block_matches(self, parity):
        """Isolates the ego block (first 8 dims) for a sharper failure message."""
        config, ids, captured = parity
        targets = [captured[i][1] for i in ids]
        world = _training_world(targets)

        for idx, agent_id in enumerate(ids):
            deployed, _ = captured[agent_id]
            trained = build_observation(world, idx, config)
            np.testing.assert_allclose(
                deployed[:8], trained[:8], rtol=1e-9, atol=1e-9, err_msg=f"ego block, agent {idx}"
            )

    def test_raycast_block_matches(self, parity):
        """Rays are built from JuPedSim wall segments on one side and from
        crowdrl-core's polygon extraction on the other -- different segment
        decompositions that must still yield identical distances."""
        config, ids, captured = parity
        targets = [captured[i][1] for i in ids]
        world = _training_world(targets)
        n_rays = config.raycast.n_rays

        for idx, agent_id in enumerate(ids):
            deployed, _ = captured[agent_id]
            trained = build_observation(world, idx, config)
            np.testing.assert_allclose(
                deployed[-n_rays:],
                trained[-n_rays:],
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"raycast block, agent {idx}",
            )
