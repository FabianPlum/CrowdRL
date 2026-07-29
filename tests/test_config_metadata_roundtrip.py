"""End-to-end contract test for issue #7.

The full journey of the embedded training configuration: a real torch export
through ``crowdrl_train.export.export_onnx`` -> the artefact on disk -> the
deployment reader (``OnnxPolicy`` + ``resolve_configs``), which must
reconstruct the exact configs with no hand-supplied information -- and refuse
loudly when a hand-supplied config disagrees.

Deliberately mirrors the real deployment shape: the 89D
``use_goal_direction=False`` + navmesh + temporal-memory configuration of the
current best runs. Imports crowdrl_jupedsim's policy surface only, so it runs
without a jupedsim build (CI).
"""

from __future__ import annotations

import numpy as np
import pytest

from crowdrl_core.action import ActionConfig
from crowdrl_core.observation import ObsConfig

from crowdrl_jupedsim.policy import OnnxPolicy, resolve_configs, resolve_dynamics
from crowdrl_train.config import NetworkConfig
from crowdrl_train.export import export_onnx
from crowdrl_train.networks import Actor

OBS = ObsConfig(use_navmesh=True, use_goal_direction=False, use_temporal_memory=True)  # 89D
ACTION = ActionConfig(
    max_heading_change=float(np.radians(4.8)),
    max_torso_change=float(np.radians(4.8)),
    speed_turn_coupling=True,
    turn_pivot_rate=float(np.radians(240.0)),
)


@pytest.fixture(scope="module")
def exported_policy(tmp_path_factory):
    actor = Actor(NetworkConfig(obs_dim=OBS.obs_dim, action_dim=4, actor_hidden_sizes=(32, 32)))
    path = tmp_path_factory.mktemp("export") / "policy.onnx"
    export_onnx(
        actor,
        normalizer=None,
        output_path=path,
        obs_config=OBS,
        action_config=ACTION,
        dynamics={"desired_velocity_weight": 0.8, "max_velocity_magnitude": 3.0},
        provenance={"run": "roundtrip-test", "git_rev": "cafe123"},
    )
    return path


class TestRoundTrip:
    def test_deployment_self_configures_exactly(self, exported_policy):
        """The whole point of issue #7: .onnx in, exact training configs out,
        zero hand-supplied information."""
        policy = OnnxPolicy(exported_policy)
        resolved_obs, resolved_action = resolve_configs(policy)
        assert resolved_obs == OBS
        assert resolved_action == ACTION
        assert policy.metadata.provenance["git_rev"] == "cafe123"
        assert policy.obs_dim == OBS.obs_dim == 89

    def test_deliberate_mismatch_raises(self, exported_policy):
        """The nogoaldir landmine: same width, different semantics -- the
        exact silent failure class this contract exists to kill."""
        policy = OnnxPolicy(exported_policy)
        flipped = ObsConfig(use_navmesh=True, use_goal_direction=True, use_temporal_memory=True)
        assert flipped.obs_dim == OBS.obs_dim  # same width, silent before #7
        with pytest.raises(ValueError, match="use_goal_direction"):
            resolve_configs(policy, flipped, ACTION)

    def test_legacy_export_requires_explicit_and_warns(self, tmp_path):
        actor = Actor(
            NetworkConfig(obs_dim=OBS.obs_dim, action_dim=4, actor_hidden_sizes=(32, 32))
        )
        path = tmp_path / "legacy.onnx"
        export_onnx(actor, normalizer=None, output_path=path)

        policy = OnnxPolicy(path)
        assert policy.metadata is None
        with pytest.raises(ValueError, match="explicit"):
            resolve_configs(policy)
        with pytest.warns(UserWarning, match="cannot be verified"):
            resolved_obs, _ = resolve_configs(policy, OBS, ACTION)
        assert resolved_obs == OBS


class TestDynamicsRoundTrip:
    def test_deployment_self_configures_dynamics(self, exported_policy):
        """Schema v2: the trained physics travel with the artefact -- the
        desired_velocity_weight filter and the speed clamp the run actually
        used, instead of adapter defaults."""
        policy = OnnxPolicy(exported_policy)
        resolved = resolve_dynamics(policy, {})
        assert resolved["desired_velocity_weight"] == 0.8
        assert resolved["max_velocity_magnitude"] == 3.0
        assert resolved["contact_stiffness"] == 30000.0  # unrecorded -> default

    def test_disagreeing_explicit_dynamics_raise(self, exported_policy):
        policy = OnnxPolicy(exported_policy)
        with pytest.raises(ValueError, match="desired_velocity_weight"):
            resolve_dynamics(policy, {"desired_velocity_weight": 0.05})
