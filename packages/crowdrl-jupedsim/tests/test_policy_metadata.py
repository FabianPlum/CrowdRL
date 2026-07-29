"""Read side of the embedded-config contract (issue #7).

Covers OnnxPolicy's metadata parsing against hand-built ONNX files and the
resolve_configs decision table. Deliberately imports neither jupedsim nor
torch: this is the deployment surface that must stay testable in CI.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from crowdrl_core.action import ActionConfig
from crowdrl_core.config_io import (
    META_ACTION_CONFIG_KEY,
    META_ACTION_DIM_KEY,
    META_OBS_CONFIG_KEY,
    META_OBS_DIM_KEY,
    META_PROVENANCE_KEY,
    META_SCHEMA_KEY,
    action_config_to_dict,
    obs_config_to_dict,
)
from crowdrl_core.observation import ObsConfig

from crowdrl_jupedsim.policy import ConstantPolicy, OnnxPolicy, resolve_configs

OBS = ObsConfig(use_navmesh=True, use_goal_direction=False, use_temporal_memory=True)  # 89D
ACTION = ActionConfig(max_heading_change=float(np.radians(4.8)), speed_turn_coupling=True)


def _write_policy(path, obs_dim=89, props=None):
    """Minimal valid ONNX file: Identity over [n_agents, obs_dim] float32."""
    inp = helper.make_tensor_value_info("observations", TensorProto.FLOAT, ["n_agents", obs_dim])
    out = helper.make_tensor_value_info("actions", TensorProto.FLOAT, ["n_agents", obs_dim])
    node = helper.make_node("Identity", ["observations"], ["actions"])
    graph = helper.make_graph([node], "policy", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    if props is not None:
        helper.set_model_props(model, props)
    onnx.save(model, str(path))
    return path


def _valid_props(obs=OBS, action=ACTION, **overrides):
    props = {
        META_SCHEMA_KEY: "1",
        META_OBS_CONFIG_KEY: json.dumps(obs_config_to_dict(obs)),
        META_ACTION_CONFIG_KEY: json.dumps(action_config_to_dict(action)),
        META_OBS_DIM_KEY: str(obs.obs_dim),
        META_ACTION_DIM_KEY: "4",
        META_PROVENANCE_KEY: json.dumps({"run": "test"}),
    }
    props.update(overrides)
    return props


class TestOnnxPolicyMetadata:
    def test_no_crowdrl_keys_gives_none(self, tmp_path):
        policy = OnnxPolicy(_write_policy(tmp_path / "legacy.onnx"))
        assert policy.metadata is None

    def test_valid_metadata_parses_exactly(self, tmp_path):
        policy = OnnxPolicy(_write_policy(tmp_path / "p.onnx", props=_valid_props()))
        meta = policy.metadata
        assert meta is not None
        assert meta.obs_config == OBS
        assert meta.action_config == ACTION
        assert meta.obs_dim == 89
        assert meta.action_dim == 4
        assert meta.provenance == {"run": "test"}
        assert meta.schema_version == "1"

    def test_partial_record_raises(self, tmp_path):
        props = _valid_props()
        del props[META_ACTION_CONFIG_KEY]
        path = _write_policy(tmp_path / "p.onnx", props=props)
        with pytest.raises(ValueError, match="corrupt"):
            OnnxPolicy(path)

    def test_unparseable_json_raises(self, tmp_path):
        props = _valid_props(**{META_OBS_CONFIG_KEY: "{not json"})
        path = _write_policy(tmp_path / "p.onnx", props=props)
        with pytest.raises(ValueError, match="unreadable"):
            OnnxPolicy(path)

    def test_unknown_schema_version_raises(self, tmp_path):
        props = _valid_props(**{META_SCHEMA_KEY: "99"})
        path = _write_policy(tmp_path / "p.onnx", props=props)
        with pytest.raises(ValueError, match="schema version"):
            OnnxPolicy(path)

    def test_unknown_config_field_raises(self, tmp_path):
        """A field from a newer crowdrl-core must refuse, not default-fill."""
        obs_dict = obs_config_to_dict(OBS)
        obs_dict["use_quantum_sensing"] = True
        props = _valid_props(**{META_OBS_CONFIG_KEY: json.dumps(obs_dict)})
        path = _write_policy(tmp_path / "p.onnx", props=props)
        with pytest.raises(ValueError, match="use_quantum_sensing"):
            OnnxPolicy(path)

    def test_recorded_dim_vs_graph_width_raises(self, tmp_path):
        """metadata says 89 but the graph takes 80: self-contradictory file."""
        path = _write_policy(tmp_path / "p.onnx", obs_dim=80, props=_valid_props())
        with pytest.raises(ValueError, match="self-contradictory"):
            OnnxPolicy(path)

    def test_reconstruction_drift_raises(self, tmp_path):
        """Embedded config no longer derives the recorded width under this
        core: the reconstruction is not faithful."""
        props = _valid_props(**{META_OBS_DIM_KEY: "80"})  # config derives 89
        path = _write_policy(tmp_path / "p.onnx", obs_dim=80, props=props)
        with pytest.raises(ValueError, match="not.*faithful|faithful"):
            OnnxPolicy(path)


class _MetaPolicy(SimpleNamespace):
    """Duck-typed stand-in for a metadata-carrying policy backend."""

    def __call__(self, obs):  # pragma: no cover - never inferenced here
        return np.zeros(4)


def _meta_policy(obs=OBS, action=ACTION, graph_dim=None):
    from crowdrl_jupedsim.policy import PolicyMetadata

    return _MetaPolicy(
        metadata=PolicyMetadata(
            obs_config=obs,
            action_config=action,
            obs_dim=obs.obs_dim,
            action_dim=4,
            provenance=None,
            schema_version="1",
        ),
        obs_dim=graph_dim if graph_dim is not None else obs.obs_dim,
        metadata_capable=True,
    )


class TestResolveConfigs:
    def test_self_configures_from_metadata(self):
        resolved_obs, resolved_action = resolve_configs(_meta_policy())
        assert resolved_obs == OBS
        assert resolved_action == ACTION

    def test_agreeing_explicit_configs_pass_silently(self):
        with warnings_as_errors():
            resolved_obs, _ = resolve_configs(_meta_policy(), OBS, ACTION)
        assert resolved_obs == OBS

    def test_disagreeing_explicit_config_raises_with_field(self):
        flipped = ObsConfig(use_navmesh=True, use_goal_direction=True, use_temporal_memory=True)
        with pytest.raises(ValueError, match="use_goal_direction"):
            resolve_configs(_meta_policy(), flipped, ACTION)

    def test_legacy_onnx_with_explicit_configs_warns(self, tmp_path):
        policy = OnnxPolicy(_write_policy(tmp_path / "legacy.onnx"))
        with pytest.warns(UserWarning, match="cannot be verified"):
            resolved_obs, _ = resolve_configs(policy, OBS, ACTION)
        assert resolved_obs == OBS

    def test_legacy_onnx_without_configs_raises(self, tmp_path):
        policy = OnnxPolicy(_write_policy(tmp_path / "legacy.onnx"))
        with pytest.raises(ValueError, match="explicit"):
            resolve_configs(policy)

    def test_constant_policy_stays_silent(self):
        with warnings_as_errors():
            resolved_obs, _ = resolve_configs(ConstantPolicy([0, 0, 0, 0]), OBS, ACTION)
        assert resolved_obs == OBS

    def test_constant_policy_without_configs_raises(self):
        with pytest.raises(ValueError, match="explicit"):
            resolve_configs(ConstantPolicy([0, 0, 0, 0]))

    def test_resolved_width_must_match_graph(self, tmp_path):
        """Legacy artefact + wrong explicit config: the 89D-checkpoint-with-
        80D-config landmine from the handover, now loud."""
        policy = OnnxPolicy(_write_policy(tmp_path / "legacy.onnx", obs_dim=89))
        with pytest.warns(UserWarning):
            with pytest.raises(ValueError, match="input width"):
                resolve_configs(policy, ObsConfig(), ActionConfig())  # 80D config


class warnings_as_errors:
    """Context manager asserting no warning is emitted inside the block."""

    def __enter__(self):
        import warnings as w

        self._ctx = w.catch_warnings()
        self._ctx.__enter__()
        w.simplefilter("error")
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)
