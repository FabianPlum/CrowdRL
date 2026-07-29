"""Tests for ONNX export pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from crowdrl_core.action import ActionConfig
from crowdrl_core.config_io import action_config_from_dict, obs_config_from_dict
from crowdrl_core.observation import ObsConfig

from crowdrl_train.export import (
    META_ACTION_CONFIG_KEY,
    META_ACTION_DIM_KEY,
    META_OBS_CONFIG_KEY,
    META_OBS_DIM_KEY,
    META_PROVENANCE_KEY,
    META_SCHEMA_KEY,
    PolicyForExport,
    export_onnx,
)
from crowdrl_train.networks import Actor, ActorCritic
from crowdrl_train.normalizer import RunningNormalizer


class TestPolicyForExport:
    """Unit tests for the PolicyForExport wrapper."""

    def test_does_not_mutate_source_actor_device(self, tiny_actor: Actor):
        """Regression: PolicyForExport must deep-copy the actor modules.

        Previously it held references, so calling ``.cpu()`` on the wrapper
        silently moved the original actor's parameters to CPU -- breaking any
        subsequent GPU operation on the training model.
        """
        # Move source actor to a known device and record it
        tiny_actor.to("cpu")
        original_device = next(tiny_actor.parameters()).device
        assert original_device.type == "cpu"

        # Build wrapper, then move wrapper to CPU (this was the trigger)
        wrapper = PolicyForExport(tiny_actor, normalizer=None)
        wrapper.cpu().eval()

        # Source actor parameters must remain on the original device
        for p in tiny_actor.parameters():
            assert p.device == original_device

        # Wrapper parameters must be independent copies
        src_params = list(tiny_actor.feature_net.parameters())
        wrap_params = list(wrapper.actor_feature_net.parameters())
        assert len(src_params) == len(wrap_params)
        for sp, wp in zip(src_params, wrap_params):
            assert sp is not wp  # not the same object

    def test_wrapper_matches_actor_forward(self, tiny_actor: Actor):
        """Numerical check: wrapper output matches actor_mean from the source."""
        obs = torch.randn(5, tiny_actor.config.obs_dim)

        wrapper = PolicyForExport(tiny_actor, normalizer=None).eval()

        with torch.no_grad():
            wrap_out = wrapper(obs)

            # Reference: run through the actor's feature_net + action_mean
            features = tiny_actor.feature_net(obs)
            expected = tiny_actor.action_mean(features).clamp(-1.0, 1.0)

        torch.testing.assert_close(wrap_out, expected)


class TestExportOnnx:
    """End-to-end export tests."""

    def test_export_does_not_mutate_actor_device(
        self, tiny_actor_critic: ActorCritic, tmp_path: Path
    ):
        """Calling export_onnx must leave the source actor on its original device."""
        actor = tiny_actor_critic.actor
        actor.to("cpu")
        expected_device = next(actor.parameters()).device

        export_onnx(actor, normalizer=None, output_path=tmp_path / "policy.onnx")

        # Source actor parameters unchanged
        for p in actor.parameters():
            assert p.device == expected_device

        assert (tmp_path / "policy.onnx").exists()

    def test_export_with_normalizer(self, tiny_actor_critic: ActorCritic, tmp_path: Path):
        """Export with a normalizer (the common path) also preserves device."""
        actor = tiny_actor_critic.actor
        actor.to("cpu")

        normalizer = RunningNormalizer(shape=(actor.config.obs_dim,))
        normalizer.update(np.random.randn(100, actor.config.obs_dim))

        export_onnx(actor, normalizer, output_path=tmp_path / "policy.onnx")

        for p in actor.parameters():
            assert p.device.type == "cpu"

        assert (tmp_path / "policy.onnx").exists()
        assert (tmp_path / "policy.onnx").stat().st_size > 0


class TestEmbeddedMetadata:
    """The .onnx must carry its own training configuration (issue #7).

    The default ``ObsConfig()`` derives obs_dim=80, matching the tiny_actor
    fixture, so the happy paths pair those; the mismatch test flips a
    width-changing flag.
    """

    ACTION = ActionConfig(
        max_heading_change=float(np.radians(4.8)),
        speed_turn_coupling=True,
    )

    def test_metadata_embeds_and_round_trips(self, tiny_actor: Actor, tmp_path: Path):
        path = tmp_path / "policy.onnx"
        export_onnx(
            tiny_actor,
            normalizer=None,
            output_path=path,
            obs_config=ObsConfig(),
            action_config=self.ACTION,
            provenance={"run": "unit-test", "git_rev": "deadbeef"},
        )

        import onnx

        props = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
        assert props[META_SCHEMA_KEY] == "1"
        assert obs_config_from_dict(json.loads(props[META_OBS_CONFIG_KEY])) == ObsConfig()
        assert action_config_from_dict(json.loads(props[META_ACTION_CONFIG_KEY])) == self.ACTION
        assert props[META_OBS_DIM_KEY] == "80"
        assert props[META_ACTION_DIM_KEY] == "4"
        assert json.loads(props[META_PROVENANCE_KEY]) == {
            "run": "unit-test",
            "git_rev": "deadbeef",
        }

    def test_metadata_write_leaves_graph_runnable(self, tiny_actor: Actor, tmp_path: Path):
        """Rewriting the file for metadata must not corrupt the graph."""
        path = tmp_path / "policy.onnx"
        export_onnx(
            tiny_actor,
            normalizer=None,
            output_path=path,
            obs_config=ObsConfig(),
            action_config=self.ACTION,
        )

        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        out = session.run(None, {"observations": np.zeros((3, 80), dtype=np.float32)})[0]
        assert out.shape == (3, 4)
        # And the metadata is visible through onnxruntime, the deployment API.
        assert META_OBS_CONFIG_KEY in session.get_modelmeta().custom_metadata_map

    def test_legacy_export_carries_no_crowdrl_keys(self, tiny_actor: Actor, tmp_path: Path):
        """Config-less export keeps producing the pre-#7 artefact."""
        path = tmp_path / "policy.onnx"
        export_onnx(tiny_actor, normalizer=None, output_path=path)

        import onnx

        keys = {p.key for p in onnx.load(str(path)).metadata_props}
        assert not any(k.startswith("crowdrl.") for k in keys)

    def test_obs_dim_mismatch_refuses_to_export(self, tiny_actor: Actor, tmp_path: Path):
        """A config that derives a different width than the actor is not the
        training config; nothing may be written."""
        path = tmp_path / "policy.onnx"
        wrong = ObsConfig(use_navmesh=True)  # 83D != the actor's 80D
        with pytest.raises(ValueError, match="obs_dim"):
            export_onnx(
                tiny_actor,
                normalizer=None,
                output_path=path,
                obs_config=wrong,
                action_config=self.ACTION,
            )
        assert not path.exists()

    def test_half_configured_export_refuses(self, tiny_actor: Actor, tmp_path: Path):
        with pytest.raises(ValueError, match="together"):
            export_onnx(
                tiny_actor,
                normalizer=None,
                output_path=tmp_path / "policy.onnx",
                obs_config=ObsConfig(),
            )
