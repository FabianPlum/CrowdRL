"""Tests for ONNX export pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from crowdrl_train.export import PolicyForExport, export_onnx
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
