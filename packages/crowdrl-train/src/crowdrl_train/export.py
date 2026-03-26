"""ONNX export pipeline for trained policies.

Exports the actor network (deterministic: action means only) with frozen
observation normalization baked in. This is the single artefact that crosses
from crowdrl-train to crowdrl-jupedsim.

The exported model:
- Input: (batch, obs_dim) float32 — raw (unnormalized) observations
- Output: (batch, action_dim) float32 — deterministic action means in [-1, 1]
- Dynamic batch axis: accepts any number of agents
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from crowdrl_train.networks import Actor
from crowdrl_train.normalizer import RunningNormalizer


class PolicyForExport(nn.Module):
    """Wrapper that prepends frozen normalization to the actor network.

    During deployment, observations come in raw (unnormalized). This module
    applies the frozen training-time normalization statistics before feeding
    through the actor to produce deterministic actions.
    """

    def __init__(self, actor: Actor, normalizer: RunningNormalizer | None = None):
        super().__init__()
        self.actor_feature_net = actor.feature_net
        self.actor_mean = actor.action_mean

        # Bake normalization statistics as buffers (not parameters)
        if normalizer is not None:
            self.register_buffer(
                "obs_mean",
                torch.tensor(normalizer.mean, dtype=torch.float32),
            )
            self.register_buffer(
                "obs_std",
                torch.tensor(np.sqrt(normalizer.var + normalizer.epsilon), dtype=torch.float32),
            )
            self.register_buffer(
                "obs_clip",
                torch.tensor(normalizer.clip, dtype=torch.float32),
            )
            self._has_normalizer = True
        else:
            self._has_normalizer = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Raw observations → deterministic action means.

        Parameters
        ----------
        obs : (batch, obs_dim) float32

        Returns
        -------
        (batch, action_dim) float32 — clipped to [-1, 1]
        """
        if self._has_normalizer:
            obs = torch.clamp(
                (obs - self.obs_mean) / self.obs_std,
                -self.obs_clip,
                self.obs_clip,
            )
        features = self.actor_feature_net(obs)
        action_mean = self.actor_mean(features)
        return action_mean.clamp(-1.0, 1.0)


def export_onnx(
    actor: Actor,
    normalizer: RunningNormalizer | None,
    output_path: str | Path,
    opset_version: int = 17,
) -> Path:
    """Export the policy to ONNX format.

    Parameters
    ----------
    actor : trained Actor network
    normalizer : observation normalizer (None = no normalization layer)
    output_path : path for the .onnx file
    opset_version : ONNX opset version

    Returns
    -------
    Path to the exported ONNX file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_model = PolicyForExport(actor, normalizer)
    export_model.eval()

    obs_dim = actor.config.obs_dim
    dummy_input = torch.randn(1, obs_dim)

    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["observations"],
        output_names=["actions"],
        dynamic_axes={
            "observations": {0: "n_agents"},
            "actions": {0: "n_agents"},
        },
    )

    return output_path


def verify_onnx(
    onnx_path: str | Path,
    actor: Actor,
    normalizer: RunningNormalizer | None,
    n_test_samples: int = 100,
    atol: float = 1e-5,
) -> bool:
    """Verify ONNX output matches PyTorch output on random inputs.

    Parameters
    ----------
    onnx_path : path to the .onnx file
    actor : the PyTorch actor used for export
    normalizer : the normalizer used for export
    n_test_samples : number of random test inputs
    atol : absolute tolerance for comparison

    Returns
    -------
    True if outputs match within tolerance
    """
    import onnxruntime as ort

    export_model = PolicyForExport(actor, normalizer)
    export_model.eval()

    session = ort.InferenceSession(str(onnx_path))
    obs_dim = actor.config.obs_dim

    test_obs = np.random.randn(n_test_samples, obs_dim).astype(np.float32)

    # PyTorch reference
    with torch.no_grad():
        pt_output = export_model(torch.from_numpy(test_obs)).numpy()

    # ONNX output
    ort_output = session.run(None, {"observations": test_obs})[0]

    return np.allclose(pt_output, ort_output, atol=atol)
