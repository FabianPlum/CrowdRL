"""ONNX export pipeline for trained policies.

Exports the actor network (deterministic: tanh(mean)) with frozen
observation normalization baked in. This is the single artefact that crosses
from crowdrl-train to crowdrl-jupedsim.

The exported model:
- Input: (batch, obs_dim) float32 — raw (unnormalized) observations
- Output: (batch, action_dim) float32 — deterministic tanh-squashed action,
  ``tanh(mean)`` in (-1, 1) (matches the training-time deterministic action)
- Dynamic batch axis: accepts any number of agents
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from crowdrl_core.action import ActionConfig
from crowdrl_core.config_io import action_config_to_dict, obs_config_to_dict
from crowdrl_core.observation import ObsConfig

from crowdrl_train.networks import Actor
from crowdrl_train.normalizer import RunningNormalizer

# metadata_props keys for the embedded training configuration (issue #7).
# The deployment reader (crowdrl_jupedsim.policy.OnnxPolicy) matches on these
# exact names; bump METADATA_SCHEMA_VERSION when their meaning changes.
METADATA_SCHEMA_VERSION = "1"
META_SCHEMA_KEY = "crowdrl.schema_version"
META_OBS_CONFIG_KEY = "crowdrl.obs_config"
META_ACTION_CONFIG_KEY = "crowdrl.action_config"
META_OBS_DIM_KEY = "crowdrl.obs_dim"
META_ACTION_DIM_KEY = "crowdrl.action_dim"
META_PROVENANCE_KEY = "crowdrl.provenance"


class PolicyForExport(nn.Module):
    """Wrapper that prepends frozen normalization to the actor network.

    During deployment, observations come in raw (unnormalized). This module
    applies the frozen training-time normalization statistics before feeding
    through the actor to produce deterministic actions.
    """

    def __init__(self, actor: Actor, normalizer: RunningNormalizer | None = None):
        super().__init__()
        # Deep-copy so subsequent .cpu()/.to() calls on this wrapper do not
        # mutate the original actor's parameters (which would silently move
        # a GPU-resident training model to CPU mid-pipeline).
        self.actor_feature_net = copy.deepcopy(actor.feature_net)
        self.actor_mean = copy.deepcopy(actor.action_mean)

        # Bake normalization statistics as buffers (not parameters)
        if normalizer is not None:
            mean = normalizer.mean
            var = normalizer.var
            # Handle both numpy arrays and torch tensors (possibly on GPU)
            if isinstance(var, torch.Tensor):
                std = torch.sqrt(var + normalizer.epsilon).cpu().float()
            else:
                std = torch.tensor(np.sqrt(var + normalizer.epsilon), dtype=torch.float32)
            if isinstance(mean, torch.Tensor):
                mean = mean.cpu().float()
            else:
                mean = torch.tensor(mean, dtype=torch.float32)
            self.register_buffer("obs_mean", mean)
            self.register_buffer("obs_std", std)
            clip = normalizer.clip
            if isinstance(clip, torch.Tensor):
                clip = clip.cpu().float()
            else:
                clip = torch.tensor(clip, dtype=torch.float32)
            self.register_buffer("obs_clip", clip)
            self._has_normalizer = True
        else:
            self._has_normalizer = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Raw observations → deterministic tanh-squashed action.

        Mirrors the training-time deterministic action ``tanh(mean)`` so the
        deployed (JuPedSim) policy is identical to the trained one.

        Parameters
        ----------
        obs : (batch, obs_dim) float32

        Returns
        -------
        (batch, action_dim) float32 — tanh-squashed into (-1, 1)
        """
        if self._has_normalizer:
            obs = torch.clamp(
                (obs - self.obs_mean) / self.obs_std,
                -self.obs_clip,
                self.obs_clip,
            )
        features = self.actor_feature_net(obs)
        action_mean = self.actor_mean(features)
        return torch.tanh(action_mean)


def export_onnx(
    actor: Actor,
    normalizer: RunningNormalizer | None,
    output_path: str | Path,
    opset_version: int = 17,
    *,
    obs_config: ObsConfig | None = None,
    action_config: ActionConfig | None = None,
    provenance: dict | None = None,
) -> Path:
    """Export the policy to ONNX format.

    When ``obs_config``/``action_config`` are given (pass both or neither),
    the resolved training configuration is embedded in the file's
    ``metadata_props`` (issue #7), making the ``.onnx`` self-describing: the
    deployment adapter reconstructs the exact perception/action configs from
    the artefact instead of trusting a hand-supplied copy. Files exported
    without them carry no config record and deploy as unverified legacy
    artefacts.

    Parameters
    ----------
    actor : trained Actor network
    normalizer : observation normalizer (None = no normalization layer)
    output_path : path for the .onnx file
    opset_version : ONNX opset version
    obs_config : the resolved ObsConfig the policy was trained with. Its
        ``obs_dim`` must match the actor's input width -- a mismatch raises
        before anything is written, the earliest loud failure point.
    action_config : the resolved ActionConfig the policy was trained with
    provenance : free-form JSON-compatible dict recording where the artefact
        came from (run id, git rev, rollout, ...)

    Returns
    -------
    Path to the exported ONNX file
    """
    if (obs_config is None) != (action_config is None):
        raise ValueError(
            "Pass obs_config and action_config together (or neither): a policy "
            "artefact with half its configuration embedded cannot be deployed."
        )
    if obs_config is not None and obs_config.obs_dim != actor.config.obs_dim:
        raise ValueError(
            f"obs_config.obs_dim ({obs_config.obs_dim}) != actor obs_dim "
            f"({actor.config.obs_dim}): this ObsConfig is not the one the "
            "checkpoint was trained with. Refusing to export a self-"
            "contradictory artefact."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_model = PolicyForExport(actor, normalizer)
    export_model.cpu().eval()

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

    if obs_config is not None and action_config is not None:
        props = {
            META_SCHEMA_KEY: METADATA_SCHEMA_VERSION,
            META_OBS_CONFIG_KEY: json.dumps(obs_config_to_dict(obs_config)),
            META_ACTION_CONFIG_KEY: json.dumps(action_config_to_dict(action_config)),
            META_OBS_DIM_KEY: str(obs_config.obs_dim),
            META_ACTION_DIM_KEY: str(actor.config.action_dim),
        }
        if provenance is not None:
            props[META_PROVENANCE_KEY] = json.dumps(provenance)
        _embed_metadata(output_path, props)

    return output_path


def _embed_metadata(onnx_path: Path, props: dict[str, str]) -> None:
    """Write ``props`` into the file's ``metadata_props``, in place."""
    import onnx

    model = onnx.load(str(onnx_path))
    for key, value in props.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, str(onnx_path))


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
    export_model.cpu().eval()

    session = ort.InferenceSession(str(onnx_path))
    obs_dim = actor.config.obs_dim

    test_obs = np.random.randn(n_test_samples, obs_dim).astype(np.float32)

    # PyTorch reference
    with torch.no_grad():
        pt_output = export_model(torch.from_numpy(test_obs)).numpy()

    # ONNX output
    ort_output = session.run(None, {"observations": test_obs})[0]

    return np.allclose(pt_output, ort_output, atol=atol)
