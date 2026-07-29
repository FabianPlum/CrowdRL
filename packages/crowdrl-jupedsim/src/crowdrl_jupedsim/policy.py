"""Policy inference backends for the JuPedSim deployment adapter.

The adapter depends only on an exported ``.onnx`` artefact -- never on PyTorch,
crowdrl-env or crowdrl-train. ``Policy`` is the narrow seam between the
operational model and whatever produces actions, which keeps the adapter
testable without a trained checkpoint.

Since issue #7, exported artefacts embed their training configuration in
``metadata_props``. ``OnnxPolicy`` reconstructs it (:class:`PolicyMetadata`)
and :func:`resolve_configs` turns it into the configs the operational model
runs with -- self-configuring from the artefact when possible, and refusing
loudly whenever two sources of truth disagree.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from crowdrl_core.action import ActionConfig
from crowdrl_core.config_io import (
    META_ACTION_CONFIG_KEY,
    META_ACTION_DIM_KEY,
    META_OBS_CONFIG_KEY,
    META_OBS_DIM_KEY,
    META_PROVENANCE_KEY,
    META_SCHEMA_KEY,
    METADATA_SCHEMA_VERSION,
    action_config_from_dict,
    action_config_to_dict,
    obs_config_from_dict,
    obs_config_to_dict,
)
from crowdrl_core.observation import ObsConfig


@runtime_checkable
class Policy(Protocol):
    """Maps a single observation vector to a raw action vector in [-1, 1]."""

    def __call__(self, obs: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class PolicyMetadata:
    """Training configuration reconstructed from an artefact's metadata."""

    obs_config: ObsConfig
    action_config: ActionConfig
    obs_dim: int | None
    action_dim: int | None
    provenance: dict | None
    schema_version: str


def _parse_metadata(raw: Mapping[str, str], source: str) -> PolicyMetadata | None:
    """Parse ``metadata_props`` into a :class:`PolicyMetadata`.

    Returns None for artefacts that simply predate the embedding (no
    ``crowdrl.*`` keys at all). Anything in between -- partial keys, invalid
    JSON, an unknown schema version, configs this crowdrl-core cannot
    reconstruct -- raises: a half-readable config record must never be
    silently downgraded to "no record".
    """
    crowdrl_keys = {k for k in raw if k.startswith("crowdrl.")}
    if not crowdrl_keys:
        return None

    schema_version = raw.get(META_SCHEMA_KEY, METADATA_SCHEMA_VERSION)
    if schema_version != METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"{source}: embedded config metadata has schema version "
            f"{schema_version!r} but this reader understands "
            f"{METADATA_SCHEMA_VERSION!r}. Upgrade crowdrl to deploy this "
            "artefact."
        )
    if META_OBS_CONFIG_KEY not in raw or META_ACTION_CONFIG_KEY not in raw:
        raise ValueError(
            f"{source}: artefact carries crowdrl metadata keys {sorted(crowdrl_keys)} "
            "but not both configs -- the config record is corrupt."
        )

    try:
        obs_config = obs_config_from_dict(json.loads(raw[META_OBS_CONFIG_KEY]))
        action_config = action_config_from_dict(json.loads(raw[META_ACTION_CONFIG_KEY]))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{source}: embedded config metadata is unreadable: {exc}") from exc

    obs_dim = int(raw[META_OBS_DIM_KEY]) if META_OBS_DIM_KEY in raw else None
    action_dim = int(raw[META_ACTION_DIM_KEY]) if META_ACTION_DIM_KEY in raw else None
    provenance = json.loads(raw[META_PROVENANCE_KEY]) if META_PROVENANCE_KEY in raw else None

    if obs_dim is not None and obs_config.obs_dim != obs_dim:
        raise ValueError(
            f"{source}: the embedded ObsConfig derives obs_dim="
            f"{obs_config.obs_dim} under this crowdrl-core, but the artefact "
            f"recorded obs_dim={obs_dim} at export. The reconstruction is not "
            "faithful (version drift); refusing to deploy."
        )

    return PolicyMetadata(
        obs_config=obs_config,
        action_config=action_config,
        obs_dim=obs_dim,
        action_dim=action_dim,
        provenance=provenance,
        schema_version=schema_version,
    )


def _flat_items(prefix: str, data: Mapping) -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for key, value in data.items():
        dotted = f"{prefix}.{key}"
        if isinstance(value, Mapping):
            out.extend(_flat_items(dotted, value))
        else:
            out.append((dotted, value))
    return out


def _diff_configs(name: str, explicit, embedded, to_dict) -> list[str]:
    """Field-level differences, dotted through nested configs."""
    if explicit == embedded:
        return []
    flat_explicit = dict(_flat_items(name, to_dict(explicit)))
    flat_embedded = dict(_flat_items(name, to_dict(embedded)))
    return [
        f"{key}: explicit {flat_explicit[key]!r} != embedded {flat_embedded[key]!r}"
        for key in flat_explicit
        if flat_explicit[key] != flat_embedded[key]
    ]


def resolve_configs(
    policy: Policy,
    obs_config: ObsConfig | None = None,
    action_config: ActionConfig | None = None,
) -> tuple[ObsConfig, ActionConfig]:
    """Decide which ObsConfig/ActionConfig a model built on ``policy`` runs with.

    * Artefact carries metadata, nothing explicit given: self-configure.
    * Artefact carries metadata AND explicit configs given: they must agree
      field-for-field; any disagreement raises rather than silently preferring
      either source.
    * No metadata (legacy artefact): explicit configs are required, and a
      warning records that they are unverified. Backends that can never carry
      metadata (``ConstantPolicy``) stay silent.

    Independent of the source, the resolved ObsConfig must derive the same
    observation width as the policy's actual input, when the backend knows it.
    """
    metadata = getattr(policy, "metadata", None)

    if metadata is None:
        if obs_config is None or action_config is None:
            raise ValueError(
                "policy carries no embedded config metadata, so explicit "
                "obs_config AND action_config are required. For .onnx files "
                "exported before issue #7, pass the configs rebuilt from the "
                "run's config_resolved.yaml."
            )
        if getattr(policy, "metadata_capable", False):
            warnings.warn(
                f"{policy!r} carries no embedded config metadata (pre-#7 "
                "artefact): deploying with hand-supplied configs that cannot "
                "be verified against the checkpoint.",
                UserWarning,
                stacklevel=2,
            )
        resolved_obs, resolved_action = obs_config, action_config
    else:
        mismatches = (
            _diff_configs("obs_config", obs_config, metadata.obs_config, obs_config_to_dict)
            if obs_config is not None
            else []
        )
        if action_config is not None:
            mismatches += _diff_configs(
                "action_config", action_config, metadata.action_config, action_config_to_dict
            )
        if mismatches:
            raise ValueError(
                "explicit config disagrees with the artefact's embedded "
                "training config -- refusing to silently prefer either. " + "; ".join(mismatches)
            )
        resolved_obs = obs_config if obs_config is not None else metadata.obs_config
        resolved_action = action_config if action_config is not None else metadata.action_config

    graph_dim = getattr(policy, "obs_dim", None)
    if graph_dim is not None and resolved_obs.obs_dim != graph_dim:
        raise ValueError(
            f"resolved ObsConfig derives obs_dim={resolved_obs.obs_dim} but "
            f"the policy's input width is {graph_dim}: this is not the config "
            "the checkpoint was trained with."
        )
    return resolved_obs, resolved_action


class OnnxPolicy:
    """ONNX Runtime policy loaded from an exported CrowdRL checkpoint.

    The exported graph bakes in the frozen observation normalizer and the
    deterministic ``tanh(mean)`` action, so this wrapper is a thin call.

    The policy file is loaded by path and never bundled with the package --
    baseline weights live outside the repository.

    Artefacts exported since issue #7 carry their training configuration in
    ``metadata_props``; it is parsed into ``self.metadata`` at load and
    cross-checked against the graph's actual input width. Older artefacts get
    ``metadata = None`` and must be deployed with explicit configs.
    """

    metadata_capable = True
    """This backend can carry embedded config metadata (a missing record on an
    OnnxPolicy is a legacy artefact, not an impossibility)."""

    def __init__(self, onnx_path: str | Path, providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        self.path = Path(onnx_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"ONNX policy not found: {self.path}")

        self._session = ort.InferenceSession(
            str(self.path), providers=providers or ["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

        # Static obs width from the graph, when the exporter pinned it. Used to
        # fail loudly on an ObsConfig/checkpoint mismatch instead of silently
        # feeding the policy a differently-shaped world.
        shape = self._session.get_inputs()[0].shape
        last = shape[-1] if shape else None
        self.obs_dim: int | None = last if isinstance(last, int) else None

        self.metadata: PolicyMetadata | None = _parse_metadata(
            self._session.get_modelmeta().custom_metadata_map, source=str(self.path)
        )
        if (
            self.metadata is not None
            and self.metadata.obs_dim is not None
            and self.obs_dim is not None
            and self.metadata.obs_dim != self.obs_dim
        ):
            raise ValueError(
                f"{self.path}: metadata records obs_dim={self.metadata.obs_dim} "
                f"but the graph input width is {self.obs_dim} -- the artefact "
                "is self-contradictory (edited after export?)."
            )

    def __call__(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        batched = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        out = self._session.run(None, {self._input_name: batched})[0]
        return np.asarray(out, dtype=np.float64).reshape(-1)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"OnnxPolicy(path={self.path.name!r}, obs_dim={self.obs_dim})"


class ConstantPolicy:
    """Returns a fixed action regardless of observation.

    Exists so the adapter's control flow (WorldState assembly, observation
    construction, action interpretation, integration) can be exercised without
    a trained checkpoint.
    """

    metadata_capable = False
    """A constant action has no checkpoint, so there is nothing to verify
    explicit configs against -- resolve_configs stays silent."""

    metadata = None

    def __init__(self, action: NDArray[np.float64] | list[float]) -> None:
        self._action = np.asarray(action, dtype=np.float64)

    def __call__(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._action.copy()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ConstantPolicy(action={self._action.tolist()})"
