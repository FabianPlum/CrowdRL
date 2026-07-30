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
import math
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
    META_DYNAMICS_KEY,
    META_OBS_CONFIG_KEY,
    META_OBS_DIM_KEY,
    META_PROVENANCE_KEY,
    META_SCHEMA_KEY,
    SUPPORTED_SCHEMA_VERSIONS,
    action_config_from_dict,
    action_config_to_dict,
    obs_config_from_dict,
    obs_config_to_dict,
    validate_dynamics_dict,
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
    dynamics: dict | None = None
    """Env-level dynamics the policy was trained under (schema v2): any of
    desired_velocity_weight, max_velocity_magnitude, contact_stiffness,
    contact_damping. None on v1 artefacts (unrecorded)."""


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

    schema_version = raw.get(META_SCHEMA_KEY, "1")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"{source}: embedded config metadata has schema version "
            f"{schema_version!r} but this reader understands "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}. Upgrade crowdrl to deploy "
            "this artefact."
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
    dynamics = None
    if META_DYNAMICS_KEY in raw:
        try:
            dynamics = validate_dynamics_dict(json.loads(raw[META_DYNAMICS_KEY]))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{source}: embedded dynamics metadata is unreadable: {exc}") from exc

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
        dynamics=dynamics,
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


# Fallbacks for artefacts that record no dynamics (pre-schema-v2). These are
# the historical ADAPTER constants, deliberately NOT the current CrowdEnvConfig
# defaults: keeping them pinned means a v1 deployment does not silently change
# behaviour when the training defaults move. They are very unlikely to be the
# physics any current checkpoint trained under -- every recent best run trains
# at desired_velocity_weight=0.8 and clamps at 3.0 -- which is why falling back
# to them warns.
_DYNAMICS_DEFAULTS = {
    "desired_velocity_weight": 0.05,
    "max_velocity_magnitude": 5.0,
    "contact_stiffness": 30000.0,
    "contact_damping": 500.0,
}


def resolve_dynamics(policy: Policy, overrides: Mapping[str, float | None]) -> dict[str, float]:
    """Decide the env-level dynamics a model built on ``policy`` runs with.

    Same philosophy as :func:`resolve_configs`, per parameter:

    * artefact records it (schema v2 dynamics block) and nothing explicit is
      given: self-configure from the artefact;
    * artefact records it AND an explicit value is given: they must agree, or
      this raises rather than silently preferring either;
    * unrecorded: the explicit value, else the legacy adapter fallback from
      ``_DYNAMICS_DEFAULTS`` -- which warns for a metadata-capable policy,
      because running unverified physics is not a neutral default (the
      ``desired_velocity_weight`` gap alone, 0.05 vs the 0.8 of every current
      best run, is a ~16x change in the velocity-response time constant).

    ``overrides`` maps dynamics field names to explicit values or None.
    """
    unknown = sorted(set(overrides) - set(_DYNAMICS_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown dynamics parameter(s): {unknown}")

    metadata = getattr(policy, "metadata", None)
    recorded = metadata.dynamics if metadata is not None and metadata.dynamics else {}

    resolved: dict[str, float] = {}
    mismatches = []
    fell_back = []
    for field_name, default in _DYNAMICS_DEFAULTS.items():
        explicit = overrides.get(field_name)
        stored = recorded.get(field_name)
        # Relative comparison: an absolute 1e-12 is sub-ULP at contact-stiffness
        # scale (ulp(30000.0) = 3.6e-12), so two adjacent doubles -- the same
        # value after a serialisation round-trip -- would raise as a mismatch.
        if (
            explicit is not None
            and stored is not None
            and not math.isclose(explicit, stored, rel_tol=1e-9, abs_tol=1e-12)
        ):
            mismatches.append(f"{field_name}: explicit {explicit!r} != embedded {stored!r}")
        if explicit is not None:
            resolved[field_name] = float(explicit)
        elif stored is not None:
            resolved[field_name] = float(stored)
        else:
            resolved[field_name] = default
            fell_back.append(field_name)
    if mismatches:
        raise ValueError(
            "explicit dynamics disagree with the artefact's embedded training "
            "dynamics -- refusing to silently prefer either. " + "; ".join(mismatches)
        )
    if fell_back and getattr(policy, "metadata_capable", False):
        warnings.warn(
            "this artefact records no trained value for "
            + ", ".join(f"{f} (using {resolved[f]!r})" for f in fell_back)
            + ". These are legacy adapter fallbacks, not the artefact's training "
            "physics: re-export with scripts/reexport_onnx.py to embed the real "
            "dynamics, or pass them explicitly.",
            UserWarning,
            stacklevel=2,
        )
    return resolved
