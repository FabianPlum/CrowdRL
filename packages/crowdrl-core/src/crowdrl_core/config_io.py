"""Lossless dict serialization for the perception/action configs.

This is the single serialization boundary for ``ObsConfig`` (with its nested
``RaycastConfig``) and ``ActionConfig`` -- the two configs that MUST travel
with a trained policy for it to be deployable (see issue #7). The dicts are
plain JSON-compatible values (``dataclasses.asdict`` output, raw radians, no
unit translation), so ``to -> json.dumps -> json.loads -> from`` reproduces
the exact dataclass. Both the ONNX exporter (``crowdrl_train.export``) and the
deployment reader (``crowdrl_jupedsim.policy``) go through these functions;
neither side carries its own field mapping, so the two cannot drift apart.

This is deliberately NOT the ``config_resolved.yaml`` schema. That YAML is the
human-authored training schema (degree-denominated, partial coverage, parsed
by ``train_mappo.build_env_config``); this module is the machine-faithful
record of what a checkpoint was actually trained with.

Version-drift policy, asymmetric on purpose:

* **Missing keys fill from dataclass defaults.** A field added to the config
  after an artefact was exported did not exist at training time, and config
  evolution in this repo keeps new features default-off -- so defaults
  reproduce the training-time behaviour.
* **Unknown keys raise.** The artefact was produced by a NEWER config surface
  than this crowdrl-core knows: the training run used a feature this version
  cannot reproduce, and observation parity cannot be guaranteed. Failing loud
  here is the entire point of embedding the config.
"""

from __future__ import annotations

import math
from dataclasses import asdict, fields
from typing import Any, Mapping

from crowdrl_core.action import ActionConfig
from crowdrl_core.observation import ObsConfig
from crowdrl_core.sensing import RaycastConfig

__all__ = [
    "DYNAMICS_FIELDS",
    "META_ACTION_CONFIG_KEY",
    "META_ACTION_DIM_KEY",
    "META_DYNAMICS_KEY",
    "META_OBS_CONFIG_KEY",
    "META_OBS_DIM_KEY",
    "META_PROVENANCE_KEY",
    "META_SCHEMA_KEY",
    "METADATA_SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "action_config_from_dict",
    "action_config_to_dict",
    "obs_config_from_dict",
    "obs_config_to_dict",
    "validate_dynamics_dict",
]

# ONNX metadata_props keys for the embedded training configuration (issue #7).
# Written by crowdrl_train.export, read by crowdrl_jupedsim.policy; both sides
# import the names from here so they cannot disagree. Bump
# METADATA_SCHEMA_VERSION when the payload semantics change -- readers refuse
# versions they do not know.
#
# Schema history:
#   "1": obs/action configs, dims, provenance.
#   "2": adds the OPTIONAL crowdrl.dynamics block -- the env-level dynamics
#        the policy was trained under (velocity-filter weight, speed clamp,
#        contact constants). Readers accept both; a v2 file without the
#        dynamics key is valid (dynamics simply unrecorded).
METADATA_SCHEMA_VERSION = "2"
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1", "2"})
META_SCHEMA_KEY = "crowdrl.schema_version"
META_OBS_CONFIG_KEY = "crowdrl.obs_config"
META_ACTION_CONFIG_KEY = "crowdrl.action_config"
META_OBS_DIM_KEY = "crowdrl.obs_dim"
META_ACTION_DIM_KEY = "crowdrl.action_dim"
META_PROVENANCE_KEY = "crowdrl.provenance"
META_DYNAMICS_KEY = "crowdrl.dynamics"

DYNAMICS_FIELDS = frozenset(
    {
        "desired_velocity_weight",
        "max_velocity_magnitude",
        "contact_stiffness",
        "contact_damping",
    }
)
"""The env-level dynamics parameters that shape the trained motion but live
outside ObsConfig/ActionConfig (they are CrowdEnvConfig fields). A policy
deployed under different values runs different physics than it was trained
under -- the desired_velocity_weight filter alone changes the response time
constant by more than an order of magnitude between the historical 0.8 runs
and the Layer-1 0.05 default."""


def validate_dynamics_dict(data: Mapping[str, Any]) -> dict[str, float]:
    """Validate a dynamics block: known keys only, finite non-negative floats.

    Missing keys are allowed (unrecorded parameters fall back to defaults on
    the read side); unknown keys raise, same asymmetry as the configs.

    Values are checked rather than merely coerced. A payload that is valid JSON
    but not an object (``[]``, ``3``, ``"x"``) raises here instead of surfacing
    as an ``AttributeError`` from the caller, and NaN/inf/negative values are
    rejected outright: all four fields are physically non-negative, and a NaN
    that reached the physics would propagate silently through every position.
    """
    if not isinstance(data, Mapping):
        raise ValueError(
            f"dynamics block must be a JSON object, got {type(data).__name__}. "
            "The artefact's metadata is malformed."
        )
    unknown = sorted(set(data) - DYNAMICS_FIELDS)
    if unknown:
        raise ValueError(
            f"dynamics block carries unknown field(s) {unknown}: the artefact "
            "was exported by a newer crowdrl than this one. Upgrade crowdrl "
            "to at least the version that exported it."
        )
    validated: dict[str, float] = {}
    for key, value in data.items():
        # bool is an int subclass, so float(True) == 1.0 would sail through.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"dynamics field {key!r} must be a number, got {value!r} ({type(value).__name__})."
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"dynamics field {key!r} must be finite, got {number!r}.")
        if number < 0.0:
            raise ValueError(f"dynamics field {key!r} must be non-negative, got {number!r}.")
        validated[key] = number
    return validated


def _checked_kwargs(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``data`` as ctor kwargs for ``cls``, raising on unknown keys."""
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"{cls.__name__} data carries unknown field(s) {unknown}: the "
            "artefact was exported by a newer crowdrl-core than this one, so "
            "its training configuration cannot be faithfully reconstructed. "
            "Upgrade crowdrl-core to at least the version that exported it."
        )
    return dict(data)


def obs_config_to_dict(config: ObsConfig) -> dict[str, Any]:
    """``ObsConfig`` -> JSON-compatible dict (nested raycast included)."""
    return asdict(config)


def obs_config_from_dict(data: Mapping[str, Any]) -> ObsConfig:
    """Exact inverse of :func:`obs_config_to_dict`."""
    kwargs = _checked_kwargs(ObsConfig, data)
    raycast = kwargs.get("raycast")
    if isinstance(raycast, Mapping):
        kwargs["raycast"] = RaycastConfig(**_checked_kwargs(RaycastConfig, raycast))
    return ObsConfig(**kwargs)


def action_config_to_dict(config: ActionConfig) -> dict[str, Any]:
    """``ActionConfig`` -> JSON-compatible dict (raw radians, no deg mapping)."""
    return asdict(config)


def action_config_from_dict(data: Mapping[str, Any]) -> ActionConfig:
    """Exact inverse of :func:`action_config_to_dict`."""
    return ActionConfig(**_checked_kwargs(ActionConfig, data))
