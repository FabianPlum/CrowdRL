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

from dataclasses import asdict, fields
from typing import Any, Mapping

from crowdrl_core.action import ActionConfig
from crowdrl_core.observation import ObsConfig
from crowdrl_core.sensing import RaycastConfig

__all__ = [
    "META_ACTION_CONFIG_KEY",
    "META_ACTION_DIM_KEY",
    "META_OBS_CONFIG_KEY",
    "META_OBS_DIM_KEY",
    "META_PROVENANCE_KEY",
    "META_SCHEMA_KEY",
    "METADATA_SCHEMA_VERSION",
    "action_config_from_dict",
    "action_config_to_dict",
    "obs_config_from_dict",
    "obs_config_to_dict",
]

# ONNX metadata_props keys for the embedded training configuration (issue #7).
# Written by crowdrl_train.export, read by crowdrl_jupedsim.policy; both sides
# import the names from here so they cannot disagree. Bump
# METADATA_SCHEMA_VERSION when the payload semantics change -- readers refuse
# versions they do not know.
METADATA_SCHEMA_VERSION = "1"
META_SCHEMA_KEY = "crowdrl.schema_version"
META_OBS_CONFIG_KEY = "crowdrl.obs_config"
META_ACTION_CONFIG_KEY = "crowdrl.action_config"
META_OBS_DIM_KEY = "crowdrl.obs_dim"
META_ACTION_DIM_KEY = "crowdrl.action_dim"
META_PROVENANCE_KEY = "crowdrl.provenance"


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
