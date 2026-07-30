"""The re-export script must never certify guessed dynamics as trained.

Schema-v2 metadata states four env-level physics constants as the ones the
policy trained under, but ``config_resolved.yaml`` records only
``desired_velocity_weight``. Anything else has to be supplied explicitly or
explicitly waived -- otherwise a re-export of an older run stamps present-day
``CrowdEnvConfig`` defaults as that run's physics, which is precisely the drift
schema v2 exists to prevent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from crowdrl_env.crowd_env import CrowdEnvConfig  # noqa: E402

from reexport_onnx import (  # noqa: E402
    DYNAMICS_FIELDS,
    DynamicsProvenanceError,
    resolve_dynamics,
)

NO_OVERRIDES = dict.fromkeys(DYNAMICS_FIELDS)


@pytest.fixture
def env_config():
    return CrowdEnvConfig()


def test_unrecorded_fields_are_refused(env_config):
    """The three fields the YAML cannot express must not be silently defaulted."""
    cfg = {"desired_velocity_weight": 0.8}
    with pytest.raises(DynamicsProvenanceError) as exc:
        resolve_dynamics(cfg, env_config, NO_OVERRIDES)
    message = str(exc.value)
    for field in ("max_velocity_magnitude", "contact_stiffness", "contact_damping"):
        assert field in message
    # The message must be actionable: it names the flags and the escape hatch.
    assert "--max-velocity-magnitude" in message
    assert "--assume-current-defaults" in message


def test_missing_desired_velocity_weight_is_refused(env_config):
    """A YAML without the key meant 0.8 before the Layer-1 refactor; today's
    parser fills 0.05, so a silent default would fabricate a 16x difference."""
    with pytest.raises(DynamicsProvenanceError, match="desired_velocity_weight"):
        resolve_dynamics({}, env_config, NO_OVERRIDES, assume_current_defaults=False)


def test_explicit_values_resolve_with_explicit_provenance(env_config):
    cfg = {"desired_velocity_weight": 0.8}
    overrides = {
        **NO_OVERRIDES,
        "max_velocity_magnitude": 3.0,
        "contact_stiffness": 30000.0,
        "contact_damping": 500.0,
    }
    dynamics, provenance = resolve_dynamics(cfg, env_config, overrides)

    assert dynamics == {
        "desired_velocity_weight": 0.8,
        "max_velocity_magnitude": 3.0,
        "contact_stiffness": 30000.0,
        "contact_damping": 500.0,
    }
    assert provenance == {
        "desired_velocity_weight": "config_resolved.yaml",
        "max_velocity_magnitude": "explicit",
        "contact_stiffness": "explicit",
        "contact_damping": "explicit",
    }


def test_waiver_stamps_assumed_default(env_config):
    cfg = {"desired_velocity_weight": 0.8}
    dynamics, provenance = resolve_dynamics(
        cfg, env_config, NO_OVERRIDES, assume_current_defaults=True
    )

    assert dynamics["desired_velocity_weight"] == 0.8
    assert provenance["desired_velocity_weight"] == "config_resolved.yaml"
    for field in ("max_velocity_magnitude", "contact_stiffness", "contact_damping"):
        assert provenance[field] == "assumed-default"
        assert dynamics[field] == getattr(env_config, field)


def test_explicit_contradicting_the_yaml_is_refused(env_config):
    """Two disagreeing sources means one is wrong; certify neither."""
    cfg = {"desired_velocity_weight": 0.8}
    overrides = {**NO_OVERRIDES, "desired_velocity_weight": 0.05}
    with pytest.raises(DynamicsProvenanceError, match="contradicts"):
        resolve_dynamics(cfg, env_config, overrides, assume_current_defaults=True)


def test_explicit_agreeing_with_the_yaml_is_accepted(env_config):
    cfg = {"desired_velocity_weight": 0.8}
    overrides = {**NO_OVERRIDES, "desired_velocity_weight": 0.8}
    dynamics, provenance = resolve_dynamics(
        cfg, env_config, overrides, assume_current_defaults=True
    )
    assert dynamics["desired_velocity_weight"] == 0.8
    assert provenance["desired_velocity_weight"] == "explicit"


def test_stale_clamp_key_defeats_the_waiver(env_config):
    """``max_speed_multiplier`` is silently ignored by today's parser, so its
    presence proves the run's clamp was a different formulation. Even with the
    waiver, the default must not be stamped as trained."""
    cfg = {"desired_velocity_weight": 0.8, "max_speed_multiplier": 4.0}
    with pytest.raises(DynamicsProvenanceError, match="max_speed_multiplier"):
        resolve_dynamics(cfg, env_config, NO_OVERRIDES, assume_current_defaults=True)


def test_stale_clamp_key_is_satisfied_by_an_explicit_value(env_config):
    cfg = {"desired_velocity_weight": 0.8, "max_speed_multiplier": 4.0}
    overrides = {**NO_OVERRIDES, "max_velocity_magnitude": 4.0}
    dynamics, provenance = resolve_dynamics(
        cfg, env_config, overrides, assume_current_defaults=True
    )
    assert dynamics["max_velocity_magnitude"] == 4.0
    assert provenance["max_velocity_magnitude"] == "explicit"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_overrides_are_refused(env_config, bad):
    """A NaN would defeat the deployment reader's mismatch detection."""
    overrides = {**NO_OVERRIDES, "contact_stiffness": bad}
    with pytest.raises(DynamicsProvenanceError, match="finite"):
        resolve_dynamics(
            {"desired_velocity_weight": 0.8},
            env_config,
            overrides,
            assume_current_defaults=True,
        )


def test_resolved_block_is_accepted_by_the_metadata_validator(env_config):
    """Whatever this script resolves must be a valid schema-v2 payload."""
    from crowdrl_core.config_io import validate_dynamics_dict

    dynamics, _ = resolve_dynamics(
        {"desired_velocity_weight": 0.8},
        env_config,
        NO_OVERRIDES,
        assume_current_defaults=True,
    )
    assert validate_dynamics_dict(dynamics) == dynamics
