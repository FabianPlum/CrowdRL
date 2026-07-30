"""Tests for the lossless config <-> dict serialization boundary.

These functions are the payload format for the ONNX ``metadata_props``
embedding (issue #7): both the exporter and the deployment reader go through
them, so exactness here is what makes the embedded config trustworthy.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from crowdrl_core.action import ActionConfig
from crowdrl_core.config_io import (
    action_config_from_dict,
    action_config_to_dict,
    obs_config_from_dict,
    obs_config_to_dict,
)
from crowdrl_core.observation import ObsConfig
from crowdrl_core.sensing import RaycastConfig

# A deliberately non-default config exercising every schema axis the YAML
# contract cannot express: raycast geometry, 2-channel rays, and social K.
NON_DEFAULT_OBS = ObsConfig(
    k_neighbours=6,
    raycast=RaycastConfig(n_rays=32, fov_deg=170.0, max_range=8.0, two_channel=True),
    use_navmesh=True,
    use_goal_direction=False,
    use_jupedsim_style_routing=True,
    use_temporal_memory=True,
    temporal_memory_window=25,
    temporal_memory_max_steps=3000,
    temporal_memory_dt=0.02,
    use_neighbor_memory=True,
    use_neighbor_vel_history=True,
)

NON_DEFAULT_ACTION = ActionConfig(
    max_forward_speed=1.8,
    max_backward_speed=0.4,
    max_heading_change=float(np.radians(4.8)),
    max_torso_change=float(np.radians(4.8)),
    speed_turn_coupling=True,
    turn_pivot_rate=float(np.radians(240.0)),
)


class TestObsConfigRoundTrip:
    def test_round_trip_is_exact(self):
        """Frozen-dataclass equality after to -> from, nested raycast included."""
        assert obs_config_from_dict(obs_config_to_dict(NON_DEFAULT_OBS)) == NON_DEFAULT_OBS

    def test_round_trip_through_json_text(self):
        """The dict must survive real JSON serialization -- that is how it is
        stored in ONNX metadata_props."""
        text = json.dumps(obs_config_to_dict(NON_DEFAULT_OBS))
        assert obs_config_from_dict(json.loads(text)) == NON_DEFAULT_OBS

    def test_obs_dim_preserved(self):
        """The consequence that matters: the rebuilt config derives the same
        observation width."""
        rebuilt = obs_config_from_dict(json.loads(json.dumps(obs_config_to_dict(NON_DEFAULT_OBS))))
        assert rebuilt.obs_dim == NON_DEFAULT_OBS.obs_dim

    def test_defaults_only_round_trips(self):
        assert obs_config_from_dict(obs_config_to_dict(ObsConfig())) == ObsConfig()

    def test_missing_keys_fill_from_defaults(self):
        """Forward compatibility: an artefact exported before a field existed
        reconstructs with that field at its (behaviour-preserving) default."""
        data = obs_config_to_dict(NON_DEFAULT_OBS)
        del data["use_neighbor_vel_history"]
        rebuilt = obs_config_from_dict(data)
        assert rebuilt.use_neighbor_vel_history is False
        assert rebuilt.raycast == NON_DEFAULT_OBS.raycast

    def test_legacy_artefact_reconstructs_funnel_routing(self):
        """Artefacts exported before use_jupedsim_style_routing existed (every
        funnel-trained run, r0400 included) must reconstruct with the flag
        False -- the behaviour they were trained under."""
        data = obs_config_to_dict(NON_DEFAULT_OBS)
        del data["use_jupedsim_style_routing"]
        rebuilt = obs_config_from_dict(data)
        assert rebuilt.use_jupedsim_style_routing is False

    def test_unknown_top_level_key_raises(self):
        """Backward incompatibility must be loud: a newer exporter's field this
        core cannot reproduce."""
        data = obs_config_to_dict(NON_DEFAULT_OBS)
        data["use_quantum_sensing"] = True
        with pytest.raises(ValueError, match="use_quantum_sensing"):
            obs_config_from_dict(data)

    def test_unknown_raycast_key_raises(self):
        data = obs_config_to_dict(NON_DEFAULT_OBS)
        data["raycast"]["lidar_noise"] = 0.1
        with pytest.raises(ValueError, match="lidar_noise"):
            obs_config_from_dict(data)


class TestActionConfigRoundTrip:
    def test_round_trip_is_exact(self):
        assert (
            action_config_from_dict(action_config_to_dict(NON_DEFAULT_ACTION))
            == NON_DEFAULT_ACTION
        )

    def test_round_trip_through_json_text(self):
        """Raw radians survive JSON bit-exactly (floats are IEEE round-trip
        safe in json), so no deg<->rad translation error can creep in."""
        text = json.dumps(action_config_to_dict(NON_DEFAULT_ACTION))
        rebuilt = action_config_from_dict(json.loads(text))
        assert rebuilt == NON_DEFAULT_ACTION
        assert rebuilt.max_heading_change == NON_DEFAULT_ACTION.max_heading_change

    def test_head_limit_and_dt_travel(self):
        """head_limit and dt are part of the motion envelope; they must be in
        the payload even though the YAML schema never carried them."""
        data = action_config_to_dict(NON_DEFAULT_ACTION)
        assert "head_limit" in data
        assert "dt" in data

    def test_unknown_key_raises(self):
        data = action_config_to_dict(NON_DEFAULT_ACTION)
        data["max_jetpack_thrust"] = 9000.0
        with pytest.raises(ValueError, match="max_jetpack_thrust"):
            action_config_from_dict(data)


class TestDynamicsBlock:
    def test_valid_block_passes_and_coerces_floats(self):
        from crowdrl_core.config_io import validate_dynamics_dict

        out = validate_dynamics_dict({"desired_velocity_weight": 0.8, "contact_damping": 500})
        assert out == {"desired_velocity_weight": 0.8, "contact_damping": 500.0}
        assert isinstance(out["contact_damping"], float)

    def test_missing_keys_are_allowed(self):
        from crowdrl_core.config_io import validate_dynamics_dict

        assert validate_dynamics_dict({}) == {}

    def test_unknown_key_raises(self):
        from crowdrl_core.config_io import validate_dynamics_dict

        with pytest.raises(ValueError, match="warp_factor"):
            validate_dynamics_dict({"warp_factor": 9.0})

    @pytest.mark.parametrize("payload", [[], 3, "x", None, ["desired_velocity_weight"]])
    def test_non_object_payload_raises_cleanly(self, payload):
        """Valid JSON that is not an object used to escape as an AttributeError
        from .items(), bypassing the caller's 'unreadable metadata' handling."""
        from crowdrl_core.config_io import validate_dynamics_dict

        with pytest.raises(ValueError, match="must be a JSON object"):
            validate_dynamics_dict(payload)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_values_raise(self, bad):
        """A NaN reaching the physics propagates silently through every
        position, and defeats any comparison-based mismatch detection."""
        from crowdrl_core.config_io import validate_dynamics_dict

        with pytest.raises(ValueError, match="finite"):
            validate_dynamics_dict({"contact_stiffness": bad})

    def test_negative_values_raise(self):
        from crowdrl_core.config_io import validate_dynamics_dict

        with pytest.raises(ValueError, match="non-negative"):
            validate_dynamics_dict({"contact_damping": -1.0})

    @pytest.mark.parametrize("bad", [True, False])
    def test_bools_raise(self, bad):
        """bool is an int subclass, so float(True) == 1.0 sailed through."""
        from crowdrl_core.config_io import validate_dynamics_dict

        with pytest.raises(ValueError, match="must be a number"):
            validate_dynamics_dict({"desired_velocity_weight": bad})

    def test_zero_is_accepted(self):
        """Zero contact stiffness (physics disabled) is legitimate."""
        from crowdrl_core.config_io import validate_dynamics_dict

        assert validate_dynamics_dict({"contact_stiffness": 0.0}) == {"contact_stiffness": 0.0}
