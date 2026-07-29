"""Unit tests for train_mappo.py helpers.

Covers the history-loading / resume-inference logic. Full end-to-end resume
is exercised manually with real GPU runs; these tests pin down the pure
bookkeeping code that decides *where* to resume from.
"""

from __future__ import annotations

import json
import sys
from collections import namedtuple
from pathlib import Path

import pytest

# Make the repo root importable so `import train_mappo` works.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_mappo import (  # noqa: E402
    _load_history_and_infer_rollout,
    _render_command,
    _resolve_render_interval,
    _scorecard_command,
    build_env_config,
    cfg_dict_from_env_config,
)


class TestBuildEnvConfigSpeed:
    """max_forward_speed / max_backward_speed must propagate from YAML all the
    way to the torch EnvConfig the training loop uses. build_env_config uses an
    explicit per-field .get() allowlist, so a field that is NOT listed silently
    falls back to the dataclass default no matter what the YAML says (the
    velocity-weighting no-op bug). Guard the speed caps against that."""

    def _torch_cfg(self, action_overrides: dict):
        from crowdrl_torch.types import EnvConfig

        crowd = build_env_config({"action": action_overrides})
        return EnvConfig.from_crowd_env_config(crowd, max_agents=8, max_segments=64)

    def test_speed_caps_propagate_from_yaml(self):
        tc = self._torch_cfg({"max_forward_speed": 1.2, "max_backward_speed": 0.3})
        assert tc.max_forward_speed == 1.2
        assert tc.max_backward_speed == 0.3

    def test_speed_caps_default_when_absent(self):
        tc = self._torch_cfg({})
        assert tc.max_forward_speed == 2.0
        assert tc.max_backward_speed == 0.5


class TestCfgDictRoundTrip:
    """``cfg_dict_from_env_config`` is the exact inverse of ``build_env_config``.

    Producers that build their config in code (the training notebooks) rely on
    this to emit a ``config_resolved.yaml`` next to the exported policy that
    re-parses to the identical env config -- the contract that lets eval /
    scorecard / the validation notebooks be config-driven instead of hardcoding
    an ObsConfig. The dump goes through real YAML (dump -> safe_load) so any
    non-serialisable leakage (e.g. numpy scalars) would fail the test too.
    """

    # A source config exercising the non-default knobs that recent runs use:
    # the no-goal-direction ablation, temporal memory, speed/turn coupling, and
    # velocity-weighted collision/proximity rewards.
    SOURCE_CFG = {
        "geometry": {
            "min_side": 10.0,
            "max_side": 18.0,
            "corridor_width": [2.0, 4.0],
            "corridor_length": [8.0, 18.0],
        },
        "observation": {
            "use_navmesh": True,
            "use_goal_direction": False,
            "use_temporal_memory": True,
            "temporal_memory_window": 50,
        },
        "action": {
            "max_heading_change_deg": 4.8,
            "max_torso_change_deg": 4.8,
            "speed_turn_coupling": True,
            "turn_lat_accel": 2.0,
            "turn_pivot_rate_deg": 240.0,
        },
        "reward": {
            "goal_bonus": 20.0,
            "collision_penalty": -2.0,
            "progress_weight": 2.0,
            "use_velocity_weighted_collision": True,
            "collision_penalty_cap": -2.0,
            "use_velocity_weighted_proximity": True,
        },
        "episode": {"stuck_termination_enabled": False},
        "max_steps": 3000,
        "desired_velocity_weight": 0.8,
    }

    def _assert_dataclass_close(self, c0, c1, tol=1e-9):
        import dataclasses

        d0, d1 = dataclasses.asdict(c0), dataclasses.asdict(c1)
        assert d0.keys() == d1.keys()
        for k in d0:
            v0, v1 = d0[k], d1[k]
            if isinstance(v0, float):
                assert abs(v0 - v1) <= tol, f"{type(c0).__name__}.{k}: {v0} != {v1}"
            else:
                assert v0 == v1, f"{type(c0).__name__}.{k}: {v0!r} != {v1!r}"

    def _round_trip(self, source_cfg):
        import yaml

        ec0 = build_env_config(source_cfg)
        # Serialise -> real YAML text -> reload -> rebuild, mirroring how the
        # notebooks write config_resolved.yaml and how eval reads it back.
        dumped = yaml.safe_load(yaml.dump(cfg_dict_from_env_config(ec0)))
        ec1 = build_env_config(dumped)
        return ec0, ec1

    def test_obs_action_reward_geometry_round_trip(self):
        ec0, ec1 = self._round_trip(self.SOURCE_CFG)
        self._assert_dataclass_close(ec0.obs, ec1.obs)
        self._assert_dataclass_close(ec0.action, ec1.action)
        self._assert_dataclass_close(ec0.reward, ec1.reward)
        self._assert_dataclass_close(ec0.geometry, ec1.geometry)

    def test_top_level_fields_round_trip(self):
        ec0, ec1 = self._round_trip(self.SOURCE_CFG)
        assert ec0.max_steps == ec1.max_steps
        assert ec0.dt == ec1.dt
        assert ec0.desired_velocity_weight == ec1.desired_velocity_weight
        assert ec0.stuck_termination_enabled == ec1.stuck_termination_enabled

    def test_obs_dim_preserved(self):
        """The whole point: the re-parsed config rebuilds the same obs width."""
        ec0, ec1 = self._round_trip(self.SOURCE_CFG)
        assert ec0.obs.obs_dim == ec1.obs.obs_dim == 89

    def test_temporal_memory_normalisers_track_top_level(self):
        """temporal_memory_max_steps/dt are derived from top-level max_steps/dt,
        not stored under observation: -- the serialiser must keep them in sync."""
        ec0, ec1 = self._round_trip(self.SOURCE_CFG)
        assert ec1.obs.temporal_memory_max_steps == ec1.max_steps == 3000
        assert ec1.obs.temporal_memory_dt == ec1.dt

    def test_defaults_only_config_round_trips(self):
        """A config that relies entirely on defaults must also round-trip."""
        ec0, ec1 = self._round_trip({})
        self._assert_dataclass_close(ec0.obs, ec1.obs)
        self._assert_dataclass_close(ec0.action, ec1.action)
        self._assert_dataclass_close(ec0.reward, ec1.reward)
        assert ec0.obs.obs_dim == ec1.obs.obs_dim


# A minimal stand-in for CurriculumPhase -- the helper only reads `.name`.
_FakePhase = namedtuple("_FakePhase", ["name"])
PHASES = (
    _FakePhase("easy"),
    _FakePhase("medium"),
    _FakePhase("hard"),
    _FakePhase("full"),
)


def _write_history(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "history.json"
    path.write_text(json.dumps(data))
    return path


class TestLoadHistoryAndInferRollout:
    def test_rollout_from_policy_loss_length(self, tmp_path: Path):
        """last_rollout must equal len(policy_loss) -- appended once per rollout."""
        path = _write_history(
            tmp_path,
            {
                "policy_loss": [0.1, 0.2, 0.3, 0.4, 0.5],
                "goal_rate": [0.1] * 50,
                "phase_idx": [0] * 50,
                "mean_reward": [0.0] * 50,
                "episode_length": [100] * 50,
                "n_agents": [10] * 50,
                "geometry_tier": ["unknown"] * 50,
            },
        )
        _history, last_rollout, total_episodes, _phase_transitions = (
            _load_history_and_infer_rollout(path, PHASES)
        )
        assert last_rollout == 5
        assert total_episodes == 50

    def test_empty_history(self, tmp_path: Path):
        """An empty history (no rollouts completed) resumes at rollout 1."""
        path = _write_history(
            tmp_path,
            {
                "policy_loss": [],
                "goal_rate": [],
                "phase_idx": [],
            },
        )
        _history, last_rollout, total_episodes, transitions = _load_history_and_infer_rollout(
            path, PHASES
        )
        assert last_rollout == 0
        assert total_episodes == 0
        assert transitions == []

    def test_phase_transitions_reconstructed(self, tmp_path: Path):
        """Phase transitions are inferred from phase_idx deltas."""
        # 3 episodes in easy (0), 2 in medium (1), 4 in hard (2)
        phase_idx = [0, 0, 0, 1, 1, 2, 2, 2, 2]
        path = _write_history(
            tmp_path,
            {
                "policy_loss": [0.0] * 3,
                "goal_rate": [0.0] * len(phase_idx),
                "phase_idx": phase_idx,
                "mean_reward": [0.0] * len(phase_idx),
                "episode_length": [100] * len(phase_idx),
                "n_agents": [10] * len(phase_idx),
                "geometry_tier": ["unknown"] * len(phase_idx),
            },
        )
        _history, _last_rollout, _total_episodes, transitions = _load_history_and_infer_rollout(
            path, PHASES
        )
        # Transition from easy->medium happens when processing episode index 3
        # (0-indexed), recorded as episode number 4. Transition medium->hard
        # happens at episode index 5, recorded as episode number 6.
        assert transitions == [(4, "medium"), (6, "hard")]

    def test_no_phase_transitions_when_single_phase(self, tmp_path: Path):
        """A run that never advances should produce an empty transitions list."""
        path = _write_history(
            tmp_path,
            {
                "policy_loss": [0.0] * 5,
                "goal_rate": [0.0] * 20,
                "phase_idx": [0] * 20,
                "mean_reward": [0.0] * 20,
                "episode_length": [100] * 20,
                "n_agents": [10] * 20,
                "geometry_tier": ["unknown"] * 20,
            },
        )
        _history, _last_rollout, _total_episodes, transitions = _load_history_and_infer_rollout(
            path, PHASES
        )
        assert transitions == []

    def test_history_dict_preserves_all_keys(self, tmp_path: Path):
        """All history keys must round-trip so plots can use them later."""
        saved = {
            "policy_loss": [0.1, 0.2],
            "value_loss": [0.3, 0.4],
            "entropy": [1.1, 1.0],
            "approx_kl": [0.01, 0.02],
            "goal_rate": [0.5, 0.6, 0.7, 0.8],
            "mean_reward": [1.0, 2.0, 3.0, 4.0],
            "episode_length": [100, 110, 120, 130],
            "n_agents": [10, 11, 12, 13],
            "phase_idx": [0, 0, 1, 1],
            "geometry_tier": ["TIER_0", "TIER_0", "TIER_1", "TIER_1"],
        }
        path = _write_history(tmp_path, saved)
        history, last_rollout, total_episodes, _ = _load_history_and_infer_rollout(path, PHASES)
        assert last_rollout == 2
        assert total_episodes == 4
        for k, v in saved.items():
            assert history[k] == v, f"key {k} mismatch"

    def test_unknown_phase_idx_uses_fallback_name(self, tmp_path: Path):
        """A phase_idx beyond the known phases tuple falls back gracefully."""
        path = _write_history(
            tmp_path,
            {
                "policy_loss": [0.0],
                "goal_rate": [0.0] * 3,
                "phase_idx": [0, 1, 99],  # 99 is out of range
                "mean_reward": [0.0] * 3,
                "episode_length": [100] * 3,
                "n_agents": [10] * 3,
                "geometry_tier": ["unknown"] * 3,
            },
        )
        _history, _last_rollout, _total_episodes, transitions = _load_history_and_infer_rollout(
            path, PHASES
        )
        # Two transitions: 0->1 (medium) and 1->99 (fallback name)
        assert len(transitions) == 2
        assert transitions[0] == (2, "medium")
        assert transitions[1][0] == 3
        assert transitions[1][1].startswith("phase_")


class TestCliArgumentValidation:
    """--start_from_zero requires --resume_training."""

    def test_start_from_zero_without_resume_errors(self):
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "train_mappo.py"),
                "--config",
                "nonexistent.yaml",
                "--start_from_zero",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "--start_from_zero requires --resume_training" in result.stderr

    def test_resume_training_without_checkpoint_errors(self, tmp_path: Path, monkeypatch):
        """When --resume_training is set but no checkpoint exists, fail fast."""
        # Build a minimal valid config file so CLI parsing passes
        import yaml

        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "seed": 42,
                    "n_envs": 1,
                    "max_agents": 5,
                    "steps_per_collect": 10,
                    "n_rollouts": 1,
                    "curriculum": {
                        "phases": [
                            {
                                "name": "easy",
                                "tiers": ["TIER_0"],
                                "agents": [3, 5],
                                "threshold": 0.0,
                            }
                        ],
                    },
                    "max_steps": 10,
                }
            )
        )

        # Run train_worker directly (not via subprocess) to avoid launching
        # torchrun. We expect FileNotFoundError before any environment is built.
        from train_mappo import load_config, train_worker

        monkeypatch.chdir(tmp_path)
        cfg = load_config(cfg_path)
        results_dir = tmp_path / "results_cfg"
        # results_dir does not exist, so no checkpoint -> FileNotFoundError
        with pytest.raises(FileNotFoundError, match="no checkpoint found"):
            train_worker(cfg, results_dir, resume_training=True)


class TestResolveRenderInterval:
    """Effective training-time render interval (must align with checkpoints)."""

    def test_disabled_by_default(self):
        assert _resolve_render_interval(False, 0, 100) == 0

    def test_disabled_even_if_interval_set(self):
        assert _resolve_render_interval(False, 100, 100) == 0

    def test_defaults_to_checkpoint_interval(self):
        assert _resolve_render_interval(True, 0, 100) == 100

    def test_custom_multiple_kept(self):
        assert _resolve_render_interval(True, 500, 100) == 500

    def test_non_multiple_snaps_to_checkpoint(self):
        assert _resolve_render_interval(True, 150, 100) == 100

    def test_no_checkpoints_disables(self):
        # Renders load the on-disk checkpoint; with checkpointing off there is
        # nothing to render from.
        assert _resolve_render_interval(True, 100, 0) == 0


class TestRenderCommand:
    """The CPU-render subprocess argv is well-formed."""

    def test_command_shape(self, tmp_path: Path):
        cmd = _render_command(
            tmp_path / "config_resolved.yaml",
            tmp_path / "checkpoint_rollout_0200.pt",
            tmp_path / "viz_r0200_tier3B.mp4",
            "exp_label",
        )
        assert cmd[0] == sys.executable
        assert cmd[1].replace("\\", "/").endswith("scripts/render_cpu.py")
        # Flags present and paired with the right values.
        for flag, val in (
            ("--config", str(tmp_path / "config_resolved.yaml")),
            ("--checkpoint", str(tmp_path / "checkpoint_rollout_0200.pt")),
            ("--out", str(tmp_path / "viz_r0200_tier3B.mp4")),
            ("--label", "exp_label"),
        ):
            assert flag in cmd
            assert cmd[cmd.index(flag) + 1] == val


class TestScorecardCommand:
    """The CPU-scorecard subprocess argv is well-formed."""

    def test_command_shape(self, tmp_path: Path):
        cmd = _scorecard_command(
            tmp_path / "config_resolved.yaml",
            tmp_path / "checkpoint_rollout_0200.pt",
            tmp_path / "scorecard_r0200.json",
            1500,
        )
        assert cmd[0] == sys.executable
        assert cmd[1].replace("\\", "/").endswith("scripts/eval_scorecard.py")
        for flag, val in (
            ("--config", str(tmp_path / "config_resolved.yaml")),
            ("--checkpoint", str(tmp_path / "checkpoint_rollout_0200.pt")),
            ("--json", str(tmp_path / "scorecard_r0200.json")),
            ("--max-steps", "1500"),
        ):
            assert flag in cmd
            assert cmd[cmd.index(flag) + 1] == val
