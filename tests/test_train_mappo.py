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

from train_mappo import _load_history_and_infer_rollout  # noqa: E402


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
