"""Tests for training configuration dataclasses."""

from __future__ import annotations

import json
from pathlib import Path


from crowdrl_env.geometry_generator import GeometryTier

from crowdrl_train.config import (
    DEFAULT_CURRICULUM_PHASES,
    NetworkConfig,
    PPOConfig,
    TrainConfig,
)


class TestNetworkConfig:
    def test_defaults(self):
        cfg = NetworkConfig()
        assert cfg.obs_dim == 79
        assert cfg.action_dim == 4
        assert cfg.actor_hidden_sizes == (256, 256)
        assert cfg.activation == "tanh"
        assert cfg.ortho_init is True

    def test_log_std_init_matches_std_0_5(self):
        """log_std_init should produce initial std ≈ 0.5 (Andrychowicz et al.)."""
        import math

        cfg = NetworkConfig()
        assert abs(math.exp(cfg.log_std_init) - 0.5) < 1e-6


class TestPPOConfig:
    def test_defaults_follow_mappo(self):
        """Verify defaults match MAPPO paper (Yu et al. 2022)."""
        cfg = PPOConfig()
        assert cfg.lr_actor == 5e-4
        assert cfg.gamma == 0.99
        assert cfg.gae_lambda == 0.95
        assert cfg.clip_epsilon == 0.2
        assert cfg.n_epochs == 10
        assert cfg.n_minibatches == 1  # MAPPO: full-batch
        assert cfg.max_grad_norm == 10.0  # MAPPO: permissive
        assert cfg.use_value_clip is False  # Andrychowicz: hurts

    def test_adam_eps_is_1e_5(self):
        """MAPPO uses eps=1e-5, not PyTorch default 1e-8."""
        cfg = PPOConfig()
        assert cfg.adam_eps == 1e-5


class TestCurriculumConfig:
    def test_default_phases(self):
        phases = DEFAULT_CURRICULUM_PHASES
        assert len(phases) == 6
        assert phases[0].name == "easy"
        assert phases[-1].name == "full"
        assert phases[-1].goal_rate_threshold == 0.0  # terminal

    def test_phases_increase_difficulty(self):
        """Final phase should have the highest max agent count."""
        phases = DEFAULT_CURRICULUM_PHASES
        # Terminal phase has the widest agent range
        assert phases[-1].n_agents_range[1] >= max(p.n_agents_range[1] for p in phases[:-1])


class TestTrainConfig:
    def test_json_roundtrip(self):
        """Config should survive serialisation to JSON and back."""
        original = TrainConfig()
        d = original.to_dict()
        json_str = json.dumps(d)
        restored = TrainConfig.from_dict(json.loads(json_str))

        assert restored.network.obs_dim == original.network.obs_dim
        assert restored.ppo.lr_actor == original.ppo.lr_actor
        assert restored.ppo.gamma == original.ppo.gamma
        assert len(restored.curriculum.phases) == len(original.curriculum.phases)

    def test_save_load_json(self, tmp_path: Path):
        """Config should roundtrip through file."""
        original = TrainConfig(seed=123, total_timesteps=5000)
        path = tmp_path / "config.json"
        original.save_json(path)

        loaded = TrainConfig.load_json(path)
        assert loaded.seed == 123
        assert loaded.total_timesteps == 5000

    def test_curriculum_phases_survive_json(self, tmp_path: Path):
        """GeometryTier enums should serialise and restore correctly."""
        original = TrainConfig()
        path = tmp_path / "config.json"
        original.save_json(path)

        loaded = TrainConfig.load_json(path)
        phases = loaded.curriculum.phases
        assert phases[0].geometry_tiers == (GeometryTier.TIER_0,)
        assert phases[2].geometry_tiers == (GeometryTier.TIER_1, GeometryTier.TIER_2)
