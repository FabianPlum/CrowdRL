"""CPU-only trajectory render from a checkpoint in a generated Tier 3B scenario.

Loads the policy + obs-normalizer from a training checkpoint, runs ONE episode
through the numpy CrowdEnv (which applies the run's real action config, incl. the
speed-turn coupling), and saves a video. Everything runs on CPU so it never
touches the GPUs -- safe to run alongside live GPU training.

Usage:
  uv run python scripts/render_cpu.py \
      --config configs/exp_coupling_smoothoff.yaml \
      --checkpoint results_exp_coupling_smoothoff/checkpoint_rollout_0500.pt \
      --out results_exp_coupling_smoothoff/viz_r500_tier3B.mp4 \
      --label "smoothness OFF"
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

# scripts/ (this file's dir) is on sys.path, but the repo-root train_mappo
# module is not -- add the repo root so `import train_mappo` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from crowdrl_env.crowd_env import CrowdEnv
from crowdrl_env.geometry_generator import GeometryTier
from crowdrl_env.spawner import SpawnConfig
from crowdrl_env.visualiser import collect_episode_frames, render_episode_video
from crowdrl_train.networks import ActorCritic
from crowdrl_torch.normalizer import TorchRunningNormalizer
from train_mappo import build_env_config, build_net_config, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--agents", type=int, nargs=2, default=[20, 40])
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(4)  # be polite: leave CPU headroom for the live training loops

    cfg = load_config(args.config)
    env_config = build_env_config(cfg)
    net_config = build_net_config(cfg, env_config)

    actor_critic = ActorCritic(net_config).to(device)
    obs_normalizer = TorchRunningNormalizer(shape=(env_config.obs.obs_dim,), device=device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor_critic.load_state_dict(ckpt["actor_critic"])
    if ckpt.get("obs_normalizer") is not None:
        obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
    actor_critic.eval()

    # Tier 3B (composed rooms) scenario, keeping the run's physics/action/obs/reward.
    render_config = dataclasses.replace(
        env_config,
        geometry_tiers=[GeometryTier.TIER_3B],
        spawn=SpawnConfig(n_agents_range=tuple(args.agents)),
    )
    env = CrowdEnv(config=render_config, seed=args.seed)

    print(f"[render_cpu] device=cpu  ckpt={args.checkpoint}  agents={tuple(args.agents)}")
    frames = collect_episode_frames(
        env, actor_critic, obs_normalizer, device, max_steps=args.max_steps
    )
    frames.title = f"Tier 3B -- {args.label} (ckpt r500)"
    out = render_episode_video(frames, args.out, fps=50, trail_length=1000)
    print(f"[render_cpu] WROTE {out}")


if __name__ == "__main__":
    main()
