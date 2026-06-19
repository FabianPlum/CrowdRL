"""CPU-only trajectory render from a checkpoint in a generated Tier 3B scenario.

Loads the policy + obs-normalizer from a training checkpoint, runs ONE episode
through the numpy CrowdEnv (which applies the run's real action config, incl. the
speed-turn coupling), and saves a video. Everything runs on CPU so it never
touches the GPUs -- safe to run alongside live GPU training.

Usage:
  uv run python scripts/render_cpu.py \
      --config configs/baseline.yaml \
      --checkpoint results_baseline/checkpoint_rollout_0500.pt \
      --out results_baseline/viz_r500_tier3B.mp4 \
      --label "baseline"
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
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
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Render episode length in steps. Default: the config's max_steps, so the "
        "render matches the run (slow-but-correct agents are shown finishing, not cut off).",
    )
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(4)  # be polite: leave CPU headroom for the live training loops

    cfg = load_config(args.config)
    env_config = build_env_config(cfg)
    net_config = build_net_config(cfg, env_config)
    # Default the render length to the run's own max_steps so the video shows the
    # full episode (slow agents finishing), not a fixed-length cutoff. The CPU sim
    # cost scales with this, so override with --max-steps for a quicker clip.
    render_max_steps = args.max_steps if args.max_steps is not None else env_config.max_steps

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

    print(
        f"[render_cpu] device=cpu  ckpt={args.checkpoint}  "
        f"agents={tuple(args.agents)}  max_steps={render_max_steps}"
    )
    t_sim0 = time.perf_counter()
    frames = collect_episode_frames(
        env, actor_critic, obs_normalizer, device, max_steps=render_max_steps
    )
    t_sim = time.perf_counter() - t_sim0
    # Tag the title with the checkpoint's actual rollout (parsed from the
    # filename, e.g. checkpoint_rollout_0100 -> r100), not a hardcoded value.
    parts = Path(args.checkpoint).stem.split("_")
    ckpt_tag = f"r{int(parts[-1])}" if parts[-1].isdigit() else Path(args.checkpoint).stem
    frames.title = f"Tier 3B -- {args.label} (ckpt {ckpt_tag})"
    t_enc0 = time.perf_counter()
    out = render_episode_video(frames, args.out, fps=50, trail_length=1000)
    t_enc = time.perf_counter() - t_enc0
    print(f"[render_cpu] WROTE {out}")
    print(
        f"[render_cpu] timing: sim {t_sim:.1f}s + encode {t_enc:.1f}s = "
        f"{t_sim + t_enc:.1f}s total  ({frames.n_frames} frames, {frames.n_agents} agents)"
    )


if __name__ == "__main__":
    main()
