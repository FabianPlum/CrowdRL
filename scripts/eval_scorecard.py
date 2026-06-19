"""Fixed-scenario behavioural scorecard for a checkpoint (CPU only).

Loads a policy + obs-normalizer from a training checkpoint and runs it through a
DETERMINISTIC scenario suite, printing a decomposed scorecard -- goal / collision
/ wall / freeze / stuck / speed / path-efficiency -- per scenario and overall.
Everything runs on CPU, so it is safe alongside live GPU training.

The two failure modes are the two ends of one throughput<->safety Pareto front:
bulldozing shows up as a high ``coll`` rate, gridlock as a high ``freeze`` /
``stuck`` rate. Compare two checkpoints/configs by whether goal rate rises
WITHOUT the other axes worsening.

Usage:
  uv run python scripts/eval_scorecard.py \
      --config configs/baseline.yaml \
      --checkpoint results_baseline/checkpoint_rollout_0300.pt \
      [--max-steps 1500] [--json results_.../scorecard_r0300.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add the repo root so the top-level ``train_mappo`` module (config/env/net
# builders) resolves, matching scripts/render_cpu.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from crowdrl_torch.normalizer import TorchRunningNormalizer
from crowdrl_train.networks import ActorCritic
from crowdrl_train.scorecard import format_scorecard, run_scorecard_policy
from train_mappo import build_env_config, build_net_config, load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Per-episode cap. Default: the config's max_steps. Lower it for a quicker pass.",
    )
    ap.add_argument("--freeze-speed", type=float, default=0.1)
    ap.add_argument("--json", default=None, help="Optional path to dump the scorecard as JSON.")
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(4)  # be polite: leave CPU for any live training loops

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

    scorecard = run_scorecard_policy(
        env_config,
        actor_critic,
        obs_normalizer,
        device=device,
        max_steps=args.max_steps,
        freeze_speed=args.freeze_speed,
    )

    print(f"\n[scorecard] {args.checkpoint}")
    print(format_scorecard(scorecard))
    if args.json:
        Path(args.json).write_text(json.dumps(scorecard, indent=2, default=str))
        print(f"\n[scorecard] wrote {args.json}")


if __name__ == "__main__":
    main()
