#!/usr/bin/env python3
"""MAPPO training script with automatic multi-GPU support.

Reads a YAML config and runs the full pipeline: training, evaluation plots,
ONNX export. When multiple GPUs are detected the script transparently
re-launches itself via ``torchrun`` (DD-PPO pattern).

Usage
-----
    uv run python train_mappo.py --config configs/full_training.yaml

All output is written to ``results_<config-stem>/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib import cm

# -- CrowdRL imports ---------------------------------------------------------
from crowdrl_core.action import ActionConfig
from crowdrl_core.observation import ObsConfig
from crowdrl_env.crowd_env import CrowdEnv, CrowdEnvConfig
from crowdrl_env.geometry_generator import GeometryConfig, GeometryTier
from crowdrl_env.reward import RewardConfig
from crowdrl_env.spawner import SpawnConfig
from crowdrl_train.config import CurriculumConfig, CurriculumPhase, NetworkConfig, PPOConfig
from crowdrl_train.curriculum import CurriculumManager, EpisodeStats
from crowdrl_train.export import export_onnx
from crowdrl_train.mappo import MAPPOUpdater
from crowdrl_train.networks import ActorCritic
from crowdrl_train.normalizer import RewardNormalizer
from crowdrl_train.train import load_checkpoint, save_checkpoint
from crowdrl_torch import (
    BatchedTorchEnv,
    TorchRolloutCollector,
    TorchRunningNormalizer,
    make_episode_factory,
)
from crowdrl_torch.distributed import (
    broadcast_curriculum_state,
    cleanup_distributed,
    distributed_seed,
    gather_episode_stats,
    init_distributed,
    is_main_rank,
    seed_everything,
    sync_reward_normalizer,
)
from crowdrl_torch.types import EnvConfig

# Non-interactive matplotlib backend -- no display needed for headless training
matplotlib.use("Agg")

# Suppress harmless torch.compile warning: "skipping cudagraphs due to cpu device".
# Our EnvConfig is a NamedTuple with plain Python scalars (dt, max_speed, ...) that
# torch.compile sees as CPU inputs, preventing CUDA graph capture for that region.
# The compiled kernels still run on GPU -- only the graph-capture optimization is
# skipped, which has negligible impact on throughput.
logging.getLogger("torch._inductor.cudagraph_utils").setLevel(logging.ERROR)

# Shorten inductor kernel names so triton cache filenames don't exceed the 255-byte
# filesystem limit. The default ``descriptive_names='original_aten'`` concatenates
# every fused aten op name into the kernel filename; once we fuse the temporal-memory
# ring-buffer ops (scatter + ~20 elementwise ops) alongside the existing step kernel
# the filename grew past 255 bytes and inductor fell back to eager with an ENAMETOOLONG
# OSError. Setting to ``False`` uses short numeric indices instead and adds no
# debugging penalty for our workflow.
torch._inductor.config.triton.descriptive_names = False


# ============================================================================
# YAML config -> dataclass configs
# ============================================================================


def load_config(path: str | Path) -> dict:
    """Load and return the raw YAML dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_curriculum_config(cfg: dict) -> CurriculumConfig:
    cur = cfg["curriculum"]
    phases = []
    for p in cur["phases"]:
        tiers = tuple(GeometryTier[t] for t in p["tiers"])
        tw = p.get("tier_weights")
        phases.append(
            CurriculumPhase(
                name=p["name"],
                geometry_tiers=tiers,
                n_agents_range=tuple(p["agents"]),
                goal_rate_threshold=p["threshold"],
                tier_weights=tuple(tw) if tw else None,
            )
        )
    return CurriculumConfig(
        phases=tuple(phases),
        metric_window=cur.get("metric_window", 100),
        min_episodes_per_phase=cur.get("min_episodes_per_phase", 200),
    )


def build_env_config(cfg: dict) -> CrowdEnvConfig:
    geo = cfg.get("geometry", {})
    obs = cfg.get("observation", {})
    act = cfg.get("action", {})
    rew = cfg.get("reward", {})
    ep = cfg.get("episode", {})

    return CrowdEnvConfig(
        geometry=GeometryConfig(
            min_side=geo.get("min_side", 8.0),
            max_side=geo.get("max_side", 15.0),
            corridor_width_range=tuple(geo.get("corridor_width", (2.0, 4.0))),
            corridor_length_range=tuple(geo.get("corridor_length", (8.0, 18.0))),
        ),
        obs=ObsConfig(
            use_navmesh=obs.get("use_navmesh", True),
            use_temporal_memory=obs.get("use_temporal_memory", False),
            temporal_memory_window=obs.get("temporal_memory_window", 50),
            temporal_memory_max_steps=cfg.get("max_steps", 2000),
            temporal_memory_dt=cfg.get("dt", 0.01),
        ),
        action=ActionConfig(
            max_heading_change=np.radians(act.get("max_heading_change_deg", 15.0)),
            max_torso_change=np.radians(act.get("max_torso_change_deg", 15.0)),
        ),
        reward=RewardConfig(
            goal_bonus=rew.get("goal_bonus", 20.0),
            collision_penalty=rew.get("collision_penalty", -5.0),
            timeout_penalty=rew.get("timeout_penalty", -10.0),
            existence_penalty=rew.get("existence_penalty", -0.01),
            progress_weight=rew.get("progress_weight", 1.0),
            inverse_distance_weight=rew.get("inverse_distance_weight", 0.0),
            speed_deviation_weight=rew.get("speed_deviation_weight", 0.0),
            use_smoothness=rew.get("use_smoothness", True),
            wall_proximity_penalty=rew.get("wall_proximity_penalty", -0.1),
            wall_proximity_threshold=rew.get("wall_proximity_threshold", 1.5),
            agent_proximity_penalty_near=rew.get("agent_proximity_penalty_near", -0.005),
            agent_proximity_penalty_far=rew.get("agent_proximity_penalty_far", -0.0001),
            personal_space_radius=rew.get("personal_space_radius", 1.0),
            action_rate_weight=rew.get("action_rate_weight", -0.01),
        ),
        max_steps=cfg.get("max_steps", 2000),
        stuck_termination_enabled=ep.get("stuck_termination_enabled", False),
        stuck_window_steps=ep.get("stuck_window_steps", 300),
        stuck_progress_threshold=ep.get("stuck_progress_threshold", 0.2),
    )


def build_net_config(cfg: dict, env_config: CrowdEnvConfig) -> NetworkConfig:
    net = cfg.get("network", {})
    return NetworkConfig(
        obs_dim=env_config.obs.obs_dim,
        action_dim=env_config.action.action_dim,
        actor_hidden_sizes=tuple(net.get("actor_hidden", [256, 256])),
        critic_hidden_sizes=tuple(net.get("critic_hidden", [256, 256])),
    )


def build_ppo_config(cfg: dict) -> PPOConfig:
    p = cfg.get("ppo", {})
    return PPOConfig(
        lr_actor=p.get("lr_actor", 5e-4),
        lr_critic=p.get("lr_critic", 5e-4),
        n_epochs=p.get("n_epochs", 10),
        clip_epsilon=p.get("clip_epsilon", 0.2),
        gamma=p.get("gamma", 0.99),
        gae_lambda=p.get("gae_lambda", 0.95),
        target_kl=p.get("target_kl", 0.02),
        lr_schedule=p.get("lr_schedule", "cosine"),
    )


# ============================================================================
# Post-training: plots, evaluation, export  (rank 0 only)
# ============================================================================


def save_training_plots(
    history: dict,
    phase_transitions: list,
    phases: tuple,
    results_dir: Path,
) -> None:
    """Save the 6-panel training curves figure."""

    def smooth(values, window=100):
        if len(values) < window:
            return np.array(values)
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="valid")

    phase_colors = ["#e8f5e9", "#fff3e0", "#fce4ec", "#ede7f6", "#e0f7fa", "#e3f2fd"]
    phase_names_unique = [p.name for p in phases]

    def add_phase_bg(ax):
        prev_ep = 0
        for ep_idx, name in phase_transitions:
            pidx = phase_names_unique.index(name) - 1
            if 0 <= pidx < len(phase_colors):
                ax.axvspan(prev_ep, ep_idx, alpha=0.3, color=phase_colors[pidx])
            prev_ep = ep_idx
        if phase_transitions:
            last_pidx = phase_names_unique.index(phase_transitions[-1][1])
            ax.axvspan(
                prev_ep,
                len(history["goal_rate"]),
                alpha=0.3,
                color=phase_colors[min(last_pidx, len(phase_colors) - 1)],
            )
        else:
            ax.axvspan(0, max(len(history["goal_rate"]), 1), alpha=0.3, color=phase_colors[0])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    add_phase_bg(ax)
    ax.plot(smooth(history["goal_rate"]), color="tab:green", linewidth=0.8)
    ax.set_ylabel("Goal Rate")
    ax.set_xlabel("Episode")
    ax.set_title("Goal Rate (rolling 100)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    add_phase_bg(ax)
    ax.plot(smooth(history["mean_reward"]), color="tab:blue", linewidth=0.8)
    ax.set_ylabel("Mean Reward")
    ax.set_xlabel("Episode")
    ax.set_title("Mean Reward (rolling 100)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    add_phase_bg(ax)
    ax.plot(smooth(history["episode_length"]), color="tab:orange", linewidth=0.8)
    ax.set_ylabel("Episode Length")
    ax.set_xlabel("Episode")
    ax.set_title("Episode Length (rolling 100)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if history["policy_loss"]:
        ax.plot(smooth(history["policy_loss"]), color="tab:red", linewidth=0.8, label="Policy")
        ax2 = ax.twinx()
        ax2.plot(smooth(history["value_loss"]), color="tab:purple", linewidth=0.8, label="Value")
        ax2.set_ylabel("Value Loss", color="tab:purple")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
    ax.set_ylabel("Policy Loss", color="tab:red")
    ax.set_xlabel("Rollout")
    ax.set_title("Losses (rolling 100)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if history["entropy"]:
        ax.plot(smooth(history["entropy"]), color="tab:cyan", linewidth=0.8)
    ax.set_ylabel("Entropy")
    ax.set_xlabel("Rollout")
    ax.set_title("Policy Entropy (rolling 100)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    if history.get("geometry_tier"):
        tier_arr = np.array(history["geometry_tier"])
        tier_colors = {
            "TIER_0": "tab:green",
            "TIER_1": "tab:blue",
            "TIER_2": "tab:orange",
            "TIER_3A": "tab:red",
            "TIER_3B": "tab:purple",
        }
        for tier_name in sorted(set(tier_arr)):
            mask = tier_arr == tier_name
            indices = np.where(mask)[0]
            rates = np.array(history["goal_rate"])[mask]
            if len(rates) >= 20:
                smoothed = smooth(rates, window=min(50, len(rates)))
                offset = len(rates) - len(smoothed)
                ax.plot(
                    indices[offset:],
                    smoothed,
                    color=tier_colors.get(str(tier_name), "gray"),
                    linewidth=0.8,
                    label=str(tier_name),
                    alpha=0.8,
                )
        ax.legend(fontsize=7, loc="lower left")
    add_phase_bg(ax)
    ax.set_ylabel("Goal Rate")
    ax.set_xlabel("Episode")
    ax.set_title("Goal Rate by Tier (rolling 50)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "MAPPO Training -- Vectorized Envs with Curriculum", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(results_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves -> {results_dir / 'training_curves.png'}")


def save_phase_summary(
    history: dict,
    phases: tuple,
    results_dir: Path,
) -> None:
    """Print and save per-phase / per-tier statistics."""
    phase_idx_arr = np.array(history["phase_idx"])
    goal_rate_arr = np.array(history["goal_rate"])
    reward_arr = np.array(history["mean_reward"])
    length_arr = np.array(history["episode_length"])
    agents_arr = np.array(history["n_agents"])

    lines = []
    header = (
        f"{'Phase':<10} {'Episodes':>8} {'GoalRate':>10} {'Reward':>10} {'EpLen':>8} {'Agents':>8}"
    )
    lines.append(header)
    lines.append("-" * 60)
    for pidx, phase in enumerate(phases):
        mask = phase_idx_arr == pidx
        if not mask.any():
            continue
        n_eps = mask.sum()
        lines.append(
            f"{phase.name:<10} {n_eps:>8} {goal_rate_arr[mask].mean():>10.3f} "
            f"{reward_arr[mask].mean():>10.2f} {length_arr[mask].mean():>8.1f} "
            f"{agents_arr[mask].mean():>8.1f}"
        )

    lines.append("\nFinal 500 episodes:")
    lines.append(f"  Goal rate: {np.mean(history['goal_rate'][-500:]):.3f}")
    lines.append(f"  Mean reward: {np.mean(history['mean_reward'][-500:]):.2f}")

    if history.get("geometry_tier"):
        tier_arr = np.array(history["geometry_tier"])
        lines.append(
            f"\n{'Tier':<12} {'Episodes':>8} {'GoalRate':>10} {'Reward':>10} {'EpLen':>8}"
        )
        lines.append("-" * 52)
        for tier_name in sorted(set(tier_arr)):
            mask = tier_arr == tier_name
            if not mask.any():
                continue
            lines.append(
                f"{str(tier_name):<12} {mask.sum():>8} "
                f"{goal_rate_arr[mask].mean():>10.3f} "
                f"{reward_arr[mask].mean():>10.2f} "
                f"{length_arr[mask].mean():>8.1f}"
            )

    summary = "\n".join(lines)
    print(summary)
    (results_dir / "phase_summary.txt").write_text(summary)


def _collect_eval_episodes_batched(
    batched_env: BatchedTorchEnv,
    actor_critic: ActorCritic,
    obs_normalizer,
    obs_dim: int,
    action_dim: int,
    max_steps: int,
    device: torch.device,
) -> dict[str, list]:
    """Run exactly one episode in each env of a BatchedTorchEnv (in parallel).

    Each env is allowed to complete its first episode and is then frozen
    (subsequent auto-resets are ignored). Returns aggregated per-episode stats.
    """
    n_envs = batched_env.n_envs
    N = batched_env.config.max_agents

    states, obs = batched_env.reset_all()
    batched_env.states = states

    # Snapshot starting agent counts: BatchedTorchEnv may auto-reset envs
    # whose episode just ended, replacing n_agents with the next episode's value.
    n_agents_initial = states.n_agents.cpu().numpy().astype(np.int32).copy()

    ep_rewards = np.zeros((n_envs, N), dtype=np.float64)
    ep_terminated = np.zeros((n_envs, N), dtype=np.bool_)
    ep_lengths = np.zeros(n_envs, dtype=np.int32)
    done_flags = np.zeros(n_envs, dtype=np.bool_)
    captured: list[dict | None] = [None] * n_envs

    # Hard cap; an env should always finish within max_steps + a small margin.
    for _ in range(max_steps + 50):
        if done_flags.all():
            break

        if obs_normalizer is not None:
            obs_norm = obs_normalizer.normalize(obs)
        else:
            obs_norm = obs

        with torch.no_grad():
            actions, _, _, _, _ = actor_critic.get_action_and_value(
                obs_norm.reshape(-1, obs_dim),
            )
        actions_gpu = actions.reshape(n_envs, N, action_dim)

        states, obs, rewards, terminated, _truncated = batched_env.step(actions_gpu)

        rewards_np = rewards.cpu().numpy()
        terminated_np = terminated.cpu().numpy()
        active_np = states.active_mask.cpu().numpy()
        episode_over_np = batched_env.episode_over.cpu().numpy()

        for i in range(n_envs):
            if done_flags[i]:
                continue
            ep_rewards[i] += rewards_np[i] * active_np[i]
            ep_terminated[i] |= terminated_np[i]
            if active_np[i].any():
                ep_lengths[i] += 1

            if episode_over_np[i]:
                n_ag = int(n_agents_initial[i])
                if n_ag == 0:
                    done_flags[i] = True
                    continue
                n_reached = int(ep_terminated[i, :n_ag].sum())
                total_rew = float(ep_rewards[i, :n_ag].sum())
                captured[i] = {
                    "goal_rate": n_reached / n_ag,
                    "mean_reward": total_rew / n_ag,
                    "episode_length": int(ep_lengths[i]),
                    "n_agents": n_ag,
                }
                done_flags[i] = True

    tier_stats: dict[str, list] = {
        "goal_rate": [],
        "mean_reward": [],
        "episode_length": [],
        "n_agents": [],
    }
    for stats in captured:
        if stats is None:
            continue
        for k in tier_stats:
            tier_stats[k].append(stats[k])
    return tier_stats


def run_evaluation(
    actor_critic: ActorCritic,
    obs_normalizer,
    env_config: CrowdEnvConfig,
    obs_config: ObsConfig,
    reward_config: RewardConfig,
    eval_cfg: dict,
    device: torch.device,
    seed: int,
    results_dir: Path,
    max_agents: int,
) -> None:
    """Run quantitative evaluation and save bar chart.

    Uses a GPU-batched env per tier with one parallel env per requested
    episode -- the same path used during training, so all n_eval episodes
    of a tier run as a single batched GPU forward pass per step.
    """
    n_eval = eval_cfg.get("n_episodes", 10)
    agent_range = tuple(eval_cfg.get("agent_range", [8, 25]))
    eval_geometry = env_config.geometry

    eval_results = {}
    actor_critic.eval()

    tiers_to_eval = [
        GeometryTier.TIER_0,
        GeometryTier.TIER_1,
        GeometryTier.TIER_2,
        GeometryTier.TIER_3A,
        GeometryTier.TIER_3B,
    ]
    print(
        f"  Running evaluation on GPU: {len(tiers_to_eval)} tiers x {n_eval} episodes "
        f"({n_eval} parallel envs per tier)...",
        flush=True,
    )
    eval_start = time.time()

    for tier in tiers_to_eval:
        tier_start = time.time()

        eval_env_config = CrowdEnvConfig(
            geometry=eval_geometry,
            geometry_tiers=[tier],
            spawn=SpawnConfig(n_agents_range=agent_range),
            obs=obs_config,
            reward=reward_config,
            max_steps=env_config.max_steps,
        )
        eval_torch_config = EnvConfig.from_crowd_env_config(
            eval_env_config,
            max_agents=max_agents,
            max_segments=512,
        )
        episode_factory = make_episode_factory(eval_env_config)

        batched_env = BatchedTorchEnv(
            n_envs=n_eval,
            config=eval_torch_config,
            make_episode_fn=episode_factory,
            device=device,
            seed=seed + 1000,
            n_reset_workers=min(n_eval, 8),
            compile_step=False,  # short eval -- compile warmup costs more than it saves
        )

        try:
            tier_stats = _collect_eval_episodes_batched(
                batched_env,
                actor_critic,
                obs_normalizer,
                env_config.obs.obs_dim,
                env_config.action.action_dim,
                env_config.max_steps,
                device,
            )
        finally:
            batched_env.close()

        eval_results[tier.name] = tier_stats
        print(
            f"    {tier.name}: goal_rate={np.mean(tier_stats['goal_rate']):.2f} "
            f"avg_len={np.mean(tier_stats['episode_length']):.0f} "
            f"({time.time() - tier_start:.1f}s)",
            flush=True,
        )

    print(f"  Evaluation complete in {time.time() - eval_start:.1f}s", flush=True)
    actor_critic.train()

    # Print table
    print(f"\n{'Tier':<10} {'GoalRate':>10} {'Reward':>10} {'EpLen':>8} {'Agents':>8}")
    print("-" * 50)
    for tier_name, stats in eval_results.items():
        print(
            f"{tier_name:<10} "
            f"{np.mean(stats['goal_rate']):>8.3f}+/-{np.std(stats['goal_rate']):.3f} "
            f"{np.mean(stats['mean_reward']):>8.2f}+/-{np.std(stats['mean_reward']):.2f} "
            f"{np.mean(stats['episode_length']):>6.0f}+/-{np.std(stats['episode_length']):.0f} "
            f"{np.mean(stats['n_agents']):>6.1f}"
        )

    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    tier_names = list(eval_results.keys())
    x = np.arange(len(tier_names))
    bar_colors = ["#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4"]

    for ax, metric, ylabel in zip(
        axes,
        ["goal_rate", "mean_reward", "episode_length"],
        ["Goal Rate", "Mean Reward", "Episode Length"],
    ):
        means = [np.mean(eval_results[t][metric]) for t in tier_names]
        stds = [np.std(eval_results[t][metric]) for t in tier_names]
        ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors)
        ax.set_xticks(x)
        ax.set_xticklabels(tier_names, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Evaluation ({n_eval} episodes per tier)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(results_dir / "evaluation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Evaluation chart -> {results_dir / 'evaluation.png'}")


def save_trajectory_plots(
    actor_critic: ActorCritic,
    obs_normalizer,
    env_config: CrowdEnvConfig,
    obs_config: ObsConfig,
    reward_config: RewardConfig,
    device: torch.device,
    seed: int,
    results_dir: Path,
) -> None:
    """Run 8 evaluation scenarios and save trajectory plot."""
    from crowdrl_env.visualiser import plot_geometry

    eval_geometry = env_config.geometry
    eval_tiers = [
        ("Tier 0 -- Open field", GeometryTier.TIER_0, (5, 10)),
        ("Tier 0 -- Dense", GeometryTier.TIER_0, (15, 25)),
        ("Tier 1 -- Corridor", GeometryTier.TIER_1, (8, 15)),
        ("Tier 1 -- Bottleneck", GeometryTier.TIER_1, (15, 25)),
        ("Tier 2 -- Branching", GeometryTier.TIER_2, (10, 20)),
        ("Tier 2 -- Dense", GeometryTier.TIER_2, (20, 35)),
        ("Tier 3A -- Room + obstacles", GeometryTier.TIER_3A, (15, 30)),
        ("Tier 3B -- Composed rooms", GeometryTier.TIER_3B, (20, 40)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(26, 14))
    axes_flat = axes.flatten()
    actor_critic.eval()

    for idx, (title, tier, agent_range) in enumerate(eval_tiers):
        ax = axes_flat[idx]
        eval_env = CrowdEnv(
            config=CrowdEnvConfig(
                geometry=eval_geometry,
                geometry_tiers=[tier],
                spawn=SpawnConfig(n_agents_range=agent_range),
                obs=obs_config,
                reward=reward_config,
                max_steps=env_config.max_steps,
            ),
            seed=seed + idx + 100,
        )

        trajs, goals, polygon, reached, _info = _run_episode_with_trajectories(
            eval_env, actor_critic, obs_normalizer, device
        )
        plot_geometry(polygon, ax=ax)

        n_agents = len(trajs)
        cmap_fn = cm.get_cmap("tab20", n_agents)
        for i in range(n_agents):
            traj = np.array(trajs[i])
            color = cmap_fn(i % 20)
            alpha = 0.8 if reached[i] else 0.3
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=0.8, alpha=alpha)
            ax.plot(traj[0, 0], traj[0, 1], "o", color=color, markersize=3, alpha=alpha)
            ax.plot(goals[i, 0], goals[i, 1], "*", color=color, markersize=6, alpha=alpha)

        goal_rate = reached.sum() / n_agents if n_agents > 0 else 0
        ax.set_title(f"{title}\n{n_agents} agents, {goal_rate:.0%} reached goal", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    actor_critic.train()
    fig.suptitle("Trained Policy -- Agent Trajectories", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(results_dir / "trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Trajectory plots -> {results_dir / 'trajectories.png'}")


def _run_episode_with_trajectories(env, actor_critic, obs_normalizer, device, max_steps=2000):
    """Run one episode collecting full agent trajectories."""
    obs, info = env.reset()
    n_agents = info["n_agents"]
    trajectories = [[] for _ in range(n_agents)]
    goal_positions = env._world.goal_positions.copy()
    polygon = env._world.walkable_polygon
    reached_goal = np.zeros(n_agents, dtype=bool)

    for i in range(n_agents):
        trajectories[i].append(env._world.positions[i].copy())

    for _step in range(max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs_normalizer is not None:
            obs_norm = obs_normalizer.normalize(obs_t)
        else:
            obs_norm = obs_t

        with torch.no_grad():
            actions, _, _, _, _ = actor_critic.get_action_and_value(obs_norm)

        obs, _rewards, terminated, _truncated, step_info = env.step(actions.cpu().numpy())

        for i in range(n_agents):
            trajectories[i].append(env._world.positions[i].copy())
        reached_goal |= terminated

        if step_info.get("episode_over", False):
            break

    return trajectories, goal_positions, polygon, reached_goal, info


def try_render_video(
    actor_critic: ActorCritic,
    obs_normalizer,
    env_config: CrowdEnvConfig,
    obs_config: ObsConfig,
    reward_config: RewardConfig,
    eval_cfg: dict,
    device: torch.device,
    seed: int,
    results_dir: Path,
) -> None:
    """Render a trajectory video if the visualiser supports it."""
    try:
        from crowdrl_env.visualiser import collect_episode_frames, render_episode_video
    except ImportError:
        print("  Video rendering skipped (visualiser not available)")
        return

    eval_tiers = [
        ("Tier 0 -- Open field", GeometryTier.TIER_0, (5, 10)),
        ("Tier 0 -- Dense", GeometryTier.TIER_0, (15, 25)),
        ("Tier 1 -- Corridor", GeometryTier.TIER_1, (8, 15)),
        ("Tier 1 -- Bottleneck", GeometryTier.TIER_1, (15, 25)),
        ("Tier 2 -- Branching", GeometryTier.TIER_2, (10, 20)),
        ("Tier 2 -- Dense", GeometryTier.TIER_2, (20, 35)),
        ("Tier 3A -- Room + obstacles", GeometryTier.TIER_3A, (15, 30)),
        ("Tier 3B -- Composed rooms", GeometryTier.TIER_3B, (20, 40)),
    ]

    scenario_idx = eval_cfg.get("video_scenario", 7)
    scenario_idx = min(scenario_idx, len(eval_tiers) - 1)
    title, tier, agent_range = eval_tiers[scenario_idx]

    video_env = CrowdEnv(
        config=CrowdEnvConfig(
            geometry=env_config.geometry,
            geometry_tiers=[tier],
            spawn=SpawnConfig(n_agents_range=agent_range),
            obs=obs_config,
            reward=reward_config,
            max_steps=env_config.max_steps,
        ),
        seed=seed + scenario_idx + 100,
    )

    actor_critic.eval()
    frames = collect_episode_frames(video_env, actor_critic, obs_normalizer, device)
    frames.title = title
    actor_critic.train()

    video_path = render_episode_video(
        frames, str(results_dir / "episode.mp4"), fps=50, trail_length=1000
    )
    print(f"  Video -> {video_path}")


# ============================================================================
# Resume helpers
# ============================================================================


def _load_history_and_infer_rollout(
    history_path: Path,
    phases: tuple,
) -> tuple[dict, int, int, list]:
    """Load history.json and infer the resume point from its contents.

    Parameters
    ----------
    history_path
        Path to ``history.json`` written by a previous run.
    phases
        The curriculum phases tuple, used to map ``phase_idx`` indices back
        to phase name strings for ``phase_transitions`` reconstruction.

    Returns
    -------
    history
        defaultdict-compatible dict with per-episode and per-rollout lists.
    last_rollout
        Number of rollouts already completed. The next rollout to run is
        ``last_rollout + 1``. Inferred from ``len(history['policy_loss'])``
        since policy_loss is appended exactly once per rollout.
    total_episodes
        Number of completed episodes. Inferred from ``len(history['goal_rate'])``.
    phase_transitions
        Reconstructed ``[(episode_idx, phase_name), ...]`` from ``phase_idx``
        deltas in the history. Used to draw phase backgrounds in plots.
    """
    with open(history_path) as f:
        saved = json.load(f)

    history: dict[str, list] = defaultdict(list)
    for k, v in saved.items():
        history[k] = list(v)

    last_rollout = len(history.get("policy_loss", []))
    total_episodes = len(history.get("goal_rate", []))

    # Reconstruct phase transitions from phase_idx deltas
    phase_transitions: list[tuple[int, str]] = []
    prev_pidx = None
    for ep_idx, pidx in enumerate(history.get("phase_idx", [])):
        if prev_pidx is not None and pidx != prev_pidx:
            name = phases[pidx].name if 0 <= pidx < len(phases) else f"phase_{pidx}"
            phase_transitions.append((ep_idx + 1, name))
        prev_pidx = pidx

    return history, last_rollout, total_episodes, phase_transitions


# ============================================================================
# Training worker (one per GPU rank)
# ============================================================================


def train_worker(
    cfg: dict,
    results_dir: Path,
    resume_training: bool = False,
    start_from_zero: bool = False,
) -> None:
    """Main training loop -- runs on each DDP rank.

    Parameters
    ----------
    cfg
        Parsed YAML config dict.
    results_dir
        Output directory for checkpoints, plots, history, ONNX policy.
    resume_training
        If True, load ``checkpoint_final.pt`` and ``history.json`` from
        ``results_dir`` and continue training from the last completed rollout.
    start_from_zero
        Only valid with ``resume_training``. Load weights and normalizer
        statistics from the checkpoint but restart rollout counting, curriculum,
        and history from scratch. Optimizer state is also reset.
    """
    rank, world_size, device = init_distributed(cfg.get("ddp_backend", "nccl"))

    # -- Build all configs from YAML ----------------------------------------
    seed = cfg.get("seed", 42)
    seed_everything(seed)
    env_seed = distributed_seed(seed)

    env_config = build_env_config(cfg)
    net_config = build_net_config(cfg, env_config)
    ppo_config = build_ppo_config(cfg)
    curriculum_config = build_curriculum_config(cfg)

    n_envs = cfg.get("n_envs", 64)
    max_agents = cfg.get("max_agents", 100)
    steps_per_collect = cfg.get("steps_per_collect", n_envs * 20000)
    n_rollouts = cfg.get("n_rollouts", 500)
    log_interval = cfg.get("log_interval", 5)
    compile_step = cfg.get("compile_step", True)

    # -- Validate max_agents covers curriculum --------------------------------
    max_phase_agents = max(p.n_agents_range[1] for p in curriculum_config.phases)
    assert max_agents >= max_phase_agents, (
        f"max_agents={max_agents} but curriculum needs up to {max_phase_agents}"
    )

    torch_env_config = EnvConfig.from_crowd_env_config(
        env_config, max_agents=max_agents, max_segments=512
    )

    # -- Create components ---------------------------------------------------
    actor_critic = ActorCritic(net_config).to(device)
    updater = MAPPOUpdater(actor_critic, ppo_config, device)
    curriculum = CurriculumManager(curriculum_config)
    obs_normalizer = TorchRunningNormalizer(shape=(env_config.obs.obs_dim,), device=device)
    reward_normalizer = RewardNormalizer(gamma=ppo_config.gamma)

    # -- Optional resume from previous run ----------------------------------
    # Defaults for a fresh run; overwritten below when resuming.
    start_rollout = 1
    prior_total_episodes = 0
    prior_total_agent_steps = 0
    loaded_history: dict[str, list] | None = None
    loaded_phase_transitions: list[tuple[int, str]] = []

    if resume_training:
        ckpt_path = results_dir / "checkpoint_final.pt"
        history_path = results_dir / "history.json"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"--resume_training requested but no checkpoint found at {ckpt_path}"
            )
        if not start_from_zero and not history_path.exists():
            raise FileNotFoundError(
                f"--resume_training requested but no history at {history_path}. "
                "Pass --start_from_zero to restart the curriculum from scratch."
            )

        if start_from_zero:
            # Load weights + normalizer stats only; discard optimizer, curriculum,
            # and rollout counters. We build throwaway objects so load_checkpoint
            # can populate them without polluting our real state.
            throwaway_updater = MAPPOUpdater(actor_critic, ppo_config, device)
            throwaway_curriculum = CurriculumManager(curriculum_config)
            load_checkpoint(
                ckpt_path,
                actor_critic,
                throwaway_updater,
                obs_normalizer,
                reward_normalizer,
                throwaway_curriculum,
            )
            # `updater` keeps its freshly initialised optimizer state; weights
            # are now the pre-trained ones because load_checkpoint mutated
            # actor_critic in place.
            if is_main_rank():
                print(f"Loaded weights + normalizers from {ckpt_path}")
                print("--start_from_zero: curriculum reset to phase 0, rollout 0")
                # Archive the old history so plots don't mix old and new runs.
                if history_path.exists():
                    archive = results_dir / "history_previous.json"
                    history_path.replace(archive)
                    print(f"Previous history archived to {archive}")
        else:
            # Full resume: weights, optimizer, normalizers, curriculum, counters.
            prior_total_agent_steps, prior_total_episodes = load_checkpoint(
                ckpt_path,
                actor_critic,
                updater,
                obs_normalizer,
                reward_normalizer,
                curriculum,
            )
            history, last_rollout, history_episodes, loaded_phase_transitions = (
                _load_history_and_infer_rollout(history_path, curriculum_config.phases)
            )
            loaded_history = history
            start_rollout = last_rollout + 1
            # Trust the history for episode count -- the checkpoint may have been
            # saved at a slightly different point than the last history flush.
            # Use whichever is larger so we never rewind the counter.
            prior_total_episodes = max(prior_total_episodes, history_episodes)
            if is_main_rank():
                print(f"Resumed from {ckpt_path}")
                print(
                    f"  last completed rollout: {last_rollout}, "
                    f"total_episodes: {prior_total_episodes}, "
                    f"total_agent_steps: {prior_total_agent_steps:,}"
                )
                print(f"  continuing at phase: {curriculum.current_phase.name}")
                if start_rollout > n_rollouts:
                    print(
                        f"Warning: start_rollout ({start_rollout}) > n_rollouts "
                        f"({n_rollouts}); increase n_rollouts in the config to "
                        "continue training."
                    )

    if is_main_rank():
        n_params = sum(p.numel() for p in actor_critic.parameters())
        print(f"Actor-Critic: {n_params:,} parameters")
        print(f"Device: {device}, world_size: {world_size}")
        print(f"Envs per rank: {n_envs}, steps/collect: {steps_per_collect:,}")
        print(f"Effective batch: {steps_per_collect * world_size:,} agent-steps/update")
        print(f"Curriculum: {' -> '.join(p.name for p in curriculum_config.phases)}")
        results_dir.mkdir(parents=True, exist_ok=True)
        # Save resolved config for reproducibility
        with open(results_dir / "config_resolved.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # -- Launch batched env --------------------------------------------------
    cur_env_config = curriculum.make_env_config(env_config)
    episode_factory = make_episode_factory(cur_env_config)

    batched_env = BatchedTorchEnv(
        n_envs=n_envs,
        config=torch_env_config,
        make_episode_fn=episode_factory,
        device=device,
        seed=env_seed,
        n_reset_workers=n_envs,
        compile_step=compile_step,
    )

    collector = TorchRolloutCollector(
        batched_env,
        actor_critic,
        obs_normalizer,
        reward_normalizer,
        device,
        obs_dim=env_config.obs.obs_dim,
        action_dim=env_config.action.action_dim,
    )

    # -- Warmup torch.compile ------------------------------------------------
    if is_main_rank():
        print("Warming up torch.compile...", flush=True)
    warmup_start = time.time()
    _states, _obs = batched_env.reset_all()
    batched_env.states = _states
    compiled = batched_env.warmup(n_steps=3)
    if is_main_rank():
        elapsed = time.time() - warmup_start
        status = "Compiled" if compiled else "Eager (compile unavailable)"
        print(f"{status} step ready in {elapsed:.1f}s\n", flush=True)

    # -- Training loop -------------------------------------------------------
    # When resuming (not --start_from_zero), seed history and counters from
    # the loaded state so plots and summaries span both runs.
    if loaded_history is not None:
        history: dict[str, list] = defaultdict(list, loaded_history)
        phase_transitions: list[tuple[int, str]] = list(loaded_phase_transitions)
    else:
        history = defaultdict(list)
        phase_transitions = []

    total_agent_steps = prior_total_agent_steps
    total_episodes = prior_total_episodes
    # Track per-session agent-steps separately so SPS reflects only the
    # current session's work, not the sum of all previous runs.
    session_agent_steps = 0
    start_time = time.time()

    if is_main_rank():
        sps_header = "SPS (local)" if world_size > 1 else "SPS"
        sps_global_header = " | SPS (global)" if world_size > 1 else ""
        print(
            f"{'Roll':>5} | {'Episodes':>8} | {'Steps':>12} | {'GoalRate':>8} | "
            f"{'Reward':>8} | {'Agents':>6} | {'Phase':>8} | {sps_header:>11}"
            f"{sps_global_header}",
            flush=True,
        )
        print("-" * (88 + (16 if world_size > 1 else 0)), flush=True)

    try:
        for rollout in range(start_rollout, n_rollouts + 1):
            # Collect
            episode_stats_list = collector.collect(steps_per_collect)
            flat_batch = collector.compute_gae_and_flatten(ppo_config.gamma, ppo_config.gae_lambda)

            # PPO update -- under DDP all ranks must agree on whether to call
            # update(), otherwise mismatched collectives cause NCCL deadlock.
            local_has_data = flat_batch.batch_size > 0
            if world_size > 1:
                flag = torch.tensor([int(local_has_data)], dtype=torch.int32, device=device)
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
                all_have_data = bool(flag.item())
            else:
                all_have_data = local_has_data

            if all_have_data:
                update_metrics = updater.update(flat_batch)
            else:
                update_metrics = {}

            local_steps = collector.total_active_agent_steps
            total_agent_steps += local_steps
            session_agent_steps += local_steps

            # Sync normalizers
            obs_normalizer.sync_across_ranks()
            if reward_normalizer is not None:
                sync_reward_normalizer(reward_normalizer, device)

            # LR decay -- progress is measured against the absolute rollout
            # number so resumed runs continue along the same schedule curve.
            progress = rollout / n_rollouts
            updater.update_learning_rate(progress)

            # Curriculum (rank 0 aggregates)
            all_episodes = gather_episode_stats(episode_stats_list)
            total_episodes += len(all_episodes) if is_main_rank() else len(episode_stats_list)

            prev_phase_idx = curriculum.current_phase_idx
            if is_main_rank():
                for ep in all_episodes:
                    cs = EpisodeStats(
                        goal_rate=ep["goal_rate"],
                        n_agents=ep["n_agents"],
                        episode_length=ep["episode_length"],
                        mean_reward=ep["mean_reward"],
                    )
                    curriculum.report_episode(cs)

            broadcast_curriculum_state(curriculum)

            # Detect phase change on every rank from the post-broadcast state
            # so non-main ranks also reassign their episode factory; otherwise
            # they stay stuck on the initial phase and flood the batch with
            # easy-phase episodes.
            if curriculum.current_phase_idx != prev_phase_idx:
                phase_transitions.append((total_episodes, curriculum.current_phase.name))
                cur_env_config = curriculum.make_env_config(env_config)
                batched_env.make_episode_fn = make_episode_factory(cur_env_config)
                if is_main_rank():
                    print(
                        f"\n>>> Phase advanced to: {curriculum.current_phase.name} "
                        f"(episode {total_episodes})\n",
                        flush=True,
                    )

            # Record history
            source_episodes = all_episodes if is_main_rank() else episode_stats_list
            for ep in source_episodes:
                history["goal_rate"].append(ep["goal_rate"])
                history["mean_reward"].append(ep["mean_reward"])
                history["episode_length"].append(ep["episode_length"])
                history["n_agents"].append(ep["n_agents"])
                history["phase_idx"].append(curriculum.current_phase_idx)
                history["geometry_tier"].append(ep.get("geometry_tier", "unknown"))

            if update_metrics:
                history["policy_loss"].append(update_metrics.get("policy_loss", 0))
                history["value_loss"].append(update_metrics.get("value_loss", 0))
                history["entropy"].append(update_metrics.get("entropy", 0))
                history["approx_kl"].append(update_metrics.get("approx_kl", 0))

            # Logging (rank 0)
            if rollout % log_interval == 0 and episode_stats_list and is_main_rank():
                elapsed = time.time() - start_time
                # SPS reflects only the current session, not time spent in
                # previous (resumed) runs which have no wall-clock here.
                local_sps = session_agent_steps / max(elapsed, 1)
                window = min(100, len(history["goal_rate"]))
                avg_gr = np.mean(history["goal_rate"][-window:])
                avg_rw = np.mean(history["mean_reward"][-window:])
                avg_ag = np.mean(history["n_agents"][-window:])
                global_col = f" | {local_sps * world_size:>11.0f}" if world_size > 1 else ""
                print(
                    f"{rollout:>5} | {total_episodes:>8} | {total_agent_steps:>12,} | "
                    f"{avg_gr:>8.3f} | {avg_rw:>8.2f} | "
                    f"{avg_ag:>6.1f} | {curriculum.current_phase.name:>8} | "
                    f"{local_sps:>11.0f}{global_col}",
                    flush=True,
                )

    finally:
        batched_env.close()

    elapsed = time.time() - start_time
    if is_main_rank():
        local_sps = session_agent_steps / max(elapsed, 1)
        global_sps = local_sps * world_size
        print(
            f"\nDone: {total_episodes} episodes total, "
            f"{session_agent_steps:,} agent-steps this session "
            f"({total_agent_steps:,} cumulative) in {elapsed:.1f}s"
        )
        if world_size > 1:
            print(
                f"Throughput: {local_sps:.0f} agent-steps/sec (local), "
                f"{global_sps:.0f} agent-steps/sec (global, {world_size} GPUs)"
            )
        else:
            print(f"Throughput: {local_sps:.0f} agent-steps/sec")

    # -- Post-training (rank 0 only) ----------------------------------------
    if is_main_rank():
        print("\n--- Saving results ---")

        # Training history (JSON)
        history_serializable = {
            k: [x if not isinstance(x, np.integer) else int(x) for x in v]
            for k, v in history.items()
        }
        with open(results_dir / "history.json", "w") as f:
            json.dump(history_serializable, f)
        print(f"  History -> {results_dir / 'history.json'}")

        # Training curves
        save_training_plots(history, phase_transitions, curriculum_config.phases, results_dir)

        # Phase summary
        save_phase_summary(history, curriculum_config.phases, results_dir)

        # Checkpoint -- save early so training output is never lost even
        # if a later step fails.
        save_checkpoint(
            results_dir / "checkpoint_final.pt",
            actor_critic,
            updater,
            obs_normalizer,
            reward_normalizer,
            curriculum,
            total_agent_steps,
            total_episodes,
        )
        print(f"  Checkpoint -> {results_dir / 'checkpoint_final.pt'}")

        # Trajectory plots
        print("  Generating trajectory plots (8 CPU eval episodes)...", flush=True)
        save_trajectory_plots(
            actor_critic,
            obs_normalizer,
            env_config,
            env_config.obs,
            env_config.reward,
            device,
            seed,
            results_dir,
        )

        # Quantitative evaluation
        eval_cfg = cfg.get("eval", {})
        run_evaluation(
            actor_critic,
            obs_normalizer,
            env_config,
            env_config.obs,
            env_config.reward,
            eval_cfg,
            device,
            seed,
            results_dir,
            max_agents,
        )

        # Video
        if eval_cfg.get("render_video", True):
            print("  Rendering trajectory video (1 CPU eval episode + ffmpeg)...", flush=True)
            try_render_video(
                actor_critic,
                obs_normalizer,
                env_config,
                env_config.obs,
                env_config.reward,
                eval_cfg,
                device,
                seed,
                results_dir,
            )

        # ONNX export -- last, because it creates a CPU copy of the actor
        # and has historically had device-aliasing bugs. Running it last
        # means any regression cannot corrupt earlier evaluation steps.
        print("  Exporting ONNX policy...", flush=True)
        onnx_path = results_dir / "policy.onnx"
        export_onnx(actor_critic.actor, obs_normalizer, onnx_path)
        print(f"  ONNX policy -> {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

        print(f"\nAll results saved to: {results_dir}/")

    cleanup_distributed()


# ============================================================================
# Entry point -- auto-launches torchrun when multiple GPUs are available
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="MAPPO training with automatic multi-GPU support")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--gpus", type=int, default=None, help="Number of GPUs (default: all available)"
    )
    parser.add_argument(
        "--resume_training",
        action="store_true",
        help="Resume from results_<config>/checkpoint_final.pt. The last completed "
        "rollout is inferred from history.json in the same folder.",
    )
    parser.add_argument(
        "--start_from_zero",
        action="store_true",
        help="Requires --resume_training. Load pre-trained weights and normalizer "
        "statistics from the checkpoint, but restart the curriculum from phase 0 "
        "at rollout 0 with a fresh optimizer and fresh history. The old history "
        "file is moved to history_previous.json before training starts.",
    )
    args = parser.parse_args()

    if args.start_from_zero and not args.resume_training:
        parser.error("--start_from_zero requires --resume_training")

    cfg = load_config(args.config)
    config_stem = Path(args.config).stem
    results_dir = Path(f"results_{config_stem}")

    # Already running under torchrun -- go straight to worker
    if "WORLD_SIZE" in os.environ:
        train_worker(
            cfg,
            results_dir,
            resume_training=args.resume_training,
            start_from_zero=args.start_from_zero,
        )
        return

    n_gpus = args.gpus or torch.cuda.device_count()

    if n_gpus > 1:
        print(f"Detected {n_gpus} GPUs -- launching via torchrun")
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(n_gpus),
            *sys.argv,
        ]
        sys.exit(subprocess.call(cmd))
    else:
        if n_gpus == 1:
            print("Single GPU detected -- running directly")
        else:
            print("No GPU detected -- running on CPU")
        train_worker(
            cfg,
            results_dir,
            resume_training=args.resume_training,
            start_from_zero=args.start_from_zero,
        )


if __name__ == "__main__":
    main()
