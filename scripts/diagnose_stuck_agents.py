"""Diagnose whether TIER_3B episode timeouts are caused by stuck agents.

Loads a trained checkpoint and runs a large batch of TIER_3B episodes on GPU,
recording per-step per-agent state. Then analyses whether agents that fail to
reach their goal are (a) truly stuck in a local minimum, (b) queuing
productively, or (c) oscillating.

Usage
-----
    uv run python scripts/diagnose_stuck_agents.py \
        --checkpoint results_baseline_seed42/checkpoint_final.pt \
        --tier TIER_3B \
        --n-envs 32 \
        --out diagnostic_tier_3B

Outputs to ``diagnostic_<name>/``:
- summary.txt          human-readable report
- per_agent.csv        one row per agent across all episodes
- trajectories.png     top-N suspect trajectories visualised
- progress_histogram.png
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from crowdrl_core.action import ActionConfig
from crowdrl_core.observation import ObsConfig
from crowdrl_env.crowd_env import CrowdEnvConfig
from crowdrl_env.geometry_generator import GeometryConfig, GeometryTier
from crowdrl_env.reward import RewardConfig
from crowdrl_env.spawner import SpawnConfig
from crowdrl_torch import BatchedTorchEnv, TorchRunningNormalizer, make_episode_factory
from crowdrl_torch.types import EnvConfig
from crowdrl_train.config import NetworkConfig
from crowdrl_train.networks import ActorCritic


# Thresholds used to classify "stuck" behaviour
STUCK_SPEED_THRESHOLD = 0.10  # m/s below this counts as "not walking"
STUCK_STEP_WINDOW = 200  # ~2s at dt=0.01; tune after seeing distributions
PROGRESS_WINDOW = 200  # steps for goal-distance-progress sliding window
PROGRESS_EPSILON = 0.20  # metres -- less progress than this in window = "no progress"


@dataclass
class EpisodeTrace:
    """Per-episode per-step log."""

    positions: np.ndarray  # (T, N, 2)
    velocities: np.ndarray  # (T, N, 2)
    goal_distances: np.ndarray  # (T, N)
    active_masks: np.ndarray  # (T, N)  # active == still in play
    reached_goal: np.ndarray  # (N,) bool -- terminal state
    goal_positions: np.ndarray  # (N, 2)
    wall_segments: np.ndarray  # (S, 2, 2)
    n_segments: int
    n_agents: int
    episode_length: int
    tier: str


def build_env_config(max_steps: int) -> CrowdEnvConfig:
    """Config matching the baseline training run."""
    return CrowdEnvConfig(
        geometry=GeometryConfig(
            min_side=8.0,
            max_side=15.0,
            corridor_width_range=(2.0, 4.0),
            corridor_length_range=(8.0, 18.0),
        ),
        obs=ObsConfig(use_navmesh=True),
        action=ActionConfig(
            max_heading_change=np.radians(15.0),
            max_torso_change=np.radians(15.0),
        ),
        reward=RewardConfig(),
        max_steps=max_steps,
    )


def run_batched_episodes(
    checkpoint_path: Path,
    tier: str,
    n_envs: int,
    max_steps: int,
    agent_range: tuple[int, int],
    seed: int,
    device: torch.device,
) -> list[EpisodeTrace]:
    """Roll out n_envs episodes in parallel and return per-env traces."""
    env_config = build_env_config(max_steps)
    tier_enum = GeometryTier[tier]
    eval_env_config = CrowdEnvConfig(
        geometry=env_config.geometry,
        geometry_tiers=[tier_enum],
        spawn=SpawnConfig(n_agents_range=agent_range),
        obs=env_config.obs,
        reward=env_config.reward,
        action=env_config.action,
        max_steps=max_steps,
    )

    max_agents = max(agent_range[1] + 10, 64)
    torch_env_config = EnvConfig.from_crowd_env_config(
        eval_env_config,
        max_agents=max_agents,
        max_segments=512,
    )
    obs_dim = env_config.obs.obs_dim
    action_dim = env_config.action.action_dim
    N = torch_env_config.max_agents

    # Load policy
    actor_critic = ActorCritic(NetworkConfig(obs_dim=obs_dim, action_dim=action_dim)).to(device)
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    actor_critic.load_state_dict(checkpoint["actor_critic"])
    actor_critic.eval()

    obs_normalizer = TorchRunningNormalizer(shape=(obs_dim,), device=device)
    if "obs_normalizer" in checkpoint:
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])

    # Build env
    factory = make_episode_factory(eval_env_config)
    batched_env = BatchedTorchEnv(
        n_envs=n_envs,
        config=torch_env_config,
        make_episode_fn=factory,
        device=device,
        seed=seed,
        n_reset_workers=min(n_envs, 8),
        compile_step=False,
    )

    try:
        states, obs = batched_env.reset_all()
        batched_env.states = states

        # Snapshot initial counts and per-env tier names
        n_agents_initial = states.n_agents.cpu().numpy().astype(np.int32).copy()
        env_tiers_initial = list(batched_env.env_tiers)

        # Preallocate per-step storage on CPU (avoid GPU fragmentation)
        T = max_steps + 10
        E = n_envs
        positions_log = np.zeros((E, T, N, 2), dtype=np.float32)
        velocities_log = np.zeros((E, T, N, 2), dtype=np.float32)
        goal_dist_log = np.zeros((E, T, N), dtype=np.float32)
        active_log = np.zeros((E, T, N), dtype=bool)
        terminated_cum = np.zeros((E, N), dtype=bool)
        ep_lengths = np.zeros(E, dtype=np.int32)
        done_flags = np.zeros(E, dtype=bool)

        # Snapshot static per-episode state for later plotting
        goal_positions_snap = states.goal_positions.cpu().numpy().copy()  # (E, N, 2)
        wall_segments_snap = states.wall_segments.cpu().numpy().copy()  # (E, S, 2, 2)
        n_segments_snap = states.n_segments.cpu().numpy().copy()

        for t in range(max_steps + 10):
            if done_flags.all():
                break

            obs_norm = obs_normalizer.normalize(obs)
            with torch.no_grad():
                actions, _, _, _, _ = actor_critic.get_action_and_value(
                    obs_norm.reshape(-1, obs_dim),
                    deterministic=True,
                )
            actions_gpu = actions.reshape(E, N, action_dim)

            states, obs, rewards, terminated, _trunc = batched_env.step(actions_gpu)

            positions_np = states.positions.cpu().numpy()  # (E, N, 2)
            velocities_np = states.velocities.cpu().numpy()  # (E, N, 2)
            goal_pos_np = states.goal_positions.cpu().numpy()
            active_np = states.active_mask.cpu().numpy()  # (E, N)
            terminated_np = terminated.cpu().numpy()
            episode_over_np = batched_env.episode_over.cpu().numpy()

            for i in range(E):
                if done_flags[i]:
                    continue
                positions_log[i, t] = positions_np[i]
                velocities_log[i, t] = velocities_np[i]
                goal_dist_log[i, t] = np.linalg.norm(goal_pos_np[i] - positions_np[i], axis=-1)
                active_log[i, t] = active_np[i]
                terminated_cum[i] |= terminated_np[i]

                if active_np[i].any():
                    ep_lengths[i] = t + 1
                if episode_over_np[i]:
                    done_flags[i] = True
    finally:
        batched_env.close()

    # Build traces
    traces: list[EpisodeTrace] = []
    for i in range(E):
        n_ag = int(n_agents_initial[i])
        el = int(ep_lengths[i])
        if n_ag == 0 or el == 0:
            continue
        traces.append(
            EpisodeTrace(
                positions=positions_log[i, :el, :n_ag].copy(),
                velocities=velocities_log[i, :el, :n_ag].copy(),
                goal_distances=goal_dist_log[i, :el, :n_ag].copy(),
                active_masks=active_log[i, :el, :n_ag].copy(),
                reached_goal=terminated_cum[i, :n_ag].copy(),
                goal_positions=goal_positions_snap[i, :n_ag].copy(),
                wall_segments=wall_segments_snap[i].copy(),
                n_segments=int(n_segments_snap[i]),
                n_agents=n_ag,
                episode_length=el,
                tier=env_tiers_initial[i],
            )
        )
    return traces


def classify_agent(
    positions: np.ndarray,
    velocities: np.ndarray,
    goal_distances: np.ndarray,
    active_mask: np.ndarray,
    reached_goal: bool,
) -> dict:
    """Classify a single agent's behaviour across an episode.

    Returns a dict with summary features and a tentative label.
    """
    T = positions.shape[0]
    speeds = np.linalg.norm(velocities, axis=-1)  # (T,)

    # Fraction of active steps below the speed threshold
    active_steps = active_mask.sum()
    if active_steps == 0:
        return {
            "label": "unused",
            "active_steps": 0,
            "final_goal_dist": float("nan"),
            "slow_fraction": float("nan"),
            "max_consecutive_slow": 0,
            "total_path_length": 0.0,
            "net_progress": 0.0,
            "oscillation_index": 0.0,
            "reached_goal": False,
        }
    slow_mask = (speeds < STUCK_SPEED_THRESHOLD) & active_mask
    slow_fraction = slow_mask.sum() / active_steps

    # Longest run of consecutive slow active steps
    max_run = 0
    cur_run = 0
    for t in range(T):
        if slow_mask[t]:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            if active_mask[t]:
                cur_run = 0

    # Path length vs net displacement -> oscillation indicator
    step_disp = np.linalg.norm(np.diff(positions, axis=0), axis=-1)  # (T-1,)
    valid_disp = step_disp[active_mask[1:]]
    total_path = float(valid_disp.sum())
    # Net progress measured on goal distance
    active_gd = goal_distances[active_mask]
    if active_gd.size >= 2:
        net_progress = float(active_gd[0] - active_gd[-1])
    else:
        net_progress = 0.0
    # Oscillation index: path length / |net displacement in (x,y)|
    first_active = int(np.argmax(active_mask))
    last_active = T - 1 - int(np.argmax(active_mask[::-1]))
    net_disp = float(np.linalg.norm(positions[last_active] - positions[first_active]))
    osc = total_path / max(net_disp, 1e-6)
    final_gd = float(goal_distances[last_active])

    # Label
    if reached_goal:
        label = "reached"
    elif max_run >= STUCK_STEP_WINDOW and net_progress < PROGRESS_EPSILON:
        label = "stuck"
    elif slow_fraction > 0.5 and net_progress < PROGRESS_EPSILON:
        label = "mostly_slow_no_progress"
    elif osc > 5.0 and net_progress < PROGRESS_EPSILON:
        label = "oscillating"
    elif net_progress >= PROGRESS_EPSILON:
        label = "slow_but_progressing"
    else:
        label = "other"

    return {
        "label": label,
        "active_steps": int(active_steps),
        "final_goal_dist": final_gd,
        "slow_fraction": float(slow_fraction),
        "max_consecutive_slow": int(max_run),
        "total_path_length": total_path,
        "net_progress": net_progress,
        "oscillation_index": float(osc),
        "reached_goal": bool(reached_goal),
    }


def plot_suspect_episode(
    trace: EpisodeTrace,
    suspect_agents: list[int],
    out_path: Path,
    title: str,
) -> None:
    """Plot trajectories for one episode, highlighting suspect agents."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: trajectories in the map
    seg_starts = trace.wall_segments[: trace.n_segments, 0]
    seg_ends = trace.wall_segments[: trace.n_segments, 1]
    for s, e in zip(seg_starts, seg_ends):
        ax1.plot([s[0], e[0]], [s[1], e[1]], "k-", linewidth=0.8, alpha=0.7)

    suspect_set = set(suspect_agents)
    for i in range(trace.n_agents):
        path = trace.positions[:, i]
        mask = trace.active_masks[:, i]
        if mask.sum() < 2:
            continue
        path_active = path[mask]
        if i in suspect_set:
            ax1.plot(
                path_active[:, 0], path_active[:, 1], "-", color="red", linewidth=1.5, alpha=0.85
            )
            ax1.scatter(
                [trace.positions[np.argmax(mask), i, 0]],
                [trace.positions[np.argmax(mask), i, 1]],
                c="red",
                s=40,
                marker="o",
                edgecolors="black",
                linewidths=0.5,
            )
            ax1.scatter(
                [trace.goal_positions[i, 0]],
                [trace.goal_positions[i, 1]],
                c="red",
                s=60,
                marker="*",
                edgecolors="black",
                linewidths=0.5,
            )
        else:
            color = "green" if trace.reached_goal[i] else "gray"
            ax1.plot(
                path_active[:, 0], path_active[:, 1], "-", color=color, linewidth=0.5, alpha=0.4
            )
    ax1.set_aspect("equal")
    ax1.set_title(f"{title} ({trace.n_agents} agents, {trace.reached_goal.sum()} reached)")
    ax1.grid(True, alpha=0.3)

    # Panel 2: speed over time for suspect agents
    for i in suspect_agents:
        mask = trace.active_masks[:, i]
        if mask.sum() < 2:
            continue
        speeds = np.linalg.norm(trace.velocities[:, i], axis=-1)
        active_t = np.where(mask)[0]
        ax2.plot(active_t, speeds[active_t], linewidth=0.9, label=f"agent {i}")
    ax2.axhline(
        STUCK_SPEED_THRESHOLD, linestyle="--", color="red", alpha=0.5, label="stuck threshold"
    )
    ax2.set_xlabel("step")
    ax2.set_ylabel("||velocity|| (m/s)")
    ax2.set_title("Suspect agents' speed over time")
    ax2.grid(True, alpha=0.3)
    if len(suspect_agents) <= 8:
        ax2.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tier", default="TIER_3B")
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--agent-min", type=int, default=20)
    ap.add_argument("--agent-max", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=9999)
    ap.add_argument("--out", type=Path, default=Path("diagnostic_stuck"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    print(
        f"Rolling out {args.n_envs} {args.tier} episodes "
        f"(agents=[{args.agent_min},{args.agent_max}], max_steps={args.max_steps})",
        flush=True,
    )
    traces = run_batched_episodes(
        args.checkpoint,
        args.tier,
        args.n_envs,
        args.max_steps,
        (args.agent_min, args.agent_max),
        args.seed,
        device,
    )
    print(f"Collected {len(traces)} episodes", flush=True)

    # Classify every agent in every episode
    rows = []
    for ep_idx, tr in enumerate(traces):
        for a in range(tr.n_agents):
            info = classify_agent(
                tr.positions[:, a],
                tr.velocities[:, a],
                tr.goal_distances[:, a],
                tr.active_masks[:, a],
                tr.reached_goal[a],
            )
            info["episode"] = ep_idx
            info["agent"] = a
            info["tier"] = tr.tier
            info["episode_length"] = tr.episode_length
            info["n_agents_episode"] = tr.n_agents
            rows.append(info)

    # Write CSV
    csv_path = args.out / "per_agent.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Aggregate stats
    labels: dict[str, int] = {}
    for r in rows:
        labels[r["label"]] = labels.get(r["label"], 0) + 1

    lines: list[str] = []
    lines.append(f"=== Stuck-agent diagnostic: {args.tier} ===")
    lines.append(f"Episodes: {len(traces)}")
    lines.append(f"Total agents: {len(rows)}")
    lines.append(
        f"Thresholds: speed<{STUCK_SPEED_THRESHOLD} for >={STUCK_STEP_WINDOW} "
        f"consecutive steps AND net_progress<{PROGRESS_EPSILON}"
    )
    lines.append("")
    lines.append("Label distribution:")
    for label in sorted(labels.keys(), key=lambda k: -labels[k]):
        pct = 100.0 * labels[label] / len(rows)
        lines.append(f"  {label:<26} {labels[label]:>5}  ({pct:.1f}%)")
    lines.append("")

    # Stuck-specific stats
    stuck_rows = [r for r in rows if r["label"] == "stuck"]
    reached_rows = [r for r in rows if r["label"] == "reached"]
    nonreached_rows = [r for r in rows if not r["reached_goal"]]
    lines.append(f"Reached-goal agents: {len(reached_rows)}")
    lines.append(f"Non-reached agents:  {len(nonreached_rows)}")
    lines.append(
        f"  of which 'stuck':  {len(stuck_rows)} "
        f"({100.0 * len(stuck_rows) / max(len(nonreached_rows), 1):.1f}% of non-reached)"
    )
    if stuck_rows:
        mcs = np.array([r["max_consecutive_slow"] for r in stuck_rows])
        sf = np.array([r["slow_fraction"] for r in stuck_rows])
        np_arr = np.array([r["net_progress"] for r in stuck_rows])
        lines.append(
            f"  stuck max_consecutive_slow: mean={mcs.mean():.0f}, "
            f"median={int(np.median(mcs))}, p90={int(np.percentile(mcs, 90))}"
        )
        lines.append(f"  stuck slow_fraction: mean={sf.mean():.2f}, median={np.median(sf):.2f}")
        lines.append(
            f"  stuck net_progress: mean={np_arr.mean():.2f}m, median={np.median(np_arr):.2f}m"
        )

    # Per-episode summary
    lines.append("")
    lines.append("Per-episode: fraction of agents in each label:")
    lines.append(
        f"{'ep':>4} {'tier':<9} {'len':>5} {'n_ag':>5} {'reached':>8} {'stuck':>6} {'slowNP':>7} {'osc':>5} {'other':>6}"
    )
    for ep_idx, tr in enumerate(traces):
        ep_rows = [r for r in rows if r["episode"] == ep_idx]
        n = len(ep_rows)
        if n == 0:
            continue
        counts = {
            "reached": 0,
            "stuck": 0,
            "mostly_slow_no_progress": 0,
            "oscillating": 0,
            "other": 0,
            "slow_but_progressing": 0,
            "unused": 0,
        }
        for r in ep_rows:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        lines.append(
            f"{ep_idx:>4} {tr.tier:<9} {tr.episode_length:>5d} {tr.n_agents:>5d} "
            f"{counts['reached']:>8d} {counts['stuck']:>6d} "
            f"{counts['mostly_slow_no_progress']:>7d} {counts['oscillating']:>5d} "
            f"{counts['other']:>6d}"
        )

    summary_text = "\n".join(lines)
    print()
    print(summary_text)
    (args.out / "summary.txt").write_text(summary_text + "\n")

    # Pick 3 most interesting episodes to plot: those with the most stuck agents
    episode_stuck_counts = {}
    for r in rows:
        if r["label"] == "stuck":
            episode_stuck_counts[r["episode"]] = episode_stuck_counts.get(r["episode"], 0) + 1

    sorted_eps = sorted(episode_stuck_counts.items(), key=lambda kv: -kv[1])[:3]
    if sorted_eps:
        for rank, (ep_idx, sc) in enumerate(sorted_eps):
            tr = traces[ep_idx]
            suspects = [
                r["agent"]
                for r in rows
                if r["episode"] == ep_idx
                and r["label"] in ("stuck", "mostly_slow_no_progress", "oscillating")
            ]
            plot_suspect_episode(
                tr,
                suspects,
                args.out / f"suspect_episode_{rank}_ep{ep_idx}.png",
                f"Episode {ep_idx} ({sc} stuck)",
            )
        print(f"\nTrajectory plots written to {args.out}/suspect_episode_*.png")

    # Histogram of max_consecutive_slow vs reached
    if rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        reached_arr = np.array([r["max_consecutive_slow"] for r in rows if r["reached_goal"]])
        stuck_arr = np.array([r["max_consecutive_slow"] for r in rows if not r["reached_goal"]])
        bins = np.linspace(0, max(args.max_steps, 1), 60)
        if reached_arr.size:
            ax.hist(reached_arr, bins=bins, alpha=0.5, label="reached", color="green")
        if stuck_arr.size:
            ax.hist(stuck_arr, bins=bins, alpha=0.5, label="non-reached", color="red")
        ax.axvline(
            STUCK_STEP_WINDOW,
            linestyle="--",
            color="black",
            label=f"threshold ({STUCK_STEP_WINDOW} steps)",
        )
        ax.set_xlabel("max consecutive slow (speed<0.1 m/s) steps")
        ax.set_ylabel("agents")
        ax.set_title("Longest slow-streak per agent, by outcome")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out / "slow_streak_histogram.png", dpi=120)
        plt.close(fig)
        print(f"Histogram written to {args.out}/slow_streak_histogram.png")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
