"""Post-run analysis for CrowdRL training experiments.

Produces a per-tier breakdown of the final window of full-phase episodes
with Wilson 95% CIs on goal rate and Student-t CIs on mean reward.

Usage
-----
    uv run python scripts/analyze_run.py results_baseline_v3_seed42
    uv run python scripts/analyze_run.py results_baseline_v3_seed42 \
                                         --compare results_exp_B_seed42
    uv run python scripts/analyze_run.py results_baseline_v3_seed42 \
                                         --last 2000

Writes a summary to stdout and to ``<run_dir>/analysis.txt``.
When ``--compare`` is given, also writes a delta table annotated with
whether each effect is outside the noise floor.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TIERS = ["TIER_0", "TIER_1", "TIER_2", "TIER_3A", "TIER_3B"]
FULL_PHASE_IDX = 5

# Reward components assumed to be shared across runs. Used only for the
# penalty-rate decomposition, not for decision-making.
GOAL_BONUS = 20.0
EXISTENCE_PENALTY = -0.01


@dataclass
class TierStats:
    tier: str
    n_episodes: int
    mean_ag: float
    mean_len: float
    goal_rate: float
    goal_rate_ci: tuple[float, float]
    mean_reward: float
    reward_sem: float
    reward_ci: tuple[float, float]
    penalty_rate: float

    def goal_rate_str(self) -> str:
        lo, hi = self.goal_rate_ci
        return f"{self.goal_rate:.3f} [{lo:.3f},{hi:.3f}]"

    def reward_str(self) -> str:
        lo, hi = self.reward_ci
        return f"{self.mean_reward:+.2f} [{lo:+.2f},{hi:+.2f}]"


def wilson_ci(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion. Robust for small n and p near 0/1."""
    if n <= 0:
        return (0.0, 0.0)
    p_hat = successes / n
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    half_width = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4 * n)) / n)
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def t_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (mean, SEM, t-95% half-width) for an array of samples."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = float(values.mean())
    if n == 1:
        return (mean, 0.0, 0.0)
    std = float(values.std(ddof=1))
    sem = std / math.sqrt(n)
    # Normal approx; fine for n>=30 which is what we'll have.
    half = 1.96 * sem
    return (mean, sem, half)


def load_history(run_dir: Path) -> dict:
    path = run_dir / "history.json"
    if not path.exists():
        raise FileNotFoundError(f"No history.json in {run_dir}")
    return json.loads(path.read_text())


def tier_stats_for_window(
    history: dict,
    last_n: int,
) -> dict[str, TierStats]:
    """Compute per-tier stats over the last ``last_n`` full-phase episodes."""
    phase_idx = np.asarray(history["phase_idx"])
    n_ag = np.asarray(history["n_agents"])
    ep_len = np.asarray(history["episode_length"])
    gr = np.asarray(history["goal_rate"])
    mean_rew = np.asarray(history["mean_reward"])
    tiers = np.asarray(history.get("geometry_tier", ["unknown"] * len(phase_idx)))

    full_mask = phase_idx == FULL_PHASE_IDX
    idx_all = np.where(full_mask)[0]
    if len(idx_all) == 0:
        raise RuntimeError("No full-phase episodes recorded — analysis window empty")

    # Take last N full-phase episodes (chronological order within the history).
    idx = idx_all[-last_n:] if last_n > 0 else idx_all

    out: dict[str, TierStats] = {}
    for tier in TIERS:
        m = tiers[idx] == tier
        if not m.any():
            continue
        sel = idx[m]
        n = int(len(sel))
        ag = float(n_ag[sel].mean())
        el = float(ep_len[sel].mean())
        # Agent-weighted goal rate: "what fraction of all agents reached
        # their goal". Consistent with the Wilson CI denominator which is
        # total agent count across the window.
        total_agents = int(n_ag[sel].sum())
        total_reached = int(round((gr[sel] * n_ag[sel]).sum()))
        goal_point = total_reached / max(total_agents, 1)
        goal_ci = wilson_ci(total_reached, total_agents)
        r_mean, r_sem, r_half = t_ci(mean_rew[sel])
        r_ci = (r_mean - r_half, r_mean + r_half)
        # Penalty rate = everything after goal bonus + existence penalty
        # (see analysis from rollout 340 earlier).
        rew_ex_goal = r_mean - goal_point * GOAL_BONUS
        penalty_rate = (rew_ex_goal + (-EXISTENCE_PENALTY) * el) / max(el, 1.0)
        out[tier] = TierStats(
            tier=tier,
            n_episodes=n,
            mean_ag=ag,
            mean_len=el,
            goal_rate=float(goal_point),
            goal_rate_ci=goal_ci,
            mean_reward=r_mean,
            reward_sem=r_sem,
            reward_ci=r_ci,
            penalty_rate=penalty_rate,
        )
    return out


def overall_window_stats(history: dict, last_n: int) -> dict:
    """Aggregate full-phase metrics over the last ``last_n`` episodes."""
    phase_idx = np.asarray(history["phase_idx"])
    gr = np.asarray(history["goal_rate"])
    mean_rew = np.asarray(history["mean_reward"])
    n_ag = np.asarray(history["n_agents"])
    ep_len = np.asarray(history["episode_length"])

    idx_all = np.where(phase_idx == FULL_PHASE_IDX)[0]
    idx = idx_all[-last_n:] if last_n > 0 else idx_all
    if len(idx) == 0:
        return {}

    total_agents = int(n_ag[idx].sum())
    total_reached = int(round((gr[idx] * n_ag[idx]).sum()))
    goal_point = total_reached / max(total_agents, 1)
    goal_ci = wilson_ci(total_reached, total_agents)
    r_mean, r_sem, r_half = t_ci(mean_rew[idx])
    return {
        "n_episodes": int(len(idx)),
        "mean_ag": float(n_ag[idx].mean()),
        "mean_len": float(ep_len[idx].mean()),
        "goal_rate": float(goal_point),
        "goal_rate_ci": goal_ci,
        "mean_reward": r_mean,
        "reward_sem": r_sem,
        "reward_ci": (r_mean - r_half, r_mean + r_half),
    }


def format_single_run(run_dir: Path, last_n: int) -> str:
    history = load_history(run_dir)
    overall = overall_window_stats(history, last_n)
    per_tier = tier_stats_for_window(history, last_n)

    lines: list[str] = []
    lines.append(f"=== {run_dir.name} ===")
    lines.append(f"Analysis window: last {last_n} full-phase episodes (n={overall['n_episodes']})")
    lines.append("")
    lines.append("Overall full phase:")
    lines.append(f"  mean_ag  = {overall['mean_ag']:.1f}")
    lines.append(f"  mean_len = {overall['mean_len']:.0f}")
    lo, hi = overall["goal_rate_ci"]
    lines.append(f"  goal_rate = {overall['goal_rate']:.4f} [{lo:.4f}, {hi:.4f}] (Wilson 95%)")
    r_lo, r_hi = overall["reward_ci"]
    lines.append(
        f"  reward   = {overall['mean_reward']:+.3f} "
        f"[{r_lo:+.3f}, {r_hi:+.3f}] "
        f"(SEM {overall['reward_sem']:.3f})"
    )
    lines.append("")
    lines.append("Per-tier breakdown:")
    lines.append(
        f"{'tier':<8} {'eps':>6} {'mean_ag':>8} {'ep_len':>8} "
        f"{'goal_rate (95%)':>25} {'reward (95%)':>28} {'pen_rate':>10}"
    )
    for tier in TIERS:
        if tier not in per_tier:
            lines.append(f"{tier:<8} (no episodes)")
            continue
        s = per_tier[tier]
        lines.append(
            f"{tier:<8} {s.n_episodes:>6d} {s.mean_ag:>8.1f} {s.mean_len:>8.0f} "
            f"{s.goal_rate_str():>25} {s.reward_str():>28} {s.penalty_rate:>+10.5f}"
        )
    lines.append("")
    lines.append("penalty_rate = (reward - goal_bonus * goal_rate + existence * ep_len) / ep_len")
    lines.append(
        "               i.e. mean per-step cost from collision + proximity + wall + action_rate"
    )
    return "\n".join(lines)


def format_comparison(
    run_a: Path,
    run_b: Path,
    last_n: int,
    noise_floor_reward: float | None = None,
    noise_floor_goal: float | None = None,
) -> str:
    """Side-by-side comparison of two runs with per-tier deltas."""
    hist_a = load_history(run_a)
    hist_b = load_history(run_b)
    tiers_a = tier_stats_for_window(hist_a, last_n)
    tiers_b = tier_stats_for_window(hist_b, last_n)
    overall_a = overall_window_stats(hist_a, last_n)
    overall_b = overall_window_stats(hist_b, last_n)

    lines: list[str] = []
    lines.append(f"=== Comparison: {run_a.name}  vs  {run_b.name} ===")
    lines.append(f"Window: last {last_n} full-phase episodes per run")
    if noise_floor_reward is not None or noise_floor_goal is not None:
        parts = []
        if noise_floor_reward is not None:
            parts.append(f"reward_noise={noise_floor_reward:+.3f}")
        if noise_floor_goal is not None:
            parts.append(f"goal_noise={noise_floor_goal:.4f}")
        lines.append(f"Noise floor (replicate gap): {', '.join(parts)}")
    lines.append("")

    # Overall deltas
    d_gr = overall_b["goal_rate"] - overall_a["goal_rate"]
    d_rw = overall_b["mean_reward"] - overall_a["mean_reward"]
    gr_mark = _mark(d_gr, noise_floor_goal)
    rw_mark = _mark(d_rw, noise_floor_reward)
    lines.append("Overall full phase:")
    lines.append(
        f"  goal_rate:   {overall_a['goal_rate']:.4f} -> "
        f"{overall_b['goal_rate']:.4f}  (delta {d_gr:+.4f}) {gr_mark}"
    )
    lines.append(
        f"  mean_reward: {overall_a['mean_reward']:+.3f} -> "
        f"{overall_b['mean_reward']:+.3f}  (delta {d_rw:+.3f}) {rw_mark}"
    )
    lines.append("")

    lines.append("Per-tier deltas (B - A):")
    lines.append(f"{'tier':<8} {'d_gr':>12} {'d_reward':>14} {'d_penrate':>14} {'d_eplen':>10}")
    for tier in TIERS:
        if tier not in tiers_a or tier not in tiers_b:
            continue
        a = tiers_a[tier]
        b = tiers_b[tier]
        d_gr_t = b.goal_rate - a.goal_rate
        d_rw_t = b.mean_reward - a.mean_reward
        d_pen = b.penalty_rate - a.penalty_rate
        d_len = b.mean_len - a.mean_len
        gr_m = _mark(d_gr_t, noise_floor_goal)
        rw_m = _mark(d_rw_t, noise_floor_reward)
        lines.append(
            f"{tier:<8} {d_gr_t:>+11.4f}{gr_m:>1} "
            f"{d_rw_t:>+13.3f}{rw_m:>1} "
            f"{d_pen:>+13.5f}  {d_len:>+10.0f}"
        )
    lines.append("")
    lines.append("Legend: * = delta exceeds the noise floor")
    return "\n".join(lines)


def _mark(delta: float, noise: float | None) -> str:
    if noise is None or noise <= 0:
        return " "
    return "*" if abs(delta) > noise else " "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="Results directory for the run to analyse")
    ap.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional second run to compare against (A=run, B=compare)",
    )
    ap.add_argument(
        "--last",
        type=int,
        default=2000,
        help="Analysis window: last N full-phase episodes (default 2000)",
    )
    ap.add_argument(
        "--noise-reward",
        type=float,
        default=None,
        help="Reward noise floor from baseline replicate gap",
    )
    ap.add_argument(
        "--noise-goal",
        type=float,
        default=None,
        help="Goal-rate noise floor from baseline replicate gap",
    )
    args = ap.parse_args()

    if not args.run.is_dir():
        print(f"error: {args.run} is not a directory", file=sys.stderr)
        return 2

    single = format_single_run(args.run, args.last)
    print(single)
    (args.run / "analysis.txt").write_text(single + "\n")

    if args.compare is not None:
        if not args.compare.is_dir():
            print(f"error: {args.compare} is not a directory", file=sys.stderr)
            return 2
        cmp_text = format_comparison(
            args.run,
            args.compare,
            args.last,
            noise_floor_reward=args.noise_reward,
            noise_floor_goal=args.noise_goal,
        )
        print()
        print(cmp_text)
        (args.compare / f"compare_vs_{args.run.name}.txt").write_text(cmp_text + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
