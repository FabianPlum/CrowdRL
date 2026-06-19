"""Fixed-scenario behavioural scorecard for a trained policy.

Runs a policy through a DETERMINISTIC suite of (tier, agent-count, seed)
scenarios and reports the decomposed behavioural metrics from
:mod:`crowdrl_env.eval_metrics` -- goal rate, agent-collision rate, wall-contact
rate, freeze / deadlock rate, speed and path efficiency -- per scenario and
aggregated.

Why fixed: the training curriculum re-samples geometry tiers and agent counts
every rollout, so the training reward curve mixes optimization progress with
task-distribution variance and two checkpoints are NOT directly comparable. This
suite pins the scenarios, so configs/checkpoints can be compared apples-to-apples.

The two failure modes live on one throughput<->safety Pareto front: "bulldozing"
shows up as a high ``agent_collision_rate``, gridlock as a high ``freeze_rate`` /
``stuck_agent_frac``. A good change raises ``goal_rate`` WITHOUT trading one axis
for the other -- it moves the whole front, rather than sliding along it.

Actions are taken deterministically (policy mean) so the numbers reflect the
policy's intended behaviour, free of sampling noise and reproducible run-to-run.

``run_scorecard_policy`` takes an in-memory policy so it is testable without a
checkpoint; ``scripts/eval_scorecard.py`` is the checkpoint-loading CLI.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from crowdrl_env.crowd_env import CrowdEnv, CrowdEnvConfig
from crowdrl_env.eval_metrics import aggregate_metrics, compute_episode_metrics
from crowdrl_env.geometry_generator import GeometryTier
from crowdrl_env.kinora_export import write_episode_h5
from crowdrl_env.visualiser import collect_episode_frames


@dataclass(frozen=True)
class ScenarioSpec:
    """One fixed eval scenario: a single geometry tier, agent count, and seed."""

    label: str
    tier: GeometryTier
    n_agents: int
    seed: int


# Deterministic suite spanning the behaviours the two failure modes live in:
# open baseline, corridor/bottleneck head-ons (the gridlock probe), branching
# crossings, and dense rooms / composed (the bulldozing probe). Two seeds per
# setting so one unlucky layout does not dominate. Keep this list STABLE --
# changing it breaks comparability with previously recorded scorecards.
DEFAULT_SCENARIOS: list[ScenarioSpec] = [
    # --- Moderate density (<=30 agents): the "clean capability" regime. ---
    ScenarioSpec("open_t0", GeometryTier.TIER_0, 20, 0),
    ScenarioSpec("open_t0", GeometryTier.TIER_0, 20, 1),
    ScenarioSpec("corridor_t1", GeometryTier.TIER_1, 20, 0),
    ScenarioSpec("corridor_t1", GeometryTier.TIER_1, 20, 1),
    ScenarioSpec("branch_t2", GeometryTier.TIER_2, 24, 0),
    ScenarioSpec("branch_t2", GeometryTier.TIER_2, 24, 1),
    ScenarioSpec("rooms_t3a", GeometryTier.TIER_3A, 30, 0),
    ScenarioSpec("rooms_t3a", GeometryTier.TIER_3A, 30, 1),
    ScenarioSpec("composed_t3b", GeometryTier.TIER_3B, 30, 0),
    ScenarioSpec("composed_t3b", GeometryTier.TIER_3B, 30, 1),
    # --- High-density tail (60-100 agents): matches what the training full
    # phase is actually dominated by (agents up to 100, 80% weight on
    # rooms/composed). The moderate scenarios above flatter the policy; these
    # probe the regime where freezing/congestion actually bites, so the
    # scorecard tracks the same difficulty the training GoalRate sees. ---
    ScenarioSpec("corridor_hi", GeometryTier.TIER_1, 40, 0),
    ScenarioSpec("rooms_hi", GeometryTier.TIER_3A, 60, 0),
    ScenarioSpec("rooms_hi", GeometryTier.TIER_3A, 100, 0),
    ScenarioSpec("composed_hi", GeometryTier.TIER_3B, 60, 0),
    ScenarioSpec("composed_hi", GeometryTier.TIER_3B, 100, 0),
]


def _scenario_env(base: CrowdEnvConfig, spec: ScenarioSpec, max_steps: int | None) -> CrowdEnv:
    """Pin ``spec``'s tier + agent count onto the run's own geometry / physics /
    obs / action / reward config, so the eval matches how the run was trained."""
    spawn = dataclasses.replace(base.spawn, n_agents_range=(spec.n_agents, spec.n_agents))
    overrides: dict = {"geometry_tiers": [spec.tier], "tier_weights": None, "spawn": spawn}
    if max_steps is not None:
        overrides["max_steps"] = max_steps
    return CrowdEnv(config=dataclasses.replace(base, **overrides), seed=spec.seed)


def run_scorecard_policy(
    env_config: CrowdEnvConfig,
    actor_critic,
    obs_normalizer=None,
    *,
    device=None,
    scenarios: list[ScenarioSpec] | None = None,
    max_steps: int | None = None,
    freeze_speed: float = 0.1,
    export_h5_dir: Path | None = None,
) -> dict:
    """Run ``actor_critic`` through ``scenarios`` and return a structured scorecard.

    Parameters
    ----------
    env_config : CrowdEnvConfig
        The run's env config; each scenario overrides only its tier + agent count.
    actor_critic, obs_normalizer, device
        In-memory policy + obs normalizer + torch device, as consumed by
        :func:`crowdrl_env.visualiser.collect_episode_frames`. Actions are taken
        deterministically (policy mean).
    scenarios : list[ScenarioSpec] or None
        Defaults to :data:`DEFAULT_SCENARIOS`.
    max_steps : int or None
        Per-episode cap; None uses each env config's ``max_steps``.
    freeze_speed : float
        Threshold forwarded to :func:`compute_episode_metrics`.
    export_h5_dir : Path or None
        If set, also write a Kinora/pedpy HDF5 per scenario into this directory
        (``<label>_a<n_agents>_s<seed>.h5``) for visualisation. Default None leaves
        the scorecard metrics-only (no behaviour change).

    Returns
    -------
    dict
        ``{"per_scenario": [{label, n_agents, seed, metrics}, ...],
           "overall": {metric: mean_across_scenarios}}``.
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    per_scenario: list[dict] = []
    for spec in scenarios:
        env = _scenario_env(env_config, spec, max_steps)
        frames = collect_episode_frames(
            env,
            actor_critic,
            obs_normalizer,
            device,
            max_steps=max_steps if max_steps is not None else env.config.max_steps,
            deterministic=True,
        )
        if export_h5_dir is not None:
            write_episode_h5(
                frames,
                Path(export_h5_dir) / f"{spec.label}_a{spec.n_agents}_s{spec.seed}.h5",
                metadata={
                    "scenario": spec.label,
                    "tier": spec.tier.name,
                    "n_agents": spec.n_agents,
                    "seed": spec.seed,
                },
            )
        metrics = compute_episode_metrics(frames, freeze_speed=freeze_speed)
        per_scenario.append(
            {
                "label": spec.label,
                "n_agents": spec.n_agents,
                "seed": spec.seed,
                "metrics": metrics,
            }
        )

    overall = aggregate_metrics([s["metrics"] for s in per_scenario])
    return {"per_scenario": per_scenario, "overall": overall}


# Columns surfaced in the printed table: throughput, the two safety axes, the two
# gridlock axes, plus rushing / path quality. (key, short header)
_COLUMNS: list[tuple[str, str]] = [
    ("goal_rate", "goal"),
    ("agent_collision_rate", "coll"),
    ("wall_contact_rate", "wall"),
    ("freeze_rate", "freeze"),
    ("stuck_agent_frac", "stuck"),
    ("speed_over_preferred", "spd/pref"),
    ("path_efficiency", "path_eff"),
]


def format_scorecard(scorecard: dict) -> str:
    """Render a scorecard dict (from :func:`run_scorecard_policy`) as a table."""
    header = f"{'scenario':<16}{'seed':>5}{'agents':>7}"
    for _key, short in _COLUMNS:
        header += f"{short:>10}"
    sep = "-" * len(header)
    lines = [header, sep]

    def fmt_row(label: str, seed, agents, metrics: dict) -> str:
        seed_s = f"{seed:>5}" if seed is not None else f"{'--':>5}"
        agents_s = f"{agents:>7}" if agents is not None else f"{'--':>7}"
        row = f"{label:<16}{seed_s}{agents_s}"
        for key, _short in _COLUMNS:
            row += f"{metrics[key]:>10.3f}" if key in metrics else f"{'--':>10}"
        return row

    for s in scorecard["per_scenario"]:
        lines.append(fmt_row(s["label"], s["seed"], s["n_agents"], s["metrics"]))
    lines.append(sep)
    lines.append(fmt_row("OVERALL", None, None, scorecard["overall"]))
    return "\n".join(lines)
