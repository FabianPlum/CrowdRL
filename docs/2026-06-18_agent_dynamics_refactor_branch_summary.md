# agent_dynamics_refactor — Branch Summary & Avenue Map

Branch: `agent_dynamics_refactor` (HEAD = `42f5f93`)
Date: 2026-06-18
Scope: retrospective over the 65 commits between `master` (`b981ad0`) and HEAD.

## TL;DR

The whole branch is one mission: **make agents that already reach their
goals move like actual humans instead of ice-skating.** That was pursued
down four parallel tracks — richer *observations* (memory), more physical
*actions* (dynamics / speed-turn coupling), better *reward* shaping, and
finally *numerical stabilization* once the aggressive configs started to
diverge.

Where it landed: a converged recipe of **A+ observation stack + navmesh-only
navigation + speed-turn coupling + hard/soft wall penalties + no
stuck-termination**, all running on the **tanh-stabilized trainer**
(`42f5f93`). The headline architectural bet from the founding plan —
**Layer 2 acceleration-based dynamics — was never built**; human-shaped,
stable motion was reached through Layer-1 recalibration, the speed-turn
clamp, and the tanh / GAE numerical fix instead.

A defining property of the branch: almost every feature landed **behind a
config flag** (the "ablation-friendly" principle), so very little code was
hard-deleted. "Dead end" below mostly means *the experiment's verdict was
negative / the config is never reused*, not *the code was reverted*.

## Methodology note

Tuning wins were only trusted against a **>=2-seed noise floor**
(|ΔGoalRate| > 0.055, |ΔReward| > 1.43, established from two baseline
seeds) using `scripts/analyze_run.py` (per-tier Wilson-CI comparison) and
`scripts/diagnose_stuck_agents.py` (per-agent failure classification).
Production throughput is ~180–193K agent-steps/s on 2x RTX 4090 (DDP).

## Avenue map — success vs dead end

| Avenue | What we tried | Verdict | Evidence |
|---|---|---|---|
| Ego temporal memory (A) | +6D trajectory scalars (82→88D) | KEPT (prereq) | Slight *training* regression alone (−0.036 GR), matched/beat in eval — root cause was the memory x stuck-term coupling, not the feature |
| Neighbor memory A+ | +persistent-ID matcher + neighbor velocity history (→~104D) | KEPT — winning obs stack | +1.23 reward vs baseline overall, **+3.5 on T2/T3B** (the social tiers); +2.99 vs plain-A |
| Neighbor memory A++ | +neighbor trajectory features (→128D) | PARKED | 20:1 first-layer input compression risk; revisit only with a wider first layer |
| Action-envelope refactor | asymmetric fwd/back speed, explicit velocity clamp, `preferred_speed` in ego obs, rename `velocity_damping`→`desired_velocity_weight` | KEPT (permanent) | Present in every current config |
| Layer 1 reward retune | speed_dev −20x, jerk −10x, loosen caps 1.146/0.573/1.719 → 4/2/4 deg/step | KEPT (params) / MIXED (run) | Broke ice-skating "on paper" (brake-before-wall −1.3 → +12); but the dedicated `exp_layer1_seed43` run decayed to ~0.50 GR and left **collision-dominated success** as the residual |
| Layer 2 second-order dynamics | reinterpret 4D action as *acceleration*, speed-coupled yaw, angular-velocity state | NEVER BUILT (parked) | The branch's founding bet — design-of-record only, 0 checkpoints, parked on a separate local branch |
| Speed-turn coupling | lateral-accel yaw envelope (slow down to turn sharp) | KEPT | The one piece of Layer 2's *spirit* that shipped, as a Layer-1 clamp; on in the frontier |
| Stuck-term + progress_weight 1.5 | terminate frozen agents + stronger progress gradient | SUCCESS (decisive, replicated) | Stuck pop **10.8% → 0.0%** (s42) / 0.3% (s43); +0.042 (s42) / +0.089 (s43) GR, both outside noise. Synergistic, not additive |
| progress_weight 1.5 alone | stronger progress gradient | PARTIAL (folded in) | TIER_1 +0.071 GR, ~8% more episodes; T2 reward −2.42 |
| max_steps 2000 → 2500 | give episodes more time | DEAD END | −0.079 GR, −10.56 reward — "max_steps is not the bottleneck" |
| Softer proximity penalty | −0.01/1.0m → −0.005/0.7m | DEAD END (reverted) | −0.10…−0.14 GR — "the magnitude *is* the spacing signal" |
| Tier-weight shift → T3B | more T3B curriculum sampling | DEAD END (neutral) | T3B GR unchanged — not a sample-count problem |
| Per-component reward logging + CPU renders | instrument every reward term; opt-in training-time video | KEPT (diagnostic infra) | Enabled diagnosing every later collapse |
| Collision-penalty "moderate" | retune avoidance weights (smooth on/off) | SUPERSEDED | Overtaken by the explicit wall-contact mechanism |
| Wall-contact penalties | hard `wall_collision` + soft `wall_proximity` band | KEPT | Both in the frontier config |
| Low-entropy / entropy-only | entropy_coef ↓ for a sharper policy | DEAD END (destabilizing) | Collision penalty exploded to −351 within 25 rollouts |
| Single next-waypoint nav + path-aware reward/stuck | navmesh waypoint replaces global bearing; progress measured along the funnel path | KEPT (frontier pillar) | Enables `use_goal_direction: false`; matches the JuPedSim deployment mode |
| nogoaldir ablation | cut the global-goal bearing entirely | FRONTIER (was NaN-prone, now stabilized) | Every pre-stabilization nogoaldir run NaN'd; this is what the live run validates |
| Numerical guardrails → tanh stabilization | NaN-grad skip + log_std clamp → **tanh policy, truncation-aware GAE, robust cursor, NaN containment** | SUCCESS (capstone) | `42f5f93`; 449 tests; the fix the live run is proving out |

## The three things that actually mattered

1. **Diagnosis beat tuning.** The biggest win (`stuck_term + pw1.5`) only
   happened because per-agent instrumentation revealed 10.8% of TIER_3B
   agents were *silently frozen* — invisible in aggregate goal-rate.
   Several "dead ends" (max_steps, tier-weights) were attempts to fix that
   symptom from the wrong end.

2. **An observation x environment-design trap.** Every memory variant
   *regressed goal-rate in training* — not because the features were bad,
   but because giving the policy a `path_efficiency` signal *while also
   terminating low-path-efficiency agents* taught it to self-diagnose as
   stuck and accept the −10 timeout penalty instead of escaping. The fix
   was not the features; it was **`stuck_termination_enabled: false`** —
   exactly what the frontier config does. A+ + no-stuck-term gave the best
   stable reward in the whole sweep (−6.5, GR 0.95).

3. **`nogoaldir` is powerful but brittle.** Removing the goal-direction
   obs (navmesh-only navigation, the JuPedSim-deployable mode) is the
   current research bet, but without it the policy crowds under the
   collision / wall penalties and the value loss → NaN cascade kicks in.
   That single failure mode — `coll_ag` blow-up (−300…−580/agent) → NaN —
   consumed the entire `sharpturn` / `bigrooms` log graveyard and is the
   direct reason the tanh / GAE stabilization commit exists.

## The keepers (the converged recipe)

- **Observation**: A+ — ego temporal memory (`use_temporal_memory`) +
  persistent-neighbor matcher (`use_neighbor_memory`) + neighbor velocity
  history (`use_neighbor_vel_history`), ~104D. A++ trajectory features off.
- **Navigation**: navmesh-only — `use_goal_direction: false`, single
  next-waypoint direction + path-deviation scalar, path-aware reward and
  stuck detection (commit `fa18862`).
- **Action**: asymmetric forward/backward speed + explicit velocity clamp;
  speed-turn coupling on (`turn_lat_accel`, `turn_pivot_rate_deg`);
  smoothness off.
- **Reward**: moderate collision penalty, hard + soft wall-contact
  penalties, existence penalty retained.
- **Episode**: `stuck_termination_enabled: false`.
- **Trainer**: tanh-squashed Gaussian policy (stable change-of-variables
  log-prob; ONNX export uses `tanh(mean)` to keep train == deploy),
  truncation-aware GAE (bootstrap V on timeout/stuck, zero only on true
  terminal), robust waypoint cursor, NaN containment. log_std clamp +
  NaN-grad skip retained as backstops.

## Dead ends (do not re-try without new evidence)

- `max_steps` increase, softer proximity penalty, TIER_3B curriculum
  reweighting — all measured regressions or neutral.
- Reducing `entropy_coef` aggressively on the nogoaldir line — destabilizes
  (collision-penalty blow-up).
- Layer-1 dynamics as a *training run* — the parameter changes are kept,
  but the dedicated run plateaued well below baseline.

## Parked / deferred

- **Layer 2 second-order (acceleration) dynamics** — design-of-record
  (`plan/agent_dynamics_refactor.md`), gated behind a `dynamics_mode`
  flag, conditional on whether the kinematic policy still collides head-on.
  Preserved on a separate local branch.
- **A++ neighbor trajectory features** — needs a wider first hidden layer
  and the stuck-term coupling resolved first.
- GRU recurrence and spatial visit counts — only if specific pathologies
  (long-memory tasks, circling) appear. Ruled out outright: occupancy
  grids, ICM/RND curiosity, full frame-stacking, transformers (until
  Tier 4–5).

## Where it landed — the live run

`configs/exp_nogoaldir_stable_bigrooms_Aplus.yaml` (untracked) is the
frontier config: A+ obs, navmesh-only, from scratch (memory changes
obs_dim, so no warm-start), both GPUs. As of this writing it has cleared
the full curriculum and is stable in `full` phase at GoalRate ~0.95,
reward ~−7, `action_std` flat at ~0.42 — i.e. `coll_ag` is being driven
*down* over training rather than blowing up, the opposite of every prior
nogoaldir run. This is the first nogoaldir run on the stabilized trainer
and is the validation of `42f5f93`. The `full`-phase `coll_ag` magnitude
(elevated with ~30 agents) remains the component to watch as the NaN
precursor.

## What lives where

| Artifact | Path |
|---|---|
| Frontier config | `configs/exp_nogoaldir_stable_bigrooms_Aplus.yaml` |
| Live run output | `results_exp_nogoaldir_stable_bigrooms_Aplus/`, log `train_nogoaldir_bigrooms_Aplus.log` |
| Founding design plan (Layer 1/2) | `plan/agent_dynamics_refactor.md` |
| Memory research + A/A+/A++ results | `plan/agent_memory_research.md`, `plan/neighbor_memory_extension.md` |
| Layer 1 v2 handover | `docs/2026-05-26_layer1_v2_handover.md` |
| Stuck-agent tuning sweep | `docs/2026-04-11_stuck_agent_tuning_summary.md` |
| Per-run hypotheses/verdicts | `results_*/notes.md` |
| Noise-floor + stuck diagnostics | `scripts/analyze_run.py`, `scripts/diagnose_stuck_agents.py` |
| Stabilization tests | `test_collector_gae`, `test_cursor`, `test_nan_robustness`, `test_navmesh_parity` |
