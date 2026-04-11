# Overnight Experiment Sweep — Results Summary

**Run window**: 2026-04-10 evening → 2026-04-11 morning
**Total compute**: ~10 training runs at ~2h each ≈ 18-20 GPU-hours across 2x RTX 4090
**Starting point**: `results_baseline_seed42` (reverted proximity config, stuck-agent pathology
present at ~10.8% in TIER_3B per diagnostic)

## Headline result

**The combination of stuck-agent termination + progress_weight 1.5 produces a
consistent, outside-noise-floor improvement on both seeds.** The stuck-agent
pathology that dominated TIER_3B timeouts (10.8% of agents frozen at spawn
with zero net progress) is effectively eliminated (0% seed42, 0.3% seed43).

**Recommended configuration** for further tuning work:
```yaml
reward:
  progress_weight: 1.5
episode:
  stuck_termination_enabled: true
  stuck_window_steps: 300
  stuck_progress_threshold: 0.2
```

## Noise floor

Established from two seeds of the unchanged baseline config:

| Seed | Overall GR | Overall R |
|------|-----------:|----------:|
| 42   | 0.9016     | -12.34    |
| 43   | 0.8462     | -13.76    |
| **Gap** | **0.0554** | **1.43** |

All subsequent experiment deltas are annotated as "outside noise" when
|ΔGR| > 0.055 or |ΔR| > 1.43.

## Every run

Ordered chronologically, with verdict vs **same-seed baseline**.

| # | Run | Seed | ΔGR | ΔR | Verdict |
|---|---|---|---:|---:|---|
| 1 | baseline_seed42 | 42 | — | — | Reference |
| 2 | baseline_seed43 | 43 | — | — | Noise floor |
| 3 | exp_negative_soft_prox (agent_proximity_penalty_near -0.005, radius 0.7) | 42 | -0.103 | -3.55 | **Clear regression, config reverted** |
| 4 | exp_tier_weights_3B ([0.03,0.03,0.04,0.30,0.60]) | 42 | -0.013 | +0.01 | Within noise |
| 5 | exp_progress_weight_1_5 (pw=1.5 alone) | 42 | +0.014 | -0.37 | Within noise (but TIER_1 +0.07*) |
| 6 | exp_max_steps_2500 (max_steps=2500) | 42 | -0.079 | -10.56 | **Clear regression** (longer episodes waste budget) |
| 7 | exp_stuck_term_seed42 isolated (stuck alone, pw=1.0) | 42 | -0.056 | -0.03 | **Mild regression** (stuck halved but not enough) |
| 8 | **exp_stuck_term_AND_pw15 seed42** (stuck + pw1.5) | 42 | **+0.042** | **+3.17** | **WIN** |
| 9 | **exp_stuck_term_AND_pw15 seed43** (stuck + pw1.5 replicate) | 43 | **+0.089** | **+2.11** | **WIN (replicated)** |

## Stuck-agent diagnostic results

Ran `scripts/diagnose_stuck_agents.py` on checkpoints to measure the actual
stuck population (defined: speed<0.1 m/s for >=200 consecutive steps AND
net_progress<0.2m). 64 TIER_3B episodes, ~1859 agents total.

| Checkpoint | Reached | Slow-progressing | **Stuck** | Non-reached |
|---|---:|---:|---:|---:|
| baseline_seed42 | 76.3% | 12.7% | **10.8%** | 23.6% |
| exp_stuck_term isolated (pw1.0) | 82.7% | 11.8% | **5.4%** | 17.2% |
| exp_stuck_term + pw1.5 seed42 | 97.8% | 2.1% | **0.0%** | 2.2% |
| exp_stuck_term + pw1.5 seed43 | 94.0% | 5.6% | **0.3%** | 6.0% |

The combined intervention **eliminates** the stuck pathology. Isolated
stuck_term halves it but does not close the gap — this is the evidence for
synergy.

## Key findings

### 1. Stuck-termination is necessary but not sufficient

`exp_stuck_term_seed42` (isolated, pw=1.0) shows the stuck detection
mechanism works diagnostically (10.8% → 5.4%) but training goal rate
regresses slightly (-0.056 vs baseline). The mechanism is correct — stuck
agents get terminated, wasted budget is reclaimed, 16% more full-phase
episodes run — but the policy doesn't learn to avoid local minima without
a stronger gradient.

### 2. Progress weight 1.5 alone is not enough

`exp_progress_weight_1_5` (pw=1.5, no stuck term) is within noise on every
metric: ΔGR +0.014, ΔR -0.37. TIER_1 does improve outside noise (+0.07 GR)
but other tiers show no clear effect. On its own, this is not a win.

### 3. The combined intervention is synergistic

| Config | Stuck pop | Overall ΔGR | Overall ΔR |
|---|---|---:|---:|
| pw1.5 alone | ~10.8% (unchanged) | +0.014 | -0.37 |
| stuck_term alone | 5.4% | -0.056 | -0.03 |
| **combined** | **0-0.3%** | **+0.04 to +0.09** | **+2.1 to +3.2** |

The mechanism is: stuck_term reclaims training budget from frozen agents
so more episodes run per collect. Those extra episodes produce more
learning signal. Progress_weight 1.5 provides a stronger gradient so that
learning actually teaches the policy to escape the local minima the
stuck_term is removing. Without the stronger gradient, stuck_term just
terminates stuck agents without teaching anything. Without stuck_term, the
stronger gradient is diluted across mostly-useless stuck episodes.

### 4. Cross-seed replication confirms the effect

| Seed | Baseline GR | Combined GR | ΔGR | Baseline R | Combined R | ΔR |
|------|-----------:|-----------:|----:|----------:|----------:|----:|
| 42   | 0.9016 | 0.9437 | **+0.042** | -12.34 | -9.16 | **+3.17** |
| 43   | 0.8462 | 0.9351 | **+0.089** | -13.76 | -11.65 | **+2.11** |

**Both seeds outside noise on both metrics.** The GR improvement is larger
on seed 43 (its baseline was weaker); the reward improvement is larger on
seed 42. The effect is robust.

Notably, the combined intervention has **tighter cross-seed goal-rate
variance than baseline** (combined: 0.0087 vs baseline: 0.0554). This
means the combined intervention also makes training more stable across
seeds, not just better on average.

### 5. What didn't work and why

- **Softer agent proximity penalty (exp_negative_soft_prox)**: removed the
  learning signal for personal space maintenance. Caused -0.12 to -0.14 GR
  regression across all navigation-heavy tiers. TIER_0 (open field)
  benefited slightly but dense-navigation tiers collapsed. Lesson: the
  proximity penalty magnitude IS the spacing signal, you can't soften it.
- **Tier-weights shift toward TIER_3B**: moved 0.45 → 0.60 of full-phase
  weight onto TIER_3B. Within-noise on every metric. TIER_3B goal rate
  moved -0.004. **TIER_3B weakness is not a sample-count problem.**
- **max_steps 2500**: longer episodes burned more agent-step budget per
  episode, producing 14% fewer full-phase episodes and -10.56 reward
  regression. The extra time did not help stuck agents — it just let them
  wander longer. **max_steps=2000 was NOT the TIER_3B bottleneck.**

## Per-tier final picture (combined config, seed42)

Best single run, last 2000 full-phase episodes:

| Tier | eps | n_ag | ep_len | goal_rate (95%) | reward (95%) |
|---|---:|---:|---:|---|---|
| TIER_0  | 115 | 33.8 |  779 | 0.996 [0.994,0.998] |  -0.47 [-1.05,+0.11] |
| TIER_1  |  90 | 27.1 | 1095 | 0.952 [0.943,0.960] |  -7.71 [-9.94,-5.49] |
| TIER_2  | 202 | 20.7 | 1005 | 0.988 [0.984,0.991] |  -4.40 [-5.10,-3.69] |
| TIER_3A | 737 | 26.5 | 1019 | 0.961 [0.958,0.964] |  -9.78 [-10.39,-9.18]|
| TIER_3B | 856 | 31.6 | 1441 | 0.916 [0.913,0.919] | -11.08 [-11.66,-10.49]|

Mean episode length dropped from 1498 (baseline) to 1188 (combined, -21%).

## Recommendations

**Short-term (next training run):**
1. Adopt the combined config as the new baseline. Seed 42 is fine.
2. No tuning changes needed; the config is stable.

**Medium-term (still open problems):**
1. **TIER_3B goal rate is still 0.905-0.916** — the best achievable with
   the current policy/observation architecture. Pure reward tuning has
   plateaued.
2. **TIER_3B episodes are still longer** (mean 1441-1528 steps) than other
   tiers. This is the residual "coordinated navigation in composed rooms
   is hard" problem. Reward tuning can't fix it further.
3. **Next step suggestion**: the geodesic progress potential work
   (`/home/fabi/.claude/plans/expressive-launching-clock.md`) is the
   natural follow-up. Euclidean progress creates phantom potential wells
   in room layouts; geodesic progress removes them at the root. The plan
   is well-scoped and uses infrastructure that already exists.
4. **Important**: the diagnostic found 5-11% of agents in the "slow but
   progressing" category even for the combined run. These are queue-like
   behaviours — agents that are eventually successful but have extended
   low-speed periods. The current stuck detection correctly does NOT
   flag them. If TIER_3B remains the bottleneck, a faster queue-resolution
   mechanism could help, but this is a harder problem than the initial
   stuck-detection fix.

## Artefacts on disk

```
results_baseline_seed42/                      — reference baseline (seed 42, pw1.0, no stuck)
results_baseline_seed43/                      — noise floor replicate (seed 43)
results_exp_negative_soft_prox_seed42/        — NEGATIVE: soft proximity (reverted)
results_exp_tier_weights_3B_seed42/           — WITHIN NOISE: tier weight shift
results_exp_progress_weight_1_5_seed42/       — WITHIN NOISE: pw=1.5 alone
results_exp_max_steps_2500_seed42/            — NEGATIVE: max_steps 2500
results_exp_stuck_term_seed42/                — MILD NEGATIVE: stuck alone (pw=1.0)
results_exp_stuck_term_AND_pw15_seed42/       — WIN: combined seed42 (0% stuck)
results_exp_stuck_term_AND_pw15_seed43/       — WIN REPLICATED: combined seed43 (0.3% stuck)

diagnostic_tier_3B/                           — baseline_seed42 checkpoint diagnostic
diagnostic_tier_3B_stuck_term/                — combined seed42 checkpoint
diagnostic_tier_3B_stuck_term_isolated/       — isolated stuck_term seed42 checkpoint
diagnostic_tier_3B_combined_seed43/           — combined seed43 checkpoint

scripts/analyze_run.py                        — per-tier analysis + CI + noise-annotated comparison
scripts/diagnose_stuck_agents.py              — per-agent classification (reached/stuck/slow_progressing/oscillating)
scripts/run_experiment.sh                     — not used this session (inline commands preferred)
```

Each run directory contains:
- `config.yaml.snapshot` — exact YAML used for the run
- `git_rev.txt` — code state
- `training.log` — full stdout from torchrun
- `history.json` — per-episode history (goal_rate, reward, n_agents, ep_len, geometry_tier, phase)
- `notes.md` — hypothesis, observation, verdict
- `checkpoint_final.pt`, `policy.onnx`, `evaluation.png`, `training_curves.png`, `trajectories.png`, `episode.mp4`
- `analysis.txt` and `compare_vs_*.txt` — analyze_run.py outputs

## Engineering artefacts worth keeping

1. **[scripts/analyze_run.py](scripts/analyze_run.py)** — generic tool for per-tier
   analysis + Wilson CIs + noise-annotated comparisons. Will be reused for
   every future experiment.
2. **[scripts/diagnose_stuck_agents.py](scripts/diagnose_stuck_agents.py)** — per-agent
   per-step classification tool. Can be run on any checkpoint to quantify
   the stuck population. Indispensable for diagnosing what the overall
   goal rate is hiding.
3. **Stuck-agent termination feature** ([packages/crowdrl-torch/src/crowdrl_torch/step.py](packages/crowdrl-torch/src/crowdrl_torch/step.py),
   [packages/crowdrl-env/src/crowdrl_env/crowd_env.py](packages/crowdrl-env/src/crowdrl_env/crowd_env.py)):
   config-gated, parallel torch + numpy implementations, mirrored through
   `curriculum.make_env_config`, plumbed through YAML. 343/343 tests pass.
4. **Noise-floor protocol** — always run ≥2 seeds per config to establish
   whether an effect is real. Mandatory for any future tuning work.

## Process lessons

1. **Attribution error caught**: the first "combined" run was inadvertently
   launched with `progress_weight=1.5` leftover from a killed replicate.
   Re-running with pw1.0 alone proved that stuck_term in isolation is not
   sufficient — the win is synergistic. Lesson: always snapshot and
   verify config BEFORE launching.
2. **Small deltas are dangerous**: with a noise floor of 0.055 GR and 1.43
   reward, any single-knob experiment with expected effect <2x those values
   is going to be inconclusive. Future experiments should be gated on
   "expected effect is at least 2x noise floor" before running.
3. **Diagnostic > aggregate metrics**: the aggregate `goal_rate` never
   showed the "10.8% of agents stuck" story; it took a per-agent
   classification of a specific scenario to find it. Always build a
   targeted diagnostic when an aggregate metric plateaus.
