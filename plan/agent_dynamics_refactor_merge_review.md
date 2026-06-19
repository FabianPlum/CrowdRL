# agent_dynamics_refactor -- Merge Review

Deep review of the `agent_dynamics_refactor` branch ahead of merging to `master`.
Scope: 70 commits, +16.3k / -1.1k lines across the dual numpy (`crowdrl-env`,
eval) and batched-torch (`crowdrl-torch`, training) implementations.

The guiding question: **which changes fixed deep underlying bugs vs. which were
performance tuning vs. which are new capabilities.** Three buckets below.

---

## 1. Executive summary / merge verdict

The branch is built on two confirmed deep fixes (a pair of *twin* normalizer
count-overflow bugs), a set of genuine RL-correctness fixes, a layer of reward
tuning, and a stack of new observation/dynamics features -- all dual-implemented
and parity-tested. Test count grew 343 -> 482 (+139), lint clean.

**The headline:** every full-speed DDP run died deterministically at ~r355.
That was NOT reward, dynamics, policy, or world generation (all chased and ruled
out as red herrings). It was a **float64 overflow of a running-normalizer sample
count**, and there were **two** of them -- one in the observation normalizer
(r355) and its twin in the reward/return normalizer (r360). Both are fixed.

Merge-readiness: **gated on the verify run** confirming the reward-normalizer
fix (481011c) clears the r360 death zone with the same warm-start/config that
previously died. Test/lint surface is ready now.

---

## 2. The two NaN root causes (the real blockers)

Both normalizers are DDP-synced **every rollout**, and both syncs do
`all_reduce(count, SUM)` on the *already-merged* total -- so each rank re-sums
the full cross-rank count every rollout and it grows **geometrically**
(~doubles/rollout), crossing ~1e305 within a few hundred rollouts. Then
`m_a = var * count` overflows to `inf` -> `var = NaN` -> every normalized value
NaN -> dead/frozen policy. The count is a *clock ticking to overflow*, which is
exactly why every config died at the same rollout regardless of its settings.

| | Obs normalizer (r355) | Reward/return normalizer (r360) |
|---|---|---|
| Class | `TorchRunningNormalizer` (GPU) | numpy `RunningNormalizer` inside `RewardNormalizer` |
| Sync | `sync_across_ranks()` | `sync_reward_normalizer()` (distributed.py) |
| Symptom | all-89 obs NaN -> policy dead, GoalRate crashes | raw reward FINITE, but normalized `Reward`+`value_loss` NaN, actor frozen (`approx_kl 0.0000` = NaN-grad-skip), GoalRate coasts on stale weights |
| Fix | `_MAX_COUNT=1e8` cap (commit 624d8e3) | same cap mirrored to the numpy normalizer + its sync writeback + non-finite-row drop (commit 481011c) |

The second was masked by the first: capping the obs count let the run survive to
r360, where the *identical, uncapped* twin in the reward path took over. The
memory note had wrongly recorded the numpy normalizer as "never synced / never
overflowed" -- it IS synced via `sync_reward_normalizer`.

How they were found: a gated tripwire (`CROWDRL_NAN_TRIPWIRE=1`) localized r355
to the obs normalizer count; r360 was localized from the log signature alone
(finite raw reward + NaN `value_loss`/normalized reward + frozen actor) plus a
code trace of the reward-normalization path.

---

## 3. Fix categorization

### A. REAL FIXES -- deep bugs that would bite any run

| Change | Commit | What it fixed |
|---|---|---|
| Obs-normalizer count overflow cap | 624d8e3 | The r355 collapse (above). |
| Reward/return-normalizer count overflow cap (twin) | 481011c | The r360 collapse (above). |
| tanh-squashed Gaussian policy | 42f5f93 | Removes the log_std/entropy runaway at the action boundary caused by hard-clamping raw actions. The log_std clamp is a retained backstop, not the fix. |
| Truncation-aware GAE | 42f5f93 | Bootstrap V(s) on timeout/stuck *truncation*, zero only on true terminal. Old code biased returns -- a standard MAPPO correctness bug. |
| Robust waypoint cursor + `terminated=newly_done` | 42f5f93 | A stuck cursor caused false negative-progress reward and false stuck-termination. |
| Path-aware progress/stuck (remaining navmesh path) | fa18862 | Straight-line distance falsely penalized route-following / falsely flagged stuck on bends. |
| YAML speed caps trapped at dataclass defaults | cf98da5 | `max_forward/backward_speed` were silently pinned to defaults -- the dual-impl allowlist plumbing trap. |
| "Feed current speeds" into both steps | 1e94e68, 0ad874e | Without it the speed-turn-coupling envelope never binds (load-bearing for that feature). |
| Neighbor-history numpy broadcast bug | 991c4a3 | Wrong-axis mask zeroing mixed an evicted neighbor's stale velocity history into a new slot. |
| nav-signal train/deploy parity | fa18862 | Torch blended bearing vs numpy single-waypoint -> obs mismatch training vs JuPedSim. |
| ONNX export `tanh(mean)` | export.py | Deployed deterministic action now matches training. (No test -- see gaps.) |
| Velocity-weighted penalty NaN hardening | 8e0104f | The earlier r855 pileup: unguarded pre-contact velocity snapshot -> NaN reward. (A *different* failure from r360.) |

### B. PERFORMANCE TUNING -- design choices that made it work well (not bug fixes)

- **`collision_penalty_cap = -2.0`** (a0fd05a) -- the v4 "fix" is tuning: it
  floors a penalty the velocity-weighting *feature* introduced. Velocity
  weighting is now discount-only (can cheapen slow contact, can't amplify fast).
- The density reward arc: `progress_weight`, `entropy_coef 0.003`,
  `personal_space_radius 0.75`, wall penalties, `turn_pivot_rate_deg 240`.
- Layer-1 retunes (63db6e7, 9ab84a5): smoothness weights x100, action caps into
  the human envelope. The `desired_velocity_weight` rename (66eb79f) is
  behavior-preserving; the action-envelope re-parameterization (540ccb0) leaves
  the velocity clamp value unchanged (no bug fixed) but adds backward speed.

### C. NEW FEATURES -- capabilities (orthogonal to stability; all parity-tested)

- Temporal memory (Option A, +6D) -- **ON** in the baseline.
- Neighbor memory (persistent IDs + vel history +16D + trajectory +24D) --
  **OFF** in the baseline (A+/A++ deferred, see lineage).
- `preferred_speed` ego obs (+1D, ego 7->8). (No dedicated parity test.)
- Impact-speed-weighted collision/proximity penalties (8102d70) -- ON, capped.
- Hard wall-contact penalty (337c449) -- fixes the "wall = free brake" exploit.
- Speed-turn coupling (888e416...) -- ON (`turn_pivot_rate_deg 240`; note: with
  coupling on, `max_heading/torso_change_deg` are inert).
- Asymmetric / backward speed envelope (540ccb0).
- Tooling (production): fixed-scenario scorecard + freeze metric + high-density
  eval, in-training scorecard, per-rollout checkpointing + `--init_from`,
  polygon-free CPU render, isolated HPC `sc_uv_crowdrl/`.

---

## 4. Test coverage & merge readiness

482 tests (+139), 10 new files. Every load-bearing fix has a regression test:

| Load-bearing fix | Regression test |
|---|---|
| Obs-normalizer count overflow (r355) | `test_nan_robustness.py::test_running_count_capped_no_overflow` |
| Reward-normalizer count overflow (r360) | `test_normalizer.py::test_count_capped_no_overflow`, `::test_update_drops_nonfinite_samples`, `TestRewardNormalizer::test_overflowed_return_var_recovers_finite` |
| Spawn-goal min distance | `test_spawner.py::TestMinSpawnGoalDistance` |
| Dual-impl reward parity | `test_equivalence.py::TestRewardEquivalence::*` |
| Dual-impl obs parity + NaN-robust | `test_equivalence.py::TestObservationEquivalence::{test_build_observations, test_build_observations_nan_robust}` |
| Velocity-weighted reward NaN (r855) | `test_nan_robustness.py::test_step_velocity_weighted_finite_under_nonfinite_velocity` |

**Known gaps (non-blocking, follow-ups):** (1) ONNX `tanh(mean)` export parity
has no committed test (only the unused `verify_onnx` helper); (2) `--init_from` /
`checkpoint_interval` validated only by manual smoke; (3) eval-mode
`disable_auto_reset` is latent/unused (no caller, no test).

---

## 5. Experiment lineage & learnings (preserved before untracking the configs)

The experimental configs are being kept local (untracked), not merged. What each
lineage taught us, so the knowledge survives:

| Config lineage | Learning |
|---|---|
| `exp_memory_optA` (+ smoke) | Option A ego temporal memory (+6D) helps -> **kept in baseline**. |
| `exp_memory_Aplus*`, `Aplusplus*`, `_nostuck`, `_wide` | A+ (neighbor IDs + vel history) improved reward but **regressed goal-rate** via an observation x stuck-termination coupling (agents self-diagnose as stuck). A++ (128D) and wider nets **deferred**. Baseline = Option A only. |
| `exp_layer1_seed43*`, `smoke_layer1_v2` | Layer-1 v2 envelope (asymmetric action range, `max_velocity_magnitude`, `preferred_speed` ego obs, smoothness x100) fixed the "ice-skating" equilibrium -> **baked into action/obs space**. |
| `exp_coupling_smooth{on,off}` | Physical speed-turn coupling makes "slow before the turn" emergent -> **coupling baked in (`turn_pivot_rate 240`), smoothness OFF**. |
| `exp_wall_contact*`, `_lowent*`, `_nogoaldir` | Hard wall-contact penalty (agents braked on walls for free), `entropy_coef 0.02->0.003` (stop runaway entropy), and `use_goal_direction=false` (waypoint-only nav) -> **all baked into baseline**. |
| `exp_collision_penalty_smooth{on,off}` | Early Moderate-tier avoidance-penalty iteration; superseded. |
| `..._stable_bigrooms`, `_sharpturn` | tanh action-squash validated (`action_std` 0.50->0.35, no runaway). The "careful-but-gridlocks" policy. |
| `..._velcoll` | Impact-speed-weighted collision un-froze gridlock (stuck 0.18->0.00, goal 0.64->0.84) but **re-froze by r100**. |
| `..._velprox` | + velocity-weighted proximity solved *moderate*-density freezing but **plateaued ~0.6**; 60-100-agent high density still jams. |
| `..._density` -> `_v2` -> `_v3` | Cheaper low-speed contact + heavier progress + anti-pileup tuning. Each NaN'd at r855 (vel-snapshot, fixed 8e0104f) then r355 (obs count overflow). |
| `..._density_v3_slow60` | Capping speed at 60% **BACKFIRED** into worse gridlock (coll_ag -1086/-1376). **Lesson: the penalty is contact-count x duration driven; slowing agents makes jams worse, and closing-speed is the wrong variable to scale by.** |
| `..._density_v4` | `collision_penalty_cap -2.0` (discount-only weighting). Cleared r355 (obs cap) but exposed r360 (reward-normalizer twin) -> **the new baseline once 481011c is verified**. |
| `..._nomemory` | Memory-off ablation: promising careful behavior (slow, waiting) without memory. |

Two meta-lessons: (1) ~5 reward-tuning iterations were wasted chasing a
numerical bug -- always check the obs/reward NaN signature (which stage is the
first non-finite) before attributing instability to design. (2) The normalizer
count-overflow is policy/reward/geometry-independent (a clock), which is the
tell that it is infrastructure, not the experiment.

---

## 6. Config disposition

**Keep tracked (vital templates):**
- `configs/full_training.yaml` -- generic documented template (referenced by
  `train_mappo.py` + `scripts/run_experiment.sh`).
- `configs/smoke_baseline.yaml` -- minimal smoke.
- the **baseline** (promoted from `exp_nogoaldir_stable_bigrooms_density_v4`).

**Untrack -> keep local (experimental):** all other `configs/exp_*` and the
experiment smokes, plus `sc_uv_crowdrl/run_nomemory.sbatch` (hardcodes an
untracked config + checkpoint -- one-off scaffolding; `run_training.sbatch`
remains the generic launcher).

CI runs only `pytest` (no config dependency), so untracking is safe. Soft
docstring references in `scripts/eval_scorecard.py` / `scripts/render_cpu.py`
point at experiment configs and are repointed at tracked configs.
