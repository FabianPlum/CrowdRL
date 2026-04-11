# Experiment plan — overnight run sequence

Baseline config at `configs/full_training.yaml` (proximity -0.01/1.0m,
max_steps 2000, tier_weights [.05,.05,.1,.35,.45], progress_weight 1.0,
seed 42).

## Run 1 (current): baseline_seed42
No config change. Fresh baseline with reverted proximity settings.

## Run 2: baseline_seed43
Change: `seed: 42` -> `seed: 43`
Purpose: noise floor for all subsequent comparisons.

## Run 3: exp_max_steps_2500_seed42
Change: `max_steps: 2000` -> `max_steps: 2500`
Reset seed: 43 -> 42
Hypothesis: TIER_3B 1699-step episodes are borderline timing out. An
extra 500 steps lets near-misses complete. Predict TIER_3B goal rate
+0.02 to +0.04, no effect on other tiers.

## Run 4: exp_tier_weights_3B_seed42
Change: `tier_weights: [0.05, 0.05, 0.1, 0.35, 0.45]`
     -> `tier_weights: [0.03, 0.03, 0.04, 0.30, 0.60]`
Other settings revert (max_steps back to 2000).
Hypothesis: More TIER_3B sample weight -> better TIER_3B training.
Predict TIER_3B goal rate +0.02 to +0.05, small dip in easier tiers.

## Run 5: exp_progress_weight_1_5_seed42
Change: `progress_weight: 1.0` -> `progress_weight: 1.5`
Other settings revert (tier_weights back to [.05,.05,.1,.35,.45]).
Hypothesis: Stronger goal-seeking gradient -> faster episodes,
possibly more collisions (higher risk). Predict all tiers see -100 to
-200 ep_len, uncertain effect on reward.

## Run 6: replicate best single exp with seed 43
Pick the winning single-knob experiment from runs 3/4/5 (largest
goal-rate improvement that is outside the noise floor). Replicate with
seed 43 to confirm the effect is not noise.

## Run 7 (optional): combined winners with seed 42
Stack the best two experiments (e.g. max_steps 2500 + tier_weights
shift) in one run. Tests whether effects stack or interact.

## Run 8 (optional): combined replicate with seed 43
Confirm the combined-winners effect.

## Stop conditions
- Goal rate < 0.5 at rollout 50 in full phase -> kill, archive, next
- NaN losses -> kill, abort sequence, report
- Wall time > 2.5h per run -> kill

## Between-run checklist
1. `mkdir -p results_<name> && cp configs/full_training.yaml results_<name>/config.yaml.snapshot`
2. `git rev-parse HEAD > results_<name>/git_rev.txt`
3. `date --iso-8601=seconds > results_<name>/start_time.txt`
4. Apply config diff for this experiment
5. `uv run python train_mappo.py --config configs/full_training.yaml 2>&1 | tee results_<name>/training.log` (in background)
6. On completion: `cp -r results_full_training/. results_<name>/`
7. `uv run python scripts/analyze_run.py results_<name> --last 2000 --compare results_baseline_seed42 --noise-reward <R> --noise-goal <G>`
8. Write `results_<name>/notes.md`
9. Revert config change for the next experiment (unless this is a cumulative test)
