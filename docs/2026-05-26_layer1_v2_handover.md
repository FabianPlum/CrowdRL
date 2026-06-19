# Layer 1 v2 — Handover

Branch: `agent_dynamics_refactor` (HEAD = `1a72b65`)
Pushed to origin.

## TL;DR

The agent_dynamics_refactor branch is in a working state. The Layer 1
v2 reward landscape diagnosed via notebook 09 fixed the ice-skating
pathology, the action caps were empirically relaxed to ~4x the
biomechanical envelope so PPO could discover real avoidance, and the
resulting training run cleared all 6 curriculum phases by rollout 40
and stabilised at 0.80–0.86 goal rate in `full`.

The remaining failure mode is **collision-dominated success** — agents
plow through neighbours because `goal_bonus = +50` outweighs short
contact penalties. A follow-up config
(`configs/exp_layer1_seed43_retune.yaml`) is staged to address this;
it has not been trained yet at the values currently on disk.

## What lives where

| Artifact | Path |
|---|---|
| Trained Layer 1 v2 policy | `results_exp_layer1_seed43/checkpoint_rollout_0170.pt` (plus every 5 rollouts) |
| Retune-iteration policy (mid-training snapshot) | `results_exp_layer1_seed43_retune/checkpoint_rollout_0085.pt` |
| Reward-landscape diagnostic notebook | `examples/09_reward_landscape.ipynb` |
| Evaluation / video-render notebook | `examples/06_full_training.ipynb` (set to LOAD_PRETRAINED=True, points at rollout 85 of retune) |
| Main training YAML | `configs/exp_layer1_seed43.yaml` |
| Next-iteration YAML | `configs/exp_layer1_seed43_retune.yaml` |
| Smoke YAML | `configs/smoke_layer1_v2.yaml` |
| Design plan | `plan/agent_dynamics_refactor.md` |

## What changed today (commits on branch since `5bd69e4`)

```
da2a78b  Training infra: per-rollout checkpoints, --init_from warm-start, matplotlib defer
3746178  Polygon-free video rendering + eval-mode batched env
ce249f5  Notebook 06 -> load + evaluate the Layer 1 v2 retune checkpoint
367b85f  Empirically working Layer 1 v2 config (action caps + goal-seeking shaping)
1a72b65  Retune YAML for collision-suppression iteration
```

(Earlier in the day on the same branch: notebook 09, action-space
envelope refactor, preferred_speed in ego obs, reward retune, F2
scenario.)

## Layer 1 v2 — what worked

Three things together pulled training out of the ice-skating regime:

1. **Reward retune (commit `63db6e7`)**: `speed_deviation_weight` -0.1
   → -0.005 (20x down), `jerk_penalty_weight` -1e-4 → -1e-5 (10x
   down). Validated by notebook 09: under the new landscape, the
   "brake before wall" trajectory is +12 reward over the "plow through"
   trajectory (was -1.3 under v1). The ice-skating equilibrium was
   broken on paper before any training.

2. **`preferred_speed` exposed as ego obs feature (commit `843c993`)**:
   the policy can now observe the per-agent speed target it's being
   penalised against. Was previously invisible. obs_dim 104 → 105.

3. **Action caps loosened from biomechanical envelope (commit
   `367b85f`)**: 1.146/0.573/1.719 deg/step (115/57/172 deg/s) → 4/2/4
   deg/step (400/200/400 deg/s). The plan's envelope was correct for
   *comfortable walking* but too tight for PPO's exploration budget
   inside the curriculum's early phases. After this change the
   curriculum cleared in ~40 rollouts. **Caveat**: the new caps are
   above the comfortable walking band; this is an empirical fix
   waiting for a literature justification, not a principled choice.

## Layer 1 v2 — what's still wrong

1. **Plenty of collisions**. Confirmed visually in the training videos
   at rollout ~150. With `goal_bonus = +50` and `collision_penalty =
   -1/step`, a 10-step contact event costs -10 vs +50 for reaching
   goal — the policy correctly chooses to plow.

2. The retune config (`exp_layer1_seed43_retune.yaml`) addresses this
   with:
   - `collision_penalty: -1 → -5` (5x stronger contact cost)
   - `agent_proximity_penalty_near: -0.01 → -0.05` (gives the policy
     a "stay back" gradient before contact triggers)
   - `desired_velocity_weight: 0.05 → 0.2` (velocity filter tau drops
     from ~200 ms to ~50 ms — policy can actually brake in one
     avoidance window)

3. **Open question on the retune YAML**: the on-disk values are 4/2/4
   for action caps, but the `results_exp_layer1_seed43_retune` run
   was trained with 10/10/10 (per its `config_resolved.yaml`). The
   YAML was edited after that run. **Decision needed before the next
   launch**: 4/2/4 or 10/10/10?

## How to continue tomorrow

### Option A — train the retune config from scratch

```powershell
uv run python train_mappo.py --config configs/exp_layer1_seed43_retune.yaml --gpus 1
```

Expected wall time: ~3-4 hours on a single 4090.

### Option B — warm-start from the working v2 checkpoint

This uses the new `--init_from` flag. Loads weights and obs/reward
normalizer from the v2 run but resets curriculum/optimizer/history,
so the retuned reward landscape sees a partially-competent policy as
its starting point instead of a fresh random one:

```powershell
uv run python train_mappo.py `
  --config configs/exp_layer1_seed43_retune.yaml `
  --gpus 1 `
  --init_from results_exp_layer1_seed43/checkpoint_rollout_0170.pt
```

The launch provenance is recorded in
`results_exp_layer1_seed43_retune/config_resolved.yaml` under
`_launch.init_from` for reproducibility.

### Option C — evaluate the current retune checkpoint first

The retune-rollout-85 checkpoint already exists. Open
`examples/06_full_training.ipynb` and run all cells with
`LOAD_PRETRAINED=True, SKIP_TRAINING=True` (already configured).
The notebook will rebuild env_config from the checkpoint's sibling
YAML and render eval videos / collision stats. Looking at those first
might inform the next config change before launching another run.

## Tooling notes

- **Per-rollout checkpoints**: `checkpoint_interval: N` in the YAML
  saves `checkpoint_rollout_<N>.pt` every N rollouts. Default is 0
  (final only). Set to 5 in exp_layer1, 1 in smoke.

- **Warm-start vs resume**: `--resume_training` continues an existing
  run from its final checkpoint with full state (optimizer,
  curriculum, history). `--init_from <ckpt>` is different: it pulls
  only weights + normalizers and starts the curriculum fresh. Use the
  latter when changing the reward / action config so the previous
  optimizer momentum doesn't drag the policy back toward the old
  landscape.

- **GPU-batched eval render**: `BatchedTorchEnv(disable_auto_reset=True)`
  + `EpisodeFrames(walls=…)` + `render_episode_video(...)` now works
  end-to-end. The batched env never needed a Shapely polygon for
  rendering; this just bridges the last gap.

## Layer 2 status

Untouched today. The plan (`plan/agent_dynamics_refactor.md` section
4) is still the design of record. F2 scenario in notebook 09 showed
that **the reward landscape would reward true avoidance maneuvers by
~150 reward over no-brake and ~107 over brake-only**, using only
Layer 1's existing torso-rotation action — so the immediate question
for Layer 2 is whether the policy *discovers* the F2 maneuver in
training, not whether the reward landscape supports it.

If the retune run produces clean head-on counterflow (lane formation,
torso-rotate-to-pass), Layer 2 may be deferrable. If it still
collides head-on after extra training, Layer 2's explicit yaw_rate
state + yaw acceleration action is the path forward.

## Known issues

- `test_vec_env.py::TestRolloutCollector::test_collect_returns_episodes`
  timed out in the most recent full test run due to GPU contention
  with a live training process. Not a regression; passes when the
  GPU is idle.
- Two DDP/libuv tests (`test_ddp_kl_*`) fail on Windows due to PyTorch
  not being built with libuv support. Pre-existing, unrelated to this
  branch's work.

## Open decisions for tomorrow

1. Confirm action caps in `exp_layer1_seed43_retune.yaml`: 4/2/4 (on
   disk now) vs 10/10/10 (what was actually trained).
2. Choose A / B / C above for the next session.
3. (Lower priority) Decide whether to commit `.claude/settings.local.json`
   — currently uncommitted; contains personal env / permissions.
