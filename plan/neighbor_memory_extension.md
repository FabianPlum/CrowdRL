# Neighbor-Memory Extension to Option A

Follow-up to `plan/agent_memory_research.md`. Assumes Option A (ego temporal
memory, +6D) is already landed. Targets the failure modes Option A does not
address: social-coordination pathologies (oscillation, deadlocks, lane
breakdown, courtesy behaviors).

## 1. Why Option A alone is not enough for social navigation

Option A only contains ego features. The social channel is still a K=8
snapshot of (rel pos, rel vel, orient, body dims) rebuilt from scratch each
step. Consequences:

- The policy cannot distinguish "a neighbor that has been consistently
  approaching me for 0.5s" from "a neighbor that appeared this frame with a
  similar velocity". Both look identical to it.
- There is no persistence of neighbor identity across timesteps. Even the
  K-nearest ranking can reshuffle when two neighbors are at similar
  distances, so "neighbor index 3 at step t" is not the same physical
  agent at step t+1. Any memory indexed by list position is noise.
- Reciprocal courtesy ("I yielded last time, so you should yield now")
  and lane inference ("most neighbors are heading east -> join the eastbound
  lane") require neighbor motion history, which is absent.

Classical pedestrian models side-step this via topological constraints
(Topology-Guided ORCA, HSFM) or explicit deadlock-resolution heuristics.
Learned policies need equivalent information in the observation.

## 2. Design

Three layered additions, each independently ablate-able.

### 2.1 Persistent neighbor identity (prerequisite, +0 obs dims)

Maintain a per-agent neighbor-ID table: for each agent i, the list of
stable neighbor slot assignments `slot_i[k] -> global_agent_id`. Each step:

1. Compute current K nearest neighbors (cheap KNN we already have).
2. Match to last step's assignments greedily (Hungarian is overkill,
   greedy by min distance works for K=8).
3. Fill empty slots with the unmatched nearest new neighbors.
4. Evict slots whose previous assignee is now out of sensing range.

Cost: one (E, N, K) int32 tensor of agent IDs, one scatter + gather per
step. No new observation dimensions; the benefit shows up when we *use*
the persistent IDs to look up neighbor history.

This is the part that is least obvious from reading the code and has the
highest payoff-to-cost ratio: it is the enabler for everything below.

### 2.2 Neighbor velocity history (+16D, cheapest informative step)

For each of the K=8 persistent neighbors, emit its velocity averaged over
the last W=5 simulation steps (~50ms at dt=0.01), in ego frame.
Dimensions: 2 per neighbor x 8 neighbors = 16.

What this encodes: second-order motion of each neighbor. "Is that person
accelerating toward me, or slowing down?" is the one question the current
snapshot cannot answer. Over a 5-step window we bias toward smoothing out
action noise without delaying response to real direction changes.

Storage: per-agent ring buffer of neighbor velocities `(E, N, W+1, K, 2)`
at float16 = 4 bytes per neighbor per slot. For 64 envs * 100 agents *
6 * 8 = 307,200 slots = 1.2 MB. Negligible.

Update rule: each step, gather current velocities from the persistent
neighbor slots and scatter into the ring buffer. Reading returns the mean
of the last W slots.

### 2.3 Neighbor relative-trajectory features (+24D, richer)

For each neighbor, emit 3 scalars computed from the neighbor's *own*
trajectory (not relative to ego):

1. Neighbor path efficiency over its last 50 steps (is this neighbor
   making purposeful progress or themselves stuck?).
2. Displacement of neighbor over last W=50 steps, normalised by
   neighbor's preferred speed (in ego frame).
3. Goal-approach rate of neighbor over last W=50 steps (not the full
   direction, just the magnitude of their goal approach).

Total: 3 x 8 neighbors = 24 dims.

Requires: a shared (E, N, W+1) ring buffer of per-agent positions and
goal distances. Option A already has this! We just need a separate
"foreign" lookup that pulls ring-buffer entries for the persistent
neighbor slots rather than for the ego agent.

This is the part that gives the policy a read on "everyone around me is
doing well, I am the anomaly" vs "we are all stuck together", which is
the core signal needed to distinguish solo failure from collective
deadlock.

### 2.4 Ablation matrix

| Config | ego memory (A) | persistent IDs | neighbor vel hist | neighbor traj feats | obs dim |
|--------|---------------|----------------|-------------------|---------------------|---------|
| A      | yes           | no             | no                | no                  | 88      |
| A+     | yes           | yes            | yes               | no                  | 104     |
| A++    | yes           | yes            | yes               | yes                 | 128     |

A+ is the cheapest test of "does any neighbor temporal memory help".
A++ is the full neighbor-history test. Run seeds 42 and 43 for both.

## 3. Implementation outline

Six focused commits, each independently testable. Follows the same layered
pattern as the Option A commit (`760d24c`). File paths and line numbers
below reference the state at the tip of `memory_experiments` after Option A
landed.

### Commit 1: persistent neighbor ID infrastructure (no behavior change)

- `packages/crowdrl-core/src/crowdrl_core/world_state.py`:
  add an optional field
  ```python
  neighbor_ids: NDArray[np.int32] | None = None
  """(n_agents, K) -- agent index of each persistent neighbor slot, or -1
  for empty. Populated when ObsConfig.use_neighbor_memory is True."""
  ```

- `packages/crowdrl-torch/src/crowdrl_torch/types.py`:
  mirror field on `TorchWorldState`:
  ```python
  neighbor_ids: Tensor  # (E, N, K) int32; -1 == empty slot
  ```
  Initialise in `make_initial_state` as `torch.full((E, N, K), -1, int32)`.

- `packages/crowdrl-core/src/crowdrl_core/sensing.py`:
  new function (paralleling `knn_social_batch`):
  ```python
  def match_persistent_neighbors(
      positions: NDArray[np.float64],         # (N, 2)
      prev_slots: NDArray[np.int32],           # (N, K)
      active_mask: NDArray[np.bool_],          # (N,)
      sensing_radius: float,
      k: int,
  ) -> NDArray[np.int32]:
      """Greedy-match prev_slots to current-frame KNN.

      Keeps a previously assigned slot if the neighbour is still within
      sensing_radius; evicts it otherwise. Empty slots (-1) are filled
      with the nearest unassigned active agent.

      Returns new (N, K) int32 assignment.
      """
  ```
  No `knn_social` changes yet -- this commit only adds the matcher and
  plumbs it through; the existing (non-persistent) KNN still powers the
  social channel.

- `packages/crowdrl-torch/src/crowdrl_torch/sensing.py`:
  add `match_persistent_neighbors` with the same contract on tensors.
  Greedy match is one gather + one scatter per step:
  ```python
  def match_persistent_neighbors(
      positions: Tensor,       # (E, N, 2)
      prev_ids: Tensor,        # (E, N, K) int32
      active_mask: Tensor,     # (E, N)
      n_agents: Tensor,        # (E,)
      config: EnvConfig,
  ) -> Tensor:                 # (E, N, K) int32
      ...
  ```

- Tests:
  - `packages/crowdrl-core/tests/test_sensing.py::test_match_persistent_neighbors_retains_stable_neighbor`
    Two agents within sensing range; after several steps their slot
    indices remain pinned to the same global ID.
  - Same file: `test_match_persistent_neighbors_evicts_out_of_range`.
  - `packages/crowdrl-torch/tests/test_equivalence.py::TestPersistentNeighborEquivalence`.

- Verify: `uv run pytest -k persistent_neighbor` passes.

### Commit 2: enable persistent matching in reset + step path

- `packages/crowdrl-env/src/crowdrl_env/crowd_env.py::CrowdEnv.reset`:
  if `use_neighbor_memory`, allocate `neighbor_ids = -1` and run one
  initial `match_persistent_neighbors` call so slots are populated
  before the first observation.

- `packages/crowdrl-env/src/crowdrl_env/crowd_env.py::CrowdEnv.step`:
  call `match_persistent_neighbors` after position update, before obs
  builder. Store the new `neighbor_ids` back onto `_world`.

- `packages/crowdrl-torch/src/crowdrl_torch/step.py::batched_step`:
  same pattern -- compute `new_neighbor_ids` right before obs build,
  pass it through to `new_state` and `build_observations`.

- `packages/crowdrl-torch/src/crowdrl_torch/batched_env.py::BatchedTorchEnv.reset`:
  propagate `neighbor_ids` in the state -> obs passthrough and in the
  completed-reset scatter.

- `packages/crowdrl-torch/src/crowdrl_torch/geometry_repr.py::prepare_reset_data`:
  the persistent-ID table starts at all -1 and is populated after the
  first match, so no extra field here -- just the `neighbor_ids` tensor
  allocated inside `make_initial_state`.

- No new observation dimensions at this point -- the field is computed
  and threaded through but nothing reads it yet. `uv run pytest` must
  stay green.

### Commit 3: neighbor velocity history state + ring buffer

- `packages/crowdrl-core/src/crowdrl_core/world_state.py`:
  ```python
  neighbor_vel_history: NDArray[np.float64] | None = None
  """(n_agents, W+1, K, 2) -- ring buffer of neighbor velocities in
  the ego frame of each agent. W is ObsConfig.neighbor_vel_history_window.
  Slots with neighbor_ids == -1 hold zeros."""
  ```

- `TorchWorldState`: `neighbor_vel_history: Tensor` with shape
  `(E, N, W+1, K, 2)` float32. Initialised as zeros in
  `make_initial_state`.

- `packages/crowdrl-torch/src/crowdrl_torch/step.py::batched_step`:
  after `new_neighbor_ids` is computed (commit 2), gather the neighbor
  velocities from the global velocity tensor via the persistent IDs,
  rotate them into the ego frame (reuse the rotation block from
  `knn_social`), and scatter into the ring buffer at
  `(step_count % buf_size)`. Mirror the existing Option A pattern.

- Equivalent update in `CrowdEnv.step`.

- `packages/crowdrl-torch/src/crowdrl_torch/geometry_repr.py`:
  `prepare_reset_data` allocates a pre-zeroed
  `(max_agents, W+1, K, 2)` float32 array -- the content at reset is
  "no history yet, effectively zero neighbor velocities", matching
  Option A's "first-step pos == spawn" convention.

- Tests:
  - `packages/crowdrl-core/tests/test_observation.py::test_neighbor_vel_history_ring_buffer`:
    step an agent with a static neighbor, confirm the buffer fills
    correctly and reads of slot `(step_count+1) % buf_size` return the
    oldest entry.
  - Equivalence test for the ring-buffer update under numpy vs torch.

### Commit 4: neighbor velocity history observation feature (+16D)

- `packages/crowdrl-core/src/crowdrl_core/observation.py`:
  - Extend `ObsConfig`:
    ```python
    use_neighbor_memory: bool = False
    neighbor_vel_history_window: int = 5   # 50ms at dt=0.01
    ```
  - Update `ObsConfig.obs_dim` to add `K*2 if use_neighbor_memory else 0`.
  - New `_per_agent_neighbor_velocity_history_features` helper that
    reads `world.neighbor_vel_history` for the ego agent and returns a
    `(K*2,)` vector (flattened mean over the window for each slot).
  - `build_observation` appends this to `parts` when the flag is set.

- `packages/crowdrl-torch/src/crowdrl_torch/observation.py`:
  - `build_observations` gains new kwargs
    `neighbor_vel_history: Tensor | None` and produces the same block
    via a vectorised mean-reduction along `dim=2` of the ring buffer,
    masked to zero where `neighbor_ids == -1`.
  - Append to `parts` when `config.use_neighbor_memory`.

- `train_mappo.py::build_env_config`: thread the new fields from YAML.

- `packages/crowdrl-torch/src/crowdrl_torch/types.py::EnvConfig`: mirror
  `use_neighbor_memory` and `neighbor_vel_history_window`.

- Tests:
  - `test_observation.py::TestNeighborMemory::test_obs_dim_grows_by_16`.
  - `test_observation.py::TestNeighborMemory::test_static_neighbor_yields_zero_vel_history`
    -- ego-frame rotation applied to a zero velocity vector is zero.
  - `test_equivalence.py::TestNeighborVelHistoryEquivalence`.

### Commit 5: neighbor trajectory features (+24D, depends on Option A buffers)

- Reuses Option A's `pos_history` and `gdist_history` ring buffers --
  this commit adds the "foreign" lookup through `neighbor_ids` to pull
  entries from those ring buffers for each persistent neighbor slot.

- `packages/crowdrl-core/src/crowdrl_core/observation.py`:
  new helper `_neighbor_trajectory_features(world, agent_idx, config)`:
  ```python
  # For each persistent slot k:
  #   nb_id = world.neighbor_ids[agent_idx, k]
  #   if nb_id < 0: return 3 zeros
  #   disp_w = ||pos_history[nb_id, read_idx] - positions[nb_id]||
  #   cum_nb = world.cumulative_path_length[nb_id]
  #   eff_nb = disp_from_spawn_nb / max(cum_nb, eps)
  #   gprog_nb = (gdist_history[nb_id, read_idx] - gdist_now[nb_id]) / expected
  #   return (disp_w, eff_nb, gprog_nb)
  ```
  Emit `K*3 = 24` scalars, flattened row-major.

- Torch equivalent in `crowdrl_torch/observation.py`:
  ```python
  # Gather cum_path, spawn_pos, pos_history, gdist_history via neighbor_ids
  # (batched gather along dim=1 with scatter-friendly indices).
  # Reuse the same temporal_memory_window parameter as Option A.
  ```

- `ObsConfig.obs_dim` becomes
  `ego + social + rays + nav + temporal(6) + neighbor_vel(K*2) + neighbor_traj(K*3)`
  = `88 + 16 + 24 = 128` at full A++ config.

- Tests:
  - Parity: synthetic setup with one ego and one static neighbor,
    assert neighbor trajectory features match Option A's features for
    the neighbor agent (modulo ego-frame rotation).
  - Equivalence numpy vs torch.

### Commit 6: configs + smoke + experiment runner

- `configs/full_training.yaml`: add `use_neighbor_memory: true` plus
  the two window fields. Document that obs_dim becomes 128.

- `configs/smoke_memory_Aplusplus.yaml`: smoke variant (one phase,
  short rollouts) mirroring the Option A smoke config.

- `configs/exp_memory_Aplus_seed43.yaml`: A+ config (persistent IDs +
  vel history only). Used as the cheap "does any neighbor temporal
  memory help" run.

- `configs/exp_memory_Aplusplus_seed43.yaml`: full A++ config.

- Run order:
  1. `uv run python train_mappo.py --config configs/smoke_memory_Aplusplus.yaml --gpus 1`
     (~1 min, confirms plumbing).
  2. `uv run python train_mappo.py --config configs/exp_memory_Aplus_seed43.yaml`
     (full 500-rollout, 2 GPUs, ~2.5h).
  3. `uv run python scripts/analyze_run.py results_exp_memory_Aplus_seed43 \
     --compare results_exp_memory_optA_seed43` to score against the
     fair Option A baseline.
  4. If A+ clearly beats A, run A++ next. If A+ is no better, stop --
     the 24 extra dims of trajectory features aren't worth the
     complexity without at least A+ moving the needle.

## 4. Compute + throughput budget

- Obs dim growth: 88 -> 104 (+18%) or 88 -> 128 (+45%). The MLP actor has
  ~178k params; adding 40 input dims = 40 * 256 * 2 = 20k extra params,
  ~11% model growth. Well within 4090 budget.
- Persistent ID matching: O(K^2 x N x E) = 64 x 100 x 64 = 410k ops/step,
  fully fused. Negligible.
- Ring-buffer scatter: one extra scatter per step for the neighbor
  velocities. Reuses the same triton kernel pattern Option A already
  compiles (modulo the `descriptive_names` fix that lets this fuse
  without hitting ENAMETOOLONG).

Expected overhead vs Option A: ~2-4% throughput hit.

## 5. Evaluation criteria

Option A successful means path_efficiency drops for stuck agents -> they
move on. Neighbor memory should further drop three specific metrics:

1. **Oscillation rate**: fraction of 2s windows where an agent's heading
   flipped sign more than 4 times. Current baseline ~ TBD (need to
   measure). Hypothesis: neighbor memory reduces this by 30%+ in Tier 1
   bottlenecks.
2. **Bottleneck throughput**: agents/s passing through the tightest
   aperture in each Tier 1 episode. Hypothesis: up from current.
3. **Mutual-stuck rate**: fraction of stuck-terminated agents whose
   nearest neighbor was also stuck-terminated in the same window. This is
   the signature of coordinated deadlock that Option A cannot fix.

## 6. Open questions

- Does W=5 for neighbor velocity history need to be seeded more carefully?
  On reset, we have no history, so first W steps use fake data. Option A
  solves this by pre-filling with spawn position; the equivalent here is
  pre-filling with the first observed velocity, which is fine.
- Should neighbor features be masked when the neighbor slot has been
  empty for some time (i.e. that physical agent moved out of sensing)?
  Yes -- emit zeros and rely on `active_mask` style gating.
- Do we need to re-tune the reward when obs dim changes? No -- the reward
  is computed from WorldState, not observations. But the normalizer must
  re-accumulate its running stats.

## 7. Experimental results (A+ run, 2026-04-12)

Full 500-rollout, seed 43, 2x RTX 4090 DDP training of the A+ variant
(commits 1-4 + 6 + vectorise fix, obs_dim 104). Wall time 1h 57m.

### A+ vs A (ego memory only, 88D) -- last 2000 full-phase episodes

Overall: goal_rate -1.3 pp, **reward +2.99** (A+ better on reward).
Every tier shows positive reward delta and reduced penalty_rate.
The mechanism is reduced per-step collision/proximity penalties from
the neighbor velocity-history feature -- agents can read second-order
motion of their neighbors and act preemptively.

### A+ vs pre-memory baseline (82D) -- last 2000 full-phase episodes

Overall: goal_rate -4.8 pp, **reward +1.23** (A+ beats baseline on
reward for the first time among memory variants). On complex tiers
(TIER_2/3A/3B) the reward improvement is strongest (+3.56 on T2,
+3.48 on T3B), confirming the social-coordination hypothesis.

### Stuck-termination interaction (persistent across all memory variants)

All three memory runs (A, A+, and presumably A++) share the same
goal-rate regression vs baseline, and it tracks with the eval/training
discrepancy: eval (no stuck termination, low density) shows memory
policies matching or beating baseline, while training (stuck
termination enabled, high density) shows them underperforming. The
mechanism: memory features give agents path_efficiency/window
signals that correlate with the stuck-termination threshold -> the
policy learns "self-diagnose and accept the timeout_penalty" instead of
"escape the deadlock". This is an observation x environment-design
interaction, not a feature-quality problem.

### A++ decision: deferred

At 104D, the first hidden layer (256 wide) has 26,624 weights -- a
16:1 compression ratio from input to first hidden. Going to 128D
pushes that to 32,768 (20:1 ratio at the bottleneck). Without
widening the first layer, A++ risks underfitting the richer feature
space. Recommended: revisit A++ with [384, 256] or [512, 256] hidden
layers after the stuck-termination coupling is resolved.

### Recommended next step

**A+ with stuck_termination_enabled=false** -- single YAML line flip.
If the goal-rate gap closes while reward stays better, A+ is the
clear winner for the observation stack and we ship it as the default.

## 8. Not in scope for this doc

- Self-attention / transformer social encoders (too much engineering for
  the marginal gain at our current tier depth).
- Learned persistent ID (e.g. a small matching network). Greedy KNN
  matching is sufficient for K=8 and the accuracy needed.
- Spatial grid visit counts (Option D from the original research doc) --
  still ruled out unless Option A/A+/A++ shows circling pathologies.
