# Agent Memory for CrowdRL: Research Findings and Design Options

## 1. Problem statement

CrowdRL agents observe a purely Markovian 79D snapshot (ego state, social sensing, raycasts). The policy cannot know:
- **Where it has been** relative to its start (displacement, path shape)
- **Whether it is stuck** (oscillating, circling, deadlocked)
- **How much time has elapsed** relative to the episode budget

These are critical for bottlenecks, corridors, and high-density Tier 1-5 scenarios. The question is what "memory" mechanism best fits our constraints: GPU-parallelisable, lightweight for 100s-1000s of agents at inference, ONNX-exportable, and preserving the WorldState transfer guarantee.

---

## 2. Literature review

### 2.1 Recurrent policies (GRU/LSTM in MAPPO)

**rMAPPO** (Yu et al. 2022, "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games") is the standard. Architecture: `FC layers -> GRU(64) -> FC output`. Each agent carries a hidden state vector h_t of size 64, updated every timestep. Training uses Truncated BPTT with chunk_length=10.

Reference implementation: [marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy). Also see [MarcoMeter/recurrent-ppo-truncated-bptt](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt) for a clean single-agent reference.

**Tradeoffs**:
- (+) Handles partial observability naturally -- GRU learns what to retain
- (+) Only 64 floats of state per agent (vs hundreds for frame stacking)
- (+) GRU outperforms LSTM in most RL benchmarks (fewer params, fewer gates)
- (-) Sequential dependency during rollout collection (h_t depends on h_{t-1})
- (-) Requires chunked sequence buffer, not flat batches -- significant refactor of our `RolloutBuffer`/`FlatBatch`
- (-) ONNX export requires explicit hidden state I/O at every `session.run()` (doable, but adds deployment complexity; see [onnxruntime#6936](https://github.com/microsoft/onnxruntime/issues/6936))
- (-) Training ~15-30% slower due to sequential GRU within chunks
- (-) "Generalization, Mayhems and Limits in Recurrent PPO" (2022, [arXiv:2205.11104](https://arxiv.org/abs/2205.11104)) shows recurrent PPO can overfit to training episode lengths and struggle to generalise to different temporal horizons

**Key detail**: Hidden state reset policy matters. Short-episode environments benefit from NOT resetting hidden states at episode boundaries. For our variable-length episodes, this needs experimentation.

### 2.2 Frame stacking / observation history

Concatenate the last K observation-action pairs into one flat vector. Used extensively in robotics locomotion:

- **Berkeley humanoid** (Radosavovic et al., 2023, Science Robotics): 16-step context window of obs-action pairs processed by a causal transformer. ~1072D input.
- **Cassie bipedal** (Li et al. 2024): Last 10-50 steps of proprioceptive obs + actions, processed by an adaptation MLP.
- **RMA** (Kumar et al. 2021): Base policy on current obs + adaptation module on last 50 steps of I/O history to infer environment parameters.
- **walk-these-ways** (Improbable-AI/MIT): `num_observation_history=15` for legged locomotion ([config](https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym/envs/base/legged_robot_config.py)).
- **legged_gym** (leggedrobotics): Tracks `last_dof_vel`, `last_root_vel` as single-step history. Reward computes jerk from velocity differences.
- **Adaptive Stacking** (Nangue et al., 2025, [OpenReview](https://openreview.net/forum?id=W7O95Q1qzT)): Agent learns WHICH past observations to retain rather than blindly stacking last N.

**Tradeoffs**:
- (+) Trivially GPU-parallelisable -- pure data transformation, no sequential dependency
- (+) No change to PPO update code; `FlatBatch` works as-is
- (+) ONNX export trivial -- model is a standard MLP
- (+) Provides velocity/acceleration inference implicitly
- (-) Obs dimension scales as K * obs_dim. K=3 with 80D = 240D; K=5 = 400D
- (-) MLPs lack inductive bias for temporal patterns in stacked frames
- (-) Most information in social/raycast dims changes slowly -- stacking repeats redundant data

**Practical insight**: A hybrid approach works well -- stack K=2-3 frames of **ego-state only** (7D each) plus the current full observation. This gives velocity/acceleration inference without exploding social/raycast dimensions.

### 2.3 Spatial memory maps / egocentric occupancy grids

- **Active Neural SLAM** (Chaplot et al., ICLR 2020): Learned mapper producing egocentric top-down 2D occupancy maps from RGB-D. Global policy consumes the map, local policy navigates to waypoints.
- **Cognitive Mapper and Planner** (Gupta et al. 2017): End-to-end differentiable encoder-decoder building a latent grid map.
- **Spatially-Enhanced Recurrent Memory** (Yang et al. 2025, [arXiv:2506.05997](https://arxiv.org/abs/2506.05997)): Modifies GRU gates with spatial transformation term for mapless navigation. 23.5% improvement over standard RNNs in navigation success. Key insight: standard LSTM/GRU struggle with spatial transformation of observations from varying egocentric perspectives.

**Assessment for CrowdRL**: These are designed for visual navigation (RGB-D, building-scale exploration). Our agents already have raycasts (local awareness) + navmesh waypoints (global path guidance) in 2D. Maintaining a per-agent grid (e.g. 16x16 = 256 cells) for 100 agents across 4096 parallel envs = 409,600 grids. Feasible but expensive and overkill for our setting. Not recommended.

### 2.4 Positional encoding / displacement scalars

Adding scalar features encoding trajectory history as simple numbers:

| Signal | What it encodes |
|--------|----------------|
| Displacement from start | `\|\|pos_t - pos_0\|\|` -- how far from spawn |
| Cumulative path length | Total distance traveled (captures circling) |
| Path efficiency ratio | displacement / path_length -- 1.0 = straight, ~0 = looping |
| Elapsed fraction | step / max_steps -- normalized time pressure |

**Literature precedent**: Robotics locomotion (RMA, Cassie RL, walk-these-ways) uses similar scalar features: elapsed time, cumulative distance, phase/clock signals. The **Self-Monitoring Navigation Agent** (Ma et al. 2019, [arXiv:1901.03035](https://arxiv.org/abs/1901.03035)) uses auxiliary progress estimation -- conceptually analogous.

In pedestrian simulation, the fundamental diagram relates density to flow. Agents aware of their own path efficiency can learn density-dependent speed adaptation.

**Tradeoffs**:
- (+) Near-zero computational cost (scalar accumulations per step)
- (+) Zero ONNX impact (observation features, not architecture)
- (+) Provides information the policy literally cannot infer from a single frame
- (+) Path efficiency ratio directly encodes "am I stuck?"
- (-) Limited expressiveness vs full temporal memory
- (-) Cannot capture direction-specific patterns ("that corridor was a dead end")

### 2.5 Stuck detection / stagnation signals

**Displacement-over-window**: Compute `||pos_t - pos_{t-N}||` for window N (e.g. 50 steps = 0.5s). If displacement is below threshold, agent is stagnant. Can be used as observation feature, reward penalty, or early termination trigger.

**EMA velocity**: `v_ema_t = alpha * |v_t| + (1-alpha) * v_ema_{t-1}` with alpha~0.05-0.1. Trivially vectorisable. Cost: 1 multiply-add per agent per step. See "Going Nowhere Fast" (Nikolov 2011) for mobile robot stuck detection patterns.

**Ring buffer computed features** (more informative than raw history):

| Feature | Formula | Meaning |
|---------|---------|---------|
| Displacement | `\|\|pos_now - pos_{t-N}\|\|` | Net progress over window |
| Path length in window | sum of step distances | Total recent distance |
| Straightness ratio | displacement / path_length | 1 = straight, ~0 = looping |
| Position spread | std(positions in buffer) | Low = stuck/oscillating |

A ring buffer of N=50 positions costs 50x2 floats per agent (~400 bytes). The computed features reduce to 3-4 scalars.

**Count-based exploration**: Tang et al. (2017, [arXiv:1611.04717](https://arxiv.org/abs/1611.04717)) use SimHash for state discretisation + visit counts. GPU-friendly variant: spatial grid `(n_envs, grid_H, grid_W)` with scatter-increment operations and exponential decay. Feature extraction: `[visit_count_at_cell, mean_count_in_neighbourhood, time_since_last_visit]`.

### 2.6 Curiosity / intrinsic motivation

- **ICM** (Pathak et al. 2017): Prediction error from forward/inverse models = curiosity. Cost: 3 extra network forward passes per step.
- **RND** (Burda et al. 2019): Prediction error between fixed random network and trained predictor. Simpler than ICM, avoids "noisy TV" problem.
- **NovelD** (Zhang et al. 2021): Difference in RND novelty between consecutive states.
- **Multi-agent extension** (Schaefer 2019, MSc thesis, University of Edinburgh): Shared forward/inverse models across agents with batched inference.

**Assessment for CrowdRL**: For 100 agents, ICM/RND means 200-300 extra forward passes per env step. Prohibitive. The scalar displacement/efficiency signals provide ~90% of the anti-stagnation benefit at <1% of the compute cost. Not recommended unless training on extremely sparse-reward settings.

### 2.7 Transformer-based memory

- **Decision Transformer** (Chen et al. 2021): RL as sequence modeling over (return-to-go, state, action) tuples.
- **Offline Pre-trained Multi-Agent Decision Transformer** (Meng et al. 2021, [arXiv:2112.02845](https://arxiv.org/abs/2112.02845)): Extends Decision Transformer to MARL with offline pre-training on StarCraft (SMAC) tasks.
- **RATE** (Cherepanov et al. 2023, [arXiv:2306.09459](https://arxiv.org/abs/2306.09459)): Recurrent Action Transformer with Memory -- processes trajectory segments recurrently, passing memory embeddings. Memory Regulation Valve gates information retention.
- **ATM** (Yang et al., NeurIPS 2022): Transformer-based working memory for MARL with action parsing and attention over factored entities.

**Assessment for CrowdRL**: Self-attention is O(K^2) in context length. For K=16 and 100 agents across thousands of envs, cost is substantial. A 2-layer, 4-head, 192D transformer has ~300K params vs our current ~70K actor MLP -- a 4x increase. Overkill for short-horizon pedestrian dynamics where the relevant temporal horizon is a few seconds. **Revisit for Tier 4-5 multi-floor evacuation** where agents must remember which corridor was a dead end.

### 2.8 What pedestrian simulation does today (nothing)

This is a key finding: **no major pedestrian simulation approach includes self-trajectory memory**.

- **Social Force Model** (Helbing & Molnar 1995) and extensions (HSFM, Farina et al. 2017): Purely reactive. No trajectory memory. Deadlocks handled by tangential force components.
- **ORCA** (Van den Berg et al. 2011) and extensions: Purely reactive. Deadlocks handled via topological constraints (Topology-Guided ORCA, 2024) or ad-hoc perturbation strategies.
- **Karamouzas predictive model** (2009): Uses anticipated time-to-collision -- a form of *neighbor* trajectory prediction, not self-memory.
- **Lee et al. (2018) "Crowd Simulation by Deep RL"**: DDPG with polar dynamics. No trajectory memory. Emergent lane formation from reactive policy alone.
- **Martinez-Gil et al. (2017) MARL-Ped**: Individual MARL agents, scales to 10x training population. No trajectory memory.
- **Li et al. (2025) "Efficient crowd simulation with DRL"**: Combines DRL with anisotropic fields for global navigation. Resolves local deadlocks via AF embedding, not memory.
- **CrowdNav / SARL / CADRL** (Chen et al. 2019): Social attention over neighbor features. No position history.

**Implication**: Adding even simple trajectory memory signals would be a differentiator in the pedestrian simulation literature. The closest work is GRU-MAPPO for crowd evacuation (implicit memory via recurrent hidden state), but no explicit scalar trajectory features.

---

## 3. Design options for CrowdRL

### Option A: Temporal scalar signals (+6D observation)

Add 6 cheap scalar features. No architecture change, trivially GPU-parallel, ONNX-transparent.

| Signal | Dims | Formula | Tells the policy |
|--------|------|---------|------------------|
| `displacement_from_spawn` | 1 | `\|\|pos_t - pos_0\|\| / initial_goal_dist` | Global progress |
| `cumulative_path_length` | 1 | `sum(\|\|pos_i - pos_{i-1}\|\|) / initial_goal_dist` | Total distance (normalised) |
| `path_efficiency` | 1 | `disp / max(cum_path, eps)` | 1.0 = straight, ~0 = looping |
| `elapsed_fraction` | 1 | `step / max_steps` | Time pressure |
| `displacement_over_window` | 1 | `\|\|pos_t - pos_{t-W}\|\| / (W * v_pref * dt)` | Recent movement vs expected |
| `goal_progress_over_window` | 1 | `(d_{t-W} - d_t) / (W * v_pref * dt)` | Recent goal approach rate |

Window W = 50 steps (0.5s). Per-agent state: spawn pos (2), cum_path (1), ring buffers (50x2 + 50x1) = ~153 floats = ~612 bytes/agent. Obs: 79D -> 85D.

**Where it fits in CrowdRL architecture**: Temporal scalars are pre-computed in CrowdEnv and placed onto WorldState as optional fields. The observation builder in crowdrl-core reads them if `use_temporal=True`. JuPedSim adapter would do the same accumulation. Transfer guarantee preserved.

### Option B: Ego-state history (+14D observation)

Stack the previous K=2 ego states (7D each) alongside the full current observation. Only ego features (not social/raycast) because those change slowly and stacking them wastes dimensions.

Total obs: 85D (with Option A) + 14D = 99D. Per-agent cost: 2x7 = 14 extra floats. Gives MLP implicit access to acceleration and angular velocity without reward-system derivation.

**Where it fits**: Ego history is maintained as a ring buffer in CrowdEnv. Past ego states placed onto WorldState as optional `ego_history` field. Obs builder appends if `use_ego_history=True`.

### Option C: GRU recurrent policy (architecture change)

GRU(hidden_dim=64) between MLP feature extractor and output head. Following rMAPPO (Yu et al. 2022).

**What changes**:
- `RolloutBuffer`: `FlatBatch` -> chunked sequences `(n_sequences, chunk_length, obs_dim)` with T-BPTT (chunk_length=10)
- `EpisodeFactory` / training loop: hidden state management, reset policy
- `ActorCritic`: new `RecurrentActor` / `RecurrentCritic` variants
- ONNX export: explicit hidden state I/O
- JuPedSim adapter: maintain `(n_agents, 64)` hidden state tensor

**When it makes sense**: If ablation of A+B shows agents failing in scenarios requiring memory >0.5s (e.g. Tier 3-4 multi-room navigation, remembering dead-end corridors).

### Option D: Spatial visit counts (observation or reward augmentation)

Discretise walkable area into 0.5m grid cells. Maintain per-agent visit count tensor. Extract scalar features: `[visit_count_at_cell, mean_count_in_neighbourhood]`. Or use as intrinsic reward: `r_explore = 1/sqrt(N(cell))`.

**Where it fits**: Grid maintained in CrowdEnv. Scalar features placed on WorldState. Decay factor 0.99/step gives recency weighting.

**When it makes sense**: If agents exhibit severe circling behaviour not captured by path_efficiency (which should be rare).

---

## 4. Approaches ruled out

| Approach | Why not |
|----------|---------|
| Egocentric occupancy grids | 256+ floats/agent, complex rotation logic, raycasts+navmesh already cover this |
| ICM / RND curiosity | 2-3 extra forward passes per agent per step, prohibitive at 100+ agents |
| Transformers | O(K^2) attention, 4x parameter increase, overkill for short-horizon dynamics |
| Full frame stacking (all dims) | K=3 with 85D = 255D input; ego-only stacking captures what matters |
| Learned adaptive stacking | Research-stage (Nangue 2025), adds meta-learning complexity |

---

## 5. Recommended sequencing

1. **Start with Option A** (temporal scalars). Cheapest, highest information-per-dimension ratio. The `path_efficiency` signal alone is likely worth more than a GRU for detecting stagnation.
2. **Ablate Option B** (ego history) alongside A. Compare A-only vs A+B on corridor/bottleneck scenarios. If B doesn't help, drop it to keep obs small.
3. **Option C** (GRU) only if agents demonstrably fail on tasks requiring long-horizon memory after A+B. This is a significant engineering investment.
4. **Option D** (visit counts) only if circling is a persistent pathology not caught by path_efficiency.

---

## 6. Complementary reward signal

Independent of observation changes, a stagnation penalty in the reward can help:

```
stagnation_penalty = -c * max(0, threshold - displacement_over_window)
```

This is complementary: the observation tells the policy it's stuck (so it can learn to avoid stagnation), while the reward penalises staying stuck (providing immediate gradient signal). The observation is more important for long-term learning; the reward helps bootstrap early training.

---

## 7. References

### Recurrent MARL
- Yu et al. (2022) "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" [arXiv:2103.01955](https://arxiv.org/abs/2103.01955)
- BAIR blog: [MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/)
- [marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy) (official rMAPPO)
- [MarcoMeter/recurrent-ppo-truncated-bptt](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt)
- "Generalization, Mayhems and Limits in Recurrent PPO" [arXiv:2205.11104](https://arxiv.org/abs/2205.11104)

### Frame stacking / observation history
- Radosavovic et al. (2023) "Real-World Humanoid Locomotion with Reinforcement Learning" [arXiv:2303.03381](https://arxiv.org/abs/2303.03381)
- [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) (Improbable-AI/MIT)
- [legged_gym](https://github.com/leggedrobotics/legged_gym) (leggedrobotics)
- Nangue et al. (2025) "Finding the FrameStack: Learning What to Remember" [OpenReview](https://openreview.net/forum?id=W7O95Q1qzT)

### Spatial memory / navigation
- Chaplot et al. (2020) "Learning to Explore using Active Neural SLAM" [arXiv:2004.05155](https://arxiv.org/abs/2004.05155)
- Gupta et al. (2017) "Cognitive Mapping and Planning for Visual Navigation" [arXiv:1702.03920](https://arxiv.org/abs/1702.03920)
- Yang et al. (2025) "Spatially-Enhanced Recurrent Memory for Long-Range Mapless Navigation via End-to-End RL" [arXiv:2506.05997](https://arxiv.org/abs/2506.05997)

### Exploration / curiosity
- Tang et al. (2017) "#Exploration: Count-Based Exploration for Deep RL" [arXiv:1611.04717](https://arxiv.org/abs/1611.04717)
- Pathak et al. (2017) "Curiosity-driven Exploration by Self-supervised Prediction" (ICM, ICML) [paper](https://pathak22.github.io/noreward-rl/resources/icml17.pdf)
- Burda et al. (2019) "Exploration by Random Network Distillation" [arXiv:1810.12894](https://arxiv.org/abs/1810.12894)
- Schaefer (2019) "Curiosity in Multi-Agent Reinforcement Learning" MSc thesis, University of Edinburgh (original URL dead; available via [ResearchGate](https://www.researchgate.net/publication/336579400))
- Lilian Weng: [Exploration Strategies in Deep RL](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/)

### Transformer memory
- Chen et al. (2021) "Decision Transformer: Reinforcement Learning via Sequence Modeling" (NeurIPS) [arXiv:2106.01345](https://arxiv.org/abs/2106.01345)
- Meng et al. (2021) "Offline Pre-trained Multi-Agent Decision Transformer" [arXiv:2112.02845](https://arxiv.org/abs/2112.02845)
- Cherepanov et al. (2023) "Recurrent Action Transformer with Memory" [arXiv:2306.09459](https://arxiv.org/abs/2306.09459)
- Yang et al. (2022) "Transformer-based Working Memory for Multiagent RL with Action Parsing" (NeurIPS)

### Pedestrian simulation
- Lee et al. (2018) "Crowd Simulation by Deep RL" [ACM](https://dl.acm.org/doi/10.1145/3274247.3274510)
- Martinez-Gil et al. (2017) "Emergent behaviors and scalability for MARL-Ped" [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1569190X17300503)
- Li et al. (2025) "Efficient crowd simulation with DRL" [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11825876/)
- [CrowdNav](https://github.com/vita-epfl/CrowdNav) (Chen et al. 2019)
- Farina et al. (2017) "Walking Ahead: The Headed Social Force Model" [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0169734)

### Reward shaping / stuck detection
- Ng et al. (1999) "Potential-Based Reward Shaping"
- Nikolov (2011) "Stuck detection for mobile robots" [blog](https://snikolov.wordpress.com/2011/02/27/going-nowhere-fast-how-to-tell-if-your-robot-is-stuck-and-what-to-do-about-it/)
- Ma et al. (2019) "Self-Monitoring Navigation Agent via Auxiliary Progress Estimation" (ICLR) [arXiv:1901.03035](https://arxiv.org/abs/1901.03035)

### GPU-parallel RL environments
- [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) (NVIDIA)
- Makoviychuk et al. (2021) "Isaac Gym" [arXiv:2108.10470](https://arxiv.org/abs/2108.10470)
