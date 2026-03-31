# MAPPO / PPO Implementation Best Practices — Literature Review

**Compiled**: March 2026
**Purpose**: Ground every design decision in `crowdrl-train` with empirical evidence.
**Context**: MAPPO with parameter sharing, continuous 4D action space, 20–100 homogeneous agents, variable agent counts, procedural environments.

---

## Key Sources

| # | Paper | Venue | Focus |
|---|-------|-------|-------|
| 1 | Yu, Velu, Vinitsky et al. — "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games" (2022) | NeurIPS 2022 D&B | MAPPO: canonical reference for multi-agent PPO. [arXiv:2103.01955](https://arxiv.org/abs/2103.01955) |
| 2 | Andrychowicz, Raichuk et al. — "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study" (2021) | — | 250k-experiment ablation of PPO implementation details. [arXiv:2006.05990](https://arxiv.org/abs/2006.05990) |
| 3 | Huang, Dossa et al. — "The 37 Implementation Details of Proximal Policy Optimization" (2022) | ICLR Blog Track | Enumerates and tests every code-level PPO detail. [Blog](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) |
| 4 | Terry, Grammel, Son, Black, Agrawal — "Revisiting Parameter Sharing in Multi-Agent Deep Reinforcement Learning" (2020) | — | When parameter sharing helps vs hurts. [arXiv:2005.13625](https://arxiv.org/abs/2005.13625) |
| 5 | Li et al. — "Adaptive Parameter Sharing for Multi-Agent RL" (2023) | — | Heterogeneous-agent parameter sharing strategies. [arXiv:2312.09009](https://arxiv.org/abs/2312.09009) |
| 6 | Narvekar et al. — "Curriculum Learning for RL Domains: A Framework and Survey" (2020) | JMLR 21 | Comprehensive curriculum learning survey. [Paper](https://jmlr.org/papers/volume21/20-212/20-212.pdf) |
| 7 | Zhao, Li, Pajarinen — "Learning Progress Driven Multi-Agent Curriculum" (2022) | — | Automatic curriculum via TD-error learning progress. [arXiv:2205.10016](https://arxiv.org/abs/2205.10016) |
| 8 | liuliu — "5 More Implementation Details of PPO and SAC" (2022) | Blog | Additional PPO continuous-control details. [Blog](https://liuliu.me/eyes/5-more-implementation-details-of-ppo-and-sac/) |

---

## 1. Network Architecture

### Separate actor and critic networks (not shared trunk)

**Evidence**: Andrychowicz et al. [2] found separate networks superior on 4 of 5 continuous control environments. MAPPO [1] uses separate actor-critic by default across all benchmarks. Huang et al. [3] (detail #26) recommend separate MLPs for continuous control.

**Rationale**: The actor and critic solve different functions — the actor maps observations to actions, while the critic maps observations (or global state) to scalar value estimates. Sharing a trunk forces a representational compromise between these two objectives, which empirically hurts continuous control performance. The value function often benefits from wider layers or different feature representations than the policy.

**Recommendation**: Separate `Actor` and `Critic` modules with independent hidden layers. This also enables using different input representations for each (see Section 5 on critic input).

### Two hidden layers with tanh activation

**Evidence**: Andrychowicz et al. [2] tested across multiple depths and widths. Two hidden layers were optimal for both policy and value networks across continuous control tasks. tanh outperformed ReLU on 4 of 5 environments. MAPPO [1] defaults to 64-unit layers (sufficient for their benchmarks but likely undersized for our 79D observation space).

**Rationale**: Deeper networks did not improve performance and increased training instability. tanh provides bounded activations that naturally constrain hidden representations, which is helpful for PPO's trust-region-like updates. ReLU is the MAPPO default, but this is primarily tested in discrete-action domains (SMAC, Hanabi) — the continuous control evidence favours tanh.

**Recommendation**: Actor: 2 hidden layers of 256 units each (scaled up from MAPPO's 64 due to our larger obs space). Critic: 2 hidden layers of 256 units. tanh activations throughout. Layer norm optional — test in ablation.

### Orthogonal initialization with specific per-layer scaling

**Evidence**: Huang et al. [3] (details #12–13): orthogonal init with gain `sqrt(2)` for hidden layers, `0.01` for policy output, `1.0` for value output, biases to 0. Andrychowicz et al. [2] independently confirm: "Initialize the last policy layer with 100x smaller weights."

**Rationale**: Small initial policy weights ensure the initial action distribution is near-zero mean with moderate variance — the policy starts by exploring uniformly rather than committing to large actions. The `sqrt(2)` gain for hidden layers compensates for the variance-reducing effect of tanh activations. Value output at gain 1.0 allows the value head to fit reward scales quickly.

**Recommendation**: Follow the Huang et al. prescription exactly. This is one of the few details where the literature is fully consistent.

---

## 2. Action Distribution (Continuous Control)

### State-independent learnable log-std

**Evidence**: Huang et al. [3] (detail #24): "Standard deviation is a learnable parameter (not state-dependent), initialized to 0." This is the default in CleanRL, Stable Baselines3 [7], and MAPPO [1]. Andrychowicz et al. [2] found initial std of ~0.5 optimal across most environments.

**Rationale**: State-dependent std (as in SAC) adds a network head that maps observations to per-dimension standard deviations. For on-policy methods like PPO, this additional complexity provides no clear benefit — the policy can modulate exploration through the mean alone, and the global std captures the overall exploration-exploitation tradeoff. State-dependent std is an off-policy technique (SAC) that adds training noise to PPO.

**Recommendation**: `log_std = nn.Parameter(torch.full((action_dim,), log(0.5)))` — a single learnable vector of shape `(action_dim,)`, initialized to `log(0.5) ~ -0.69` to produce initial std ~ 0.5 per Andrychowicz et al.

### Clip actions to bounds, do NOT use tanh squashing

**Evidence**: Huang et al. [3] (detail #27): clip sampled actions to environment bounds. liuliu [8]: "Clipping to [-1, 1] then scaling proved more stable on all MuJoCo tasks." Tanh squashing is a SAC-specific technique that requires log-probability corrections (the Jacobian of the tanh transform) and is unnecessary for PPO.

**Rationale**: PPO computes policy gradients using the ratio of new/old log-probabilities. Tanh squashing changes the distribution's support and requires a correction term in the log-prob computation. This adds implementation complexity and a source of numerical instability (log-prob of a near-boundary action diverges). Clipping is simpler and more numerically stable.

**Critical detail**: Store the *unclipped* action sample for log-probability computation. Use the *clipped* action for the environment step. This ensures the gradient signal is correct even when actions saturate at the bounds.

---

## 3. PPO Hyperparameters

### Clipping epsilon: 0.2, lower for harder multi-agent tasks

**Evidence**: Yu et al. [1]: "Maintain a clipping ratio epsilon under 0.2." For hardest SMAC maps, reduced to 0.05. Andrychowicz et al. [2] suggest 0.25 as starting point for single-agent continuous control.

**Rationale**: In multi-agent settings, non-stationarity is higher because each agent's environment changes as other agents' policies update. Larger policy steps amplify this instability. A lower clipping epsilon constrains per-update policy change, which is more important in MARL than single-agent RL.

**Recommendation**: Start at 0.2 for easy curriculum phases. Consider reducing to 0.1–0.15 for higher-density/harder phases. This is an ablation-worthy parameter.

### PPO epochs: 5–10, fewer for harder settings

**Evidence**: Yu et al. [1]: "15 epochs for easy tasks, 10 or 5 for difficult tasks." They explicitly note that epoch count "potentially controls the challenge of non-stationarity in MARL" — excessive data reuse makes other agents' transitions stale faster.

**Rationale**: Each epoch re-processes the same rollout data. In single-agent RL, more epochs improve sample efficiency. In MARL, the data was collected under joint behaviour of all agents; after updating one agent's policy, the data no longer reflects the true transition dynamics. Too many epochs overfit to stale joint dynamics.

**Recommendation**: 10 epochs as default. Reduce to 5 if training becomes unstable at high agent counts.

### Mini-batches: 1 (full-batch update)

**Evidence**: Yu et al. [1]: "Avoid splitting data into mini-batches" — their default `num_mini_batch = 1`. This is specific to multi-agent settings where batch coherence matters.

**Rationale**: Mini-batch splitting introduces additional stochasticity in gradient estimates. In single-agent RL, this is beneficial (reduces overfitting to the batch). In MARL with parameter sharing, the batch already contains diverse experiences from multiple agents, providing natural gradient diversity. Splitting further adds noise without benefit.

**Practical caveat**: With 50–100 agents × 500 steps, each rollout produces 25k–50k agent-steps. This fits comfortably in GPU memory as a single batch. If memory becomes a constraint at very high agent counts, splitting into 2–4 mini-batches is acceptable.

**Recommendation**: Full-batch updates (`n_minibatches = 1`). Only split if GPU memory requires it.

### Learning rate: 5e-4, Adam with eps=1e-5

**Evidence**: Yu et al. [1]: 5e-4 to 7e-4 depending on domain. Andrychowicz et al. [2]: "3e-4 is a safe default." Huang et al. [3]: 3e-4 for MuJoCo, with linear annealing to 0. MAPPO uses Adam with eps=1e-5 (not PyTorch default 1e-8).

**Rationale**: The Adam epsilon parameter affects the denominator of the update rule. A larger epsilon (1e-5) provides more conservative updates when gradient second moments are small, improving stability. Linear LR annealing ensures the policy converges rather than oscillating at the end of training.

**Recommendation**: 5e-4 initial LR with linear decay to 0. Adam optimizer with eps=1e-5, default betas (0.9, 0.999). Optionally different LR for actor and critic (MAPPO sometimes uses higher critic LR, e.g., 1e-3 for critic vs 5e-4 for actor).

### Entropy coefficient: 0.01, possibly anneal to 0

**Evidence**: Yu et al. [1]: default 0.01 across domains. Andrychowicz et al. [2]: "We do not find evidence that any of the investigated regularizers helps significantly" for continuous control. Huang et al. [3]: entropy coeff 0.0 for MuJoCo continuous control, 0.01 for Atari discrete.

**Rationale**: In continuous action spaces, the Gaussian policy already maintains exploration through its std parameter. Entropy regularization is more important in discrete action spaces where the policy can collapse to a deterministic choice. In continuous spaces, the learnable log_std provides a dedicated exploration mechanism. A small entropy bonus (0.01) adds marginal benefit for preventing premature std collapse.

**Recommendation**: 0.01 initially, anneal toward 0 in later training. Not a high-priority hyperparameter for continuous control.

### Gradient clipping: max_grad_norm = 10.0

**Evidence**: Yu et al. [1]: `max_grad_norm = 10.0`. Huang et al. [3] (detail #11): `max_grad_norm = 0.5`. Andrychowicz et al. [2]: "Gradient clipping might slightly help but is of secondary importance."

**Contradiction**: MAPPO uses 10.0 (very permissive); CleanRL uses 0.5 (aggressive). The MAPPO value is tested in multi-agent settings where many agents contribute diverse gradients, potentially requiring less aggressive clipping.

**Recommendation**: 10.0 (MAPPO default). Monitor gradient norms during training. If norms consistently stay below 1.0, the exact threshold is irrelevant.

### GAE: gamma=0.99, lambda=0.95

**Evidence**: Yu et al. [1]: gamma=0.99, lambda=0.95 across all domains. Andrychowicz et al. [2]: "Use GAE with lambda=0.9." gamma=0.99 agreed upon by both.

**Rationale**: lambda controls the bias-variance tradeoff in advantage estimation. Higher lambda (0.95) gives lower bias but higher variance — better for longer-horizon tasks. Our episodes are up to 500 steps (50 seconds of simulated time), which is moderately long-horizon, justifying lambda=0.95 over 0.9.

**Recommendation**: gamma=0.99, lambda=0.95.

---

## 4. Normalization (Critical for Stability)

### Observation normalization: always on, running mean/std

**Evidence**: Andrychowicz et al. [2]: "Always use observation normalization." Essential for performance on virtually all tested environments. Huang et al. [3] (detail #28): "Critical for MuJoCo performance." Clip normalized observations to [-10, 10] (detail #29).

**Rationale**: Different observation components have vastly different scales (e.g., raycasts in [0, 1], velocities in m/s, goal direction as unit vectors). Without normalization, the network must learn to handle multi-scale inputs, wasting capacity. Running normalization adapts to the actual data distribution encountered during training.

**For MARL with parameter sharing**: Since all agents share one network, use shared running statistics. All active agent observations contribute to the same normalizer. This is natural and efficient.

**Recommendation**: Welford's online algorithm for running mean/var. Normalize: `clip((obs - mean) / sqrt(var + 1e-8), -10, 10)`. Update statistics with every batch of active agent observations. Freeze statistics for ONNX export.

### Value normalization: use PopArt or ValueNorm

**Evidence**: Yu et al. [1]: "Value normalization often helps and never hurts MAPPO's performance." They use PopArt by default across all domains. This is one of the few details where they observed consistent improvement.

**Rationale**: The value function must predict returns, which can span a wide range depending on episode length, reward scale, and discount factor. Without normalization, the value network's output layer must adapt to the reward scale, which can cause gradient instability. PopArt solves this by normalizing value targets and correcting the network's output layer when statistics change.

**PopArt vs running-mean ValueNorm**: PopArt (Hessel et al., 2019) preserves the value function's predictions exactly when updating normalization statistics — it adjusts the output layer weights/biases to compensate. Running-mean ValueNorm is simpler but introduces small prediction errors when statistics shift. For long training runs with curriculum (where reward distributions shift as difficulty changes), PopArt is more robust.

**Recommendation**: Implement ValueNorm (simpler) first. Consider PopArt if curriculum phase transitions cause value function instability.

### Reward normalization: divide by running std (no mean subtraction)

**Evidence**: Yu et al. [1]: reward normalization enabled by default. Huang et al. [3] (detail #30): divide by running std of returns, no mean subtraction, clip to [-10, 10].

**Rationale**: Mean subtraction changes the optimal policy (it shifts which states appear "good" vs "bad" in absolute terms). Variance normalization preserves the relative ordering of returns while bringing them to a consistent scale, which stabilizes value function learning.

**Recommendation**: Track running std of per-agent returns. Divide rewards by this std before feeding to GAE/value computation. No mean subtraction.

### Advantage normalization: per-batch, zero mean, unit variance

**Evidence**: Yu et al. [1]: standard practice. Huang et al. [3] (detail #7): normalize per mini-batch. Andrychowicz et al. [2]: "seems not to affect performance too much."

**Recommendation**: Normalize across the full batch (since we use `n_minibatches=1`). Low priority — include but do not spend ablation time on this.

---

## 5. Critic Architecture in MAPPO (CTDE)

### Agent-specific feature-pruned (FP) global state

**Evidence**: Yu et al. [1] tested four critic input representations:
- **CL (Concatenated Local)**: Concatenate all agents' observations. Dimensionality explodes with agent count. Poor performance.
- **EP (Environment-Provided global state)**: Use only global features. Lacks agent-specific context. Suboptimal.
- **AS (Agent-Specific)**: Global state + agent's own local observation. Strong performance.
- **FP (Feature-Pruned)**: AS with redundant features removed (e.g., if the agent's position appears in both local obs and global state, include it only once). Best performance.

Omitting local agent information from the critic is "highly detrimental."

**Rationale for CrowdRL**: In Centralized Training with Decentralized Execution (CTDE), the critic can use privileged global information during training that the actor cannot access during deployment. This reduces the partial observability of the value estimation problem. The actor only sees local egocentric observations (as required for deployment in JuPedSim).

**Recommendation for CrowdRL**: The critic receives the agent's own observation (79D) augmented with compact global context: mean crowd density, number of active agents, geometry tier indicator. Remove any features duplicated between local obs and global context. The actor receives only the local 79D observation.

**Important**: This means the actor and critic have different input dimensions — another reason to use separate networks.

---

## 6. Parameter Sharing

### Full sharing is appropriate for homogeneous agent populations

**Evidence**: Yu et al. [1]: parameter sharing is the standard MAPPO approach. Terry et al. [4]: full sharing with one-hot agent ID works well for homogeneous agents. Li et al. [5]: full sharing struggles with genuinely heterogeneous agents (different action spaces or fundamentally different roles).

**For CrowdRL**: Agents are functionally homogeneous — same dynamics, same observation structure, same action space. Heterogeneity enters through body dimensions and preferred speed, which are already in the observation (social sensing includes body dims). No agent-ID conditioning is needed — the body dims themselves serve as the "identity" signal.

**Recommendation**: Full parameter sharing, no agent-ID augmentation. Body dimensions and preferred speed in the observation naturally differentiate agents.

---

## 7. Variable Agent Count Handling

### Pad to fixed maximum with active-agent masking

**Evidence**: This is standard practice in multi-agent RL implementations (MAPPO codebase, EPyMARL, MARLlib). Variable-count episodes are handled by:
1. Defining a maximum agent count
2. Padding observations/actions to this maximum
3. Masking padded agents out of loss computation and statistics

**Alternative for parameter-shared PPO**: Since all agents share one network, agent identity doesn't matter. Flatten all active agent-steps across an episode into a single batch. No padding needed — just concatenate valid transitions.

**Recommendation**: Use the flattening approach. During rollout collection, store per-timestep arrays with the actual agent count. During `flatten()`, concatenate only active agent-steps into a single `(total_active_steps, ...)` batch. This is memory-efficient and avoids wasted computation on padded zeros.

---

## 8. Curriculum Learning

### Success-rate-driven phase advancement with difficulty mixing

**Evidence**: Narvekar et al. [6] (JMLR survey): the Zone of Proximal Development (ZPD) principle — present tasks of moderate difficulty. OpenAI (2019, Rubik's Cube): Automatic Domain Randomization (ADR) advances when success rate exceeds a threshold (typically 70–80%). Zhao et al. [7]: TD-error-based learning progress is a principled automatic alternative.

**Key findings for multi-agent curriculum**:
- Agent count is itself a curriculum variable — start with few agents, increase gradually (Zhao et al. [7]).
- Maintain 20–30% episodes from earlier difficulty levels to prevent catastrophic forgetting.
- No universal threshold exists — domain-specific tuning is required.

**Recommendation for CrowdRL**:
1. Four phases: Tier 0 easy → Tier 0–1 medium → Tier 1–2 hard → Tier 0–2 full.
2. Advance when rolling-window goal rate (last 50 episodes) exceeds phase-specific threshold.
3. Increase agent count gradually within each phase before advancing tier.
4. Mix in 20% episodes from earlier phases after advancement.
5. Consider TD-error-based automatic curriculum as a future refinement.

---

## 9. Value Function Loss

### Use MSE, NOT PPO-style value clipping

**Evidence**: Andrychowicz et al. [2]: value loss clipping "hurt the performance regardless of the clipping threshold." Explicit recommendation to avoid. Huang et al. [3] (detail #9): listed as optional, "recent work suggests it may hurt."

**Rationale**: PPO-style value clipping was originally proposed by analogy with the policy clipping — constrain value function updates to stay close to previous predictions. However, the value function does not suffer from the same trust-region concerns as the policy (large value updates do not cause action distribution shift). Clipping the value loss simply prevents the value function from fitting well, degrading GAE quality.

**Huber loss**: Yu et al. [1] use Huber loss with delta=10.0 (which approximates MSE for most values but is robust to rare large errors). Andrychowicz et al. [2] recommend plain MSE. Both are acceptable; Huber with large delta is a conservative middle ground.

**Recommendation**: MSE loss for the value function. No clipping. Consider Huber (delta=10.0) if value loss shows occasional large spikes.

---

## 10. Summary: Recommended Configuration for CrowdRL

Based on the above literature, here is the evidence-based starting configuration:

| Decision | Choice | Primary Source | Confidence |
|----------|--------|---------------|------------|
| **Architecture** | Separate actor (256, 256) and critic (256, 256) | Andrychowicz [2] | High |
| **Activation** | tanh | Andrychowicz [2] | High |
| **Initialization** | Orthogonal (sqrt(2) hidden, 0.01 actor out, 1.0 critic out) | Huang [3] | High |
| **Action distribution** | Diagonal Gaussian, state-independent log_std, init log(0.5) | Andrychowicz [2], Huang [3] | High |
| **Action bounding** | Clip to [-1,1], store unclipped for log-prob | Huang [3], liuliu [8] | High |
| **Clip epsilon** | 0.2 (reduce for hard phases) | Yu [1] | High |
| **PPO epochs** | 10 (reduce to 5 for hard settings) | Yu [1] | Medium-High |
| **Mini-batches** | 1 (full-batch) | Yu [1] | Medium |
| **Learning rate** | 5e-4, Adam eps=1e-5, linear decay | Yu [1], Huang [3] | High |
| **Gamma / Lambda** | 0.99 / 0.95 | Yu [1] | High |
| **Entropy coeff** | 0.01, anneal to 0 | Yu [1] | Low (not critical) |
| **Max grad norm** | 10.0 | Yu [1] | Low (secondary) |
| **Obs normalization** | Running mean/std, clip [-10, 10] | Andrychowicz [2] | High |
| **Value normalization** | ValueNorm (running std) | Yu [1] | High |
| **Reward normalization** | Divide by running std (no mean subtraction) | Yu [1], Huang [3] | Medium-High |
| **Value loss** | MSE, no clipping | Andrychowicz [2] | High |
| **Advantage normalization** | Per-batch zero-mean unit-var | Yu [1] | Medium |
| **Critic input** | Agent obs + compact global context (FP) | Yu [1] | High |
| **Parameter sharing** | Full, no agent-ID (body dims serve as identity) | Yu [1], Terry [4] | High |
| **Variable agents** | Flatten active agent-steps, mask inactive | Standard practice | High |
| **Curriculum** | 4 phases, success-rate advancement, 20% history mixing | Narvekar [6], Zhao [7] | Medium |

### Key contradictions in the literature (to resolve via ablation)

1. **GAE lambda**: 0.95 (MAPPO) vs 0.9 (Andrychowicz). Starting with 0.95; ablate if needed.
2. **Grad norm clipping**: 10.0 (MAPPO) vs 0.5 (CleanRL). Starting with 10.0; monitor norms.
3. **Mini-batches**: 1 (MAPPO) vs 4+ (CleanRL). Starting with 1; split only if memory requires it.
4. **Entropy**: 0.01 (MAPPO) vs 0.0 (CleanRL continuous). Starting with 0.01; low priority.

---

## Appendix: Changes from Initial Plan

The literature review motivated the following changes from the initial plan draft:

| Initial Plan | Revised (Literature-Based) | Reason |
|-------------|---------------------------|--------|
| Shared actor-critic trunk | **Separate** actor and critic networks | Andrychowicz [2]: separate is better for continuous control |
| `n_minibatches = 4` | **`n_minibatches = 1`** (full batch) | Yu [1]: avoid mini-batch splitting in MARL |
| `max_grad_norm = 0.5` | **`max_grad_norm = 10.0`** | Yu [1]: MAPPO uses looser clipping |
| `lr = 3e-4` | **`lr = 5e-4`** | Yu [1]: MAPPO default, slightly higher |
| `log_std_init = -0.5` | **`log_std_init = log(0.5) ~ -0.69`** | Andrychowicz [2]: initial std ~0.5 optimal |
| Value loss clipping ON | **Value loss clipping OFF** | Andrychowicz [2]: hurts performance |
| Critic sees same obs as actor | **Critic sees obs + global context (CTDE)** | Yu [1]: FP global state best for MAPPO |
| No reward normalization | **Reward normalization ON** | Yu [1], Huang [3]: divide by running std |
| No value normalization | **Value normalization ON (ValueNorm)** | Yu [1]: "often helps, never hurts" |
