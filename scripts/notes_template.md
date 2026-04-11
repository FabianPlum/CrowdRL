# <experiment_name>

## Hypothesis
<one-sentence claim being tested>

## Configuration delta vs baseline
<exact lines changed in configs/full_training.yaml>

## Predicted effect
- Goal rate: <prediction, which tiers>
- Mean reward: <prediction>
- Episode length: <prediction>
- Failure modes to watch for: <what could go wrong>

## Observed result
### Run metadata
- Seed: <seed>
- Wall time: <seconds>
- Total episodes: <n>
- Total agent-steps: <m>

### Per-tier final window (last 2000 full-phase episodes)
<paste analyze_run.py output>

### Delta vs baseline
<paste compare output>

## Verdict
- [ ] Outside noise floor
- [ ] Within noise floor (inconclusive)
- [ ] Clear regression
- [ ] Clear improvement

<one-paragraph attribution: what caused the observed effect, any surprises>

## Next action
<what this result implies for subsequent experiments>
