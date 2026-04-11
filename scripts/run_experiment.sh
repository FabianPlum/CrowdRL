#!/usr/bin/env bash
# run_experiment.sh <experiment_name>
#
# Archives the current config + git rev, runs training with full stdout/stderr
# capture, then archives the entire results_full_training directory under a
# descriptive name. Designed to be unattended and idempotent-ish:
#
#   - Fails fast if the target archive name already exists.
#   - Writes training log via tee so we can read it mid-run or post-mortem.
#
# After the run finishes, use scripts/analyze_run.py on the archived directory.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <experiment_name>" >&2
    exit 2
fi

EXP_NAME="$1"
ARCHIVE_DIR="results_${EXP_NAME}"
CONFIG_PATH="configs/full_training.yaml"
RESULTS_DIR="results_full_training"

if [[ -e "$ARCHIVE_DIR" ]]; then
    echo "error: archive dir $ARCHIVE_DIR already exists" >&2
    exit 1
fi

# Snapshot config + code state BEFORE running.
mkdir -p "$ARCHIVE_DIR"
cp "$CONFIG_PATH" "$ARCHIVE_DIR/config.yaml.snapshot"
git rev-parse HEAD > "$ARCHIVE_DIR/git_rev.txt" 2>/dev/null || echo "unknown" > "$ARCHIVE_DIR/git_rev.txt"
date --iso-8601=seconds > "$ARCHIVE_DIR/start_time.txt"

echo "=== Running experiment: $EXP_NAME ===" | tee "$ARCHIVE_DIR/training.log"
echo "Config snapshot: $ARCHIVE_DIR/config.yaml.snapshot" | tee -a "$ARCHIVE_DIR/training.log"
echo "Git rev: $(cat $ARCHIVE_DIR/git_rev.txt)" | tee -a "$ARCHIVE_DIR/training.log"
echo "Start: $(cat $ARCHIVE_DIR/start_time.txt)" | tee -a "$ARCHIVE_DIR/training.log"
echo "---" | tee -a "$ARCHIVE_DIR/training.log"

# Run training (exit status propagated via set -e and tee's pipefail).
uv run python train_mappo.py --config "$CONFIG_PATH" 2>&1 | tee -a "$ARCHIVE_DIR/training.log"

date --iso-8601=seconds > "$ARCHIVE_DIR/end_time.txt"
echo "End: $(cat $ARCHIVE_DIR/end_time.txt)" | tee -a "$ARCHIVE_DIR/training.log"

# Archive the full results_full_training directory contents (history.json,
# checkpoint_final.pt, training_curves.png, evaluation.png, trajectories.png,
# policy.onnx, episode.mp4, config_resolved.yaml).
if [[ -d "$RESULTS_DIR" ]]; then
    cp -r "$RESULTS_DIR"/. "$ARCHIVE_DIR"/
    echo "Archived $RESULTS_DIR -> $ARCHIVE_DIR"
else
    echo "warning: $RESULTS_DIR not found post-run" >&2
fi

echo "=== Done: $EXP_NAME ==="
