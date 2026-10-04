#!/usr/bin/env bash
# Run from an activated environment with requirements.txt installed.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$(dirname "$PROJECT_DIR")${PYTHONPATH:+:$PYTHONPATH}"

# Experiment II: paired censored/uncensored Cases II and III.
OUT="${OUT:-/tmp/joint_softcomp/full/uncensored}"
REPS="${REPS:-10}"

if ! [[ "$REPS" =~ ^[1-9][0-9]*$ ]] || [ "$REPS" -lt 2 ]; then
  echo "REPS must be an integer >= 2 (sample standard deviations need two repeats)." >&2
  exit 2
fi
mkdir -p "$OUT"

for ((r=0; r<REPS; r++)); do
  for case_id in 2 3; do
    for method in JointSoftComp SoftComp SoftComp-noaug NeuralFG; do
      python -m competing_risks.experiments.joint_softcomp.uncensored --case "$case_id" --method "$method" --replicate "$r" --threads 1 --out-dir "$OUT/censored"
      python -m competing_risks.experiments.joint_softcomp.uncensored --case "$case_id" --method "$method" --replicate "$r" --threads 1 --out-dir "$OUT/uncensored" --uncensored
    done
  done
done
python -m competing_risks.experiments.joint_softcomp.analyze_uncensored "$OUT/censored" "$OUT/uncensored" "$REPS" 2>&1 | tee "$OUT/summary.txt"
