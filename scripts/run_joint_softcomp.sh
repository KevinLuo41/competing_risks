#!/usr/bin/env bash
# Run from an activated environment with requirements.txt installed.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$(dirname "$PROJECT_DIR")${PYTHONPATH:+:$PYTHONPATH}"

# Experiment I: three beta values, references, and the full M sweep.
OUT="${OUT:-/tmp/joint_softcomp/full/simple}"
REPS="${REPS:-200}"

if ! [[ "$REPS" =~ ^[1-9][0-9]*$ ]] || [ "$REPS" -lt 2 ]; then
  echo "REPS must be an integer >= 2 (sample standard deviations need two repeats)." >&2
  exit 2
fi
mkdir -p "$OUT"

for beta in 0 0.4054651081 0.6931471806; do
  python -m competing_risks.experiments.joint_softcomp.simple reference --beta "$beta" --m 0 --reps "$REPS" --threads 1 --out-dir "$OUT"
  for m in 0 1 2 4 8; do
    python -m competing_risks.experiments.joint_softcomp.simple softcomp --beta "$beta" --m "$m" --reps "$REPS" --threads 1 --out-dir "$OUT"
  done
  for m in 1 2 4 8; do
    python -m competing_risks.experiments.joint_softcomp.simple joint --beta "$beta" --m "$m" --reps "$REPS" --threads 1 --out-dir "$OUT"
  done
done
python -m competing_risks.experiments.joint_softcomp.analyze_simple "$OUT" 2>&1 | tee "$OUT/summary.txt"
