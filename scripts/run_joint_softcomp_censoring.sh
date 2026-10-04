#!/usr/bin/env bash
# Run from an activated environment with requirements.txt installed.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$(dirname "$PROJECT_DIR")${PYTHONPATH:+:$PYTHONPATH}"

# Experiment III: fixed-horizon censoring, AJ checks, and population targets.
OUT="${OUT:-/tmp/joint_softcomp/full/censoring}"
REPS="${REPS:-5}"

if ! [[ "$REPS" =~ ^[1-9][0-9]*$ ]] || [ "$REPS" -lt 2 ]; then
  echo "REPS must be an integer >= 2 (sample standard deviations need two repeats)." >&2
  exit 2
fi
mkdir -p "$OUT"

for rho in 0 0.2 0.5 0.8; do
  for method in softcomp joint; do
    python -m competing_risks.experiments.joint_softcomp.censoring_levels aj-check --method "$method" --rho "$rho" --seed 0 --n-train 10000 --threads 1 --out-dir "$OUT"
    for ((seed=0; seed<REPS; seed++)); do
      python -m competing_risks.experiments.joint_softcomp.censoring_levels case3 --method "$method" --rho "$rho" --seed "$seed" --threads 1 --out-dir "$OUT"
      python -m competing_risks.experiments.joint_softcomp.censoring_levels constant --method "$method" --rho "$rho" --seed "$seed" --threads 1 --out-dir "$OUT"
      if [ "$rho" != 0 ]; then
        python -m competing_risks.experiments.joint_softcomp.censoring_levels depcens --method "$method" --rho "$rho" --seed "$seed" --threads 1 --out-dir "$OUT"
      fi
    done
  done
done
python -m competing_risks.experiments.joint_softcomp.censoring_levels summarize --out-dir "$OUT" 2>&1 | tee "$OUT/summary.txt"
python -m competing_risks.experiments.joint_softcomp.population_targets 2>&1 | tee "$OUT/population_targets.txt"
