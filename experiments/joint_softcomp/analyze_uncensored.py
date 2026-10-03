"""Compare every method on censored and uncensored versions of the same replicates.

Stdlib only:
``python3 analyze_uncensored.py <censored_dir> <uncensored_dir> [n_replicates]``.
Both directories hold ``uncensored.py`` outputs; the uncensored runs keep the
subjects, event times, causes, and evaluation grid of the censored runs. Only
replicates below n_replicates (default 10) that exist in both directories are used.
Entries are mean (SD) of MSE (1e-3), C_td, and IBS.
"""

from __future__ import annotations

import collections
import glob
import json
import statistics
import sys

METHODS = ("JointSoftComp", "SoftComp", "SoftComp-noaug", "NeuralFG")
LABELS = {
    "SoftComp": "SoftComp (manuscript)",
    "SoftComp-noaug": "SoftComp, no augmentation",
}


def load(out_dir: str, n_replicates: int) -> dict[tuple[int, str], dict[int, tuple]]:
    cells = collections.defaultdict(dict)
    for path in glob.glob(f"{out_dir}/case*_rep*.json"):
        with open(path) as f:
            record = json.load(f)
        if record["replicate"] >= n_replicates:
            continue
        m = record["result"]["metrics"]
        values = (m["MSE_overall"] * 1e3, m["Ctd_overall"], m["IBS_overall"])
        cells[(record["case"], record["method"])][record["replicate"]] = values
    return cells


def cell(runs: list[tuple], index: int) -> str:
    values = [run[index] for run in runs]
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    digits = 2 if index == 0 else 3
    return f"{statistics.mean(values):.{digits}f} ({sd:.{digits}f})"


def main() -> None:
    censored_dir, uncensored_dir = sys.argv[1], sys.argv[2]
    n_replicates = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    censored = load(censored_dir, n_replicates)
    uncensored = load(uncensored_dir, n_replicates)
    for case in (2, 3):
        print(f"\nCase {case}")
        print(
            "| Method | n | Censored MSE | C_td | IBS | Uncensored MSE | C_td | IBS |"
        )
        print("|---|---|---|---|---|---|---|---|")
        for method in METHODS:
            cens = censored.get((case, method), {})
            unc = uncensored.get((case, method), {})
            paired = sorted(set(cens) & set(unc))
            if not paired:
                continue
            cells = [cell([cens[r] for r in paired], i) for i in range(3)]
            cells += [cell([unc[r] for r in paired], i) for i in range(3)]
            label = LABELS.get(method, method)
            print(f"| {label} | {len(paired)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
