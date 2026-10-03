"""Summarize the simplest-setting runs (``simple.py``): Brier(10), AUC(10), IPA per beta.

Stdlib only: ``python3 analyze_simple.py <out_dir>``. Chunks of replicates written by
``simple.py --rep-start`` are pooled; entries are mean (SD) over replicates. Rows marked
"limit" are the large-sample limits of SoftComp (no training error).
"""

from __future__ import annotations

import collections
import glob
import json
import statistics
import sys

METRICS = (("brier", 4), ("auc", 3), ("ipa", 3), ("mse_true", 4))


def fmt(values: list[float], digits: int) -> str:
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return (
        f"{statistics.mean(values):.{digits}f} ({statistics.stdev(values):.{digits}f})"
    )


def main() -> None:
    out_dir = sys.argv[1]
    runs = collections.defaultdict(list)
    reference = {}
    for path in glob.glob(f"{out_dir}/simple_b*.json"):
        with open(path) as f:
            d = json.load(f)
        beta = round(d["beta"], 4)
        if d["method"] == "reference":
            reference[beta] = d
            continue
        runs[(beta, d["method"], d["m"])].extend(d["runs"])
    for beta in sorted({key[0] for key in runs} | set(reference)):
        ref = reference.get(beta)
        label = {0.0: "0", 0.4055: "log 1.5", 0.6931: "log 2"}.get(beta, str(beta))
        print(
            f"\nbeta = {label}"
            + (
                f"  (test event fraction {ref['test_event_fraction']:.3f})"
                if ref
                else ""
            )
        )
        print("| Method | M | n | Brier(10) | AUC(10) | IPA | MSE vs truth |")
        print("|---|---|---|---|---|---|---|")
        rows = []
        if ref:
            rows.append(("True model", "", [ref["oracle"]]))
            rows.append(("Cox", "", ref["cox"]["runs"]))
            rows.append(("No covariate", "", ref["null"]["runs"]))
        for m in (0, 1, 2, 4, 8):
            if (beta, "softcomp", m) in runs:
                rows.append(("SoftComp", str(m), runs[(beta, "softcomp", m)]))
            if ref and str(m) in ref["softcomp_limit"]:
                rows.append(
                    ("SoftComp, limit", str(m), [ref["softcomp_limit"][str(m)]])
                )
        for m in (1, 2, 4, 8):
            if (beta, "joint", m) in runs:
                rows.append(("JointSoftComp", str(m), runs[(beta, "joint", m)]))
        for name, m, rs in rows:

            print(f"| {name} | {m} | {len(rs)} | {cells} |")


if __name__ == "__main__":
    main()
