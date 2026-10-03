#!/usr/bin/env python3
"""Simplest setting: one predictor, uncensored data, Brier(10) and AUC(10).

One event type with hazard lambda0 * exp(beta * x), x ~ N(0, 1), lambda0 = 0.05 per year,
and no censoring. Each invocation fits one method on R training sets of size n and
evaluates the predicted risk F(10 | x) on one large uncensored test set:
  softcomp   the manuscript's SoftComp: Eq. (5) plus M augmented times (weight 0.5),
             PAV and simplex post-processing
  joint      JointSoftComp (unified configuration) with M sampled times per subject
  reference  Cox with Breslow baseline, the true model, the marginal (no-covariate)
             model, and the large-sample limit of SoftComp for each M
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from pathlib import Path

import torch
from torch import Tensor

from ...crsoft_model import CRSoftNet
from ...crsoft_model.joint_softcomp import JointSoftComp
from ...evaluation import isotonic_project_cif
from ...evaluation.postprocess import enforce_cif_simplex

logger: logging.Logger = logging.getLogger(__name__)

LAMBDA0 = 0.05
HORIZON = 10.0
AUG_WEIGHT = 0.5
LIMIT_MS = (0, 1, 2, 4, 8)
TRAIN_SEED, TEST_SEED = 700_000, 999_999


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["softcomp", "joint", "reference"])
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=50_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def simulate(n: int, beta: float, seed: int) -> tuple[Tensor, Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, generator=gen)
    rate = LAMBDA0 * torch.exp(beta * x[:, 0])
    return x, torch.empty(n).exponential_(1.0, generator=gen) / rate


def true_risk(x: Tensor, beta: float) -> Tensor:
    return 1.0 - torch.exp(-HORIZON * LAMBDA0 * torch.exp(beta * x))


def scaled_exp1(z: float) -> float:
    """exp(z) * E1(z) for z > 0 (series for z <= 1, continued fraction otherwise)."""
    if z <= 1.0:
        total, term = -math.log(z) - 0.5772156649015329, 1.0
        for i in range(1, 200):
            term *= -z / i
            total -= term / i
            if abs(term / i) < 1e-16 * abs(total):
                break
        return math.exp(z) * total
    b, c, d = z + 1.0, 1e300, 1.0 / (z + 1.0)
    h = d
    for i in range(1, 500):
        a = -float(i * i)
        b += 2.0
        d = 1.0 / (a * d + b)
        c = b + a / c
        h *= c * d
        if abs(c * d - 1.0) < 1e-15:
            break
    return h


def softcomp_limit(x: Tensor, beta: float, m: int) -> Tensor:
    """Population minimizer of Eq. (5) plus augmentation at the horizon (no censoring)."""
    z = HORIZON * LAMBDA0 * torch.exp(beta * x)
    scaled = torch.tensor([scaled_exp1(float(v)) for v in z], dtype=x.dtype)
    return 1.0 / (1.0 + m * AUG_WEIGHT * scaled)


def predict_softcomp(x: Tensor, t: Tensor, m: int, grid: Tensor) -> Tensor:
    model = CRSoftNet(input_dim=1, num_causes=1, hidden_dim=32, num_blocks=1)
    model.fit(
        x,
        t,
        torch.ones(len(t), dtype=torch.long),
        epochs=1000,
        lr=1e-3,
        weight_decay=3e-3,
        n_aug=m,
        aug_weight=AUG_WEIGHT,
        verbose=False,
        device="cpu",
    )
    cif, _ = model.predict_cif_survival_grid(grid[:, None], torch.tensor([HORIZON]))
    return enforce_cif_simplex(isotonic_project_cif(cif))[:, 0, 0]


def predict_joint(x: Tensor, t: Tensor, m: int, grid: Tensor) -> Tensor:
    model = JointSoftComp(input_dim=1, num_causes=1, grid_size=400)
    model.fit(
        x,
        t,
        torch.ones(len(t), dtype=torch.long),
        n_times=m,
        verbose=False,
        device="cpu",
    )
    cif, _ = model.predict_cif_survival_grid(grid[:, None], torch.tensor([HORIZON]))
    return cif[:, 0, 0]


def predict_cox(x: Tensor, t: Tensor, grid: Tensor) -> Tensor:
    """Cox model with one covariate (no ties) and Breslow baseline hazard."""
    order = torch.argsort(t)
    xs, ts = x[order, 0].double(), t[order].double()
    beta = 0.0
    for _ in range(100):
        w = torch.exp(beta * xs)
        s0 = w.flip(0).cumsum(0).flip(0)
        s1 = (w * xs).flip(0).cumsum(0).flip(0)
        s2 = (w * xs * xs).flip(0).cumsum(0).flip(0)
        score = float((xs - s1 / s0).sum())
        info = float((s2 / s0 - (s1 / s0) ** 2).sum())
        beta += score / info
        if abs(score / info) < 1e-10:
            break
    s0 = torch.exp(beta * xs).flip(0).cumsum(0).flip(0)
    baseline = float((1.0 / s0)[ts <= HORIZON].sum())
    return (1.0 - torch.exp(-baseline * torch.exp(beta * grid.double()))).float()


def interpolate(grid: Tensor, values: Tensor, x: Tensor) -> Tensor:
    idx = torch.searchsorted(grid, x).clamp(1, len(grid) - 1)
    x0, x1 = grid[idx - 1], grid[idx]
    w = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0)
    return values[idx - 1] + w * (values[idx] - values[idx - 1])


def auc(score: Tensor, case: Tensor) -> float:
    """Mann-Whitney AUC with ties counted as 1/2."""
    _, inverse, counts = torch.unique(score, return_inverse=True, return_counts=True)
    avg_rank = (torch.cumsum(counts, 0) - (counts - 1) / 2.0).double()[inverse]
    n1 = int(case.sum())
    n0 = len(case) - n1
    return float((avg_rank[case].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def evaluate(pred: Tensor, test: dict, brier_null: float) -> dict[str, float]:
    brier = float(((pred - test["d"]) ** 2).mean())
    return {
        "brier": brier,
        "auc": auc(pred, test["d"] > 0.5),
        "ipa": 1.0 - brier / brier_null,
        "mse_true": float(((pred - test["f"]) ** 2).mean()),
        "mean_risk": float(pred.mean()),
    }


def summarize(runs: list[dict[str, float]]) -> dict[str, list[float]]:
    return {
        key: [
            statistics.mean(r[key] for r in runs),
            statistics.stdev(r[key] for r in runs),
        ]
        for key in runs[0]
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    torch.set_num_threads(args.threads)
    grid = torch.linspace(-5.0, 5.0, 1001)
    x_test, t_test = simulate(args.n_test, args.beta, TEST_SEED)
    test = {"d": (t_test <= HORIZON).float(), "f": true_risk(x_test[:, 0], args.beta)}
    runs, cox_runs, null_runs = [], [], []
    for r in range(args.rep_start, args.rep_start + args.reps):
        x, t = simulate(args.n_train, args.beta, TRAIN_SEED + r)
        null_risk = float((t <= HORIZON).float().mean())
        brier_null = float(((null_risk - test["d"]) ** 2).mean())
        if args.method == "reference":
            cox = interpolate(grid, predict_cox(x, t, grid), x_test[:, 0])
            cox_runs.append(evaluate(cox, test, brier_null))
            null_runs.append(
                evaluate(torch.full_like(test["f"], null_risk), test, brier_null)
            )
            continue
        torch.manual_seed(r)
        fit = predict_softcomp if args.method == "softcomp" else predict_joint
        pred = interpolate(grid, fit(x, t, args.m, grid), x_test[:, 0])
        runs.append(evaluate(pred, test, brier_null))
    payload: dict[str, object] = {
        "beta": args.beta,
        "method": args.method,
        "m": args.m,
        "rep_start": args.rep_start,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "test_event_fraction": float(test["d"].mean()),
    }
    if args.method == "reference":
        brier_marginal = float(((test["d"].mean() - test["d"]) ** 2).mean())
        payload["cox"] = {"runs": cox_runs, "summary": summarize(cox_runs)}
        payload["null"] = {"runs": null_runs, "summary": summarize(null_runs)}
        payload["oracle"] = evaluate(test["f"], test, brier_marginal)
        payload["softcomp_limit"] = {
            str(m): evaluate(
                softcomp_limit(x_test[:, 0], args.beta, m), test, brier_marginal
            )
            for m in LIMIT_MS
        }
    else:
        payload["runs"] = runs
        payload["summary"] = summarize(runs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"simple_b{args.beta:.4f}_{args.method}_m{args.m}_r{args.rep_start}.json"
    (args.out_dir / name).write_text(json.dumps(payload))
    logger.info("wrote %s", args.out_dir / name)
