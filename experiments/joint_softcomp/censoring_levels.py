#!/usr/bin/env python3
"""Censoring levels relative to a fixed horizon: JointSoftComp vs. the manuscript's SoftComp.

Censoring is defined relative to the fixed horizon TAU: exponential censoring times are
calibrated so that a fraction --rho of the training subjects is censored before TAU,
P(C < min(T, TAU)) = rho, and follow-up ends at TAU (rho = 0: complete data up to TAU).
Test subjects are uncensored, and all metrics are evaluated on a fixed grid on (0, TAU].

Each invocation trains one model and writes one JSON file:
  aj-check   no covariates; compare to the Aalen-Johansen estimate and the true marginal CIF
  case3      Case III DGP with independent exponential censoring
  depcens    Case III DGP with covariate-dependent exponential censoring
  constant   constant cause-specific hazards with independent exponential censoring
  summarize  aggregate JSON files in --out-dir into a table
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

from ...crsoft_model import CRSoftNet
from ...crsoft_model.joint_softcomp import JointSoftComp
from ...data import case3_v5
from ...data.utils import solve_inverse_cdf
from ...evaluation import (
    compute_mse_accuracy,
    evaluate_cif_metrics,
    isotonic_project_cif,
)
from ...evaluation.postprocess import enforce_cif_simplex

logger: logging.Logger = logging.getLogger(__name__)

K = 3
P = 4
TAU = 20.0
HORIZONS = (5.0, 10.0, 15.0, 20.0)
CifFn = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]
CONSTANT_LOG_RATES = torch.log(torch.tensor([0.05, 0.035, 0.025]))
CONSTANT_EFFECTS = torch.tensor(
    [[0.6, 0.0, 0.0, 0.0], [0.0, 0.6, -0.4, 0.0], [-0.5, 0.0, 0.0, 0.5]]
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["aj-check", "case3", "depcens", "constant", "summarize"]
    )
    parser.add_argument("--method", choices=["softcomp", "joint"], default="joint")
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--n-times", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _constant_hazards(x: Tensor) -> Tensor:
    return torch.exp(CONSTANT_LOG_RATES + x @ CONSTANT_EFFECTS.T)


def constant_cif(x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    rates = _constant_hazards(x)
    total = rates.sum(dim=1)
    times = t.expand(x.shape[0]) if t.dim() == 0 else t
    survival = torch.exp(-total * times.clamp(min=0.0))
    return rates / total.unsqueeze(1) * (1.0 - survival).unsqueeze(1), survival


def _event_times(x: Tensor, cif_fn: CifFn, constant: bool) -> tuple[Tensor, Tensor]:
    if constant:
        rates = _constant_hazards(x)
        t_event = torch.distributions.Exponential(rates.sum(dim=1)).sample()
        cause = torch.multinomial(rates / rates.sum(1, keepdim=True), 1).squeeze(1) + 1
        return t_event, cause
    t_event = solve_inverse_cdf(x, torch.rand(x.shape[0]), cif_fn, t_max=case3_v5.T_MAX)
    f_at_t, _ = cif_fn(x, t_event)
    cause = torch.multinomial(f_at_t / f_at_t.sum(1, keepdim=True), 1).squeeze(1) + 1
    return t_event, cause


def _censoring_times(x: Tensor, t_event: Tensor, rho: float, dependent: bool) -> Tensor:
    """Exponential censoring at rate base (times exp(0.8 x1) if dependent), with base
    set so that P(C < min(T, TAU)) = rho."""
    if rho == 0.0:
        return torch.full_like(t_event, math.inf)
    shape = torch.exp(0.8 * x[:, 0]) if dependent else torch.ones_like(t_event)
    follow_up = t_event.clamp(max=TAU)
    lo, hi = 1e-6, 100.0
    for _ in range(100):
        base = math.sqrt(lo * hi)
        frac = float((1.0 - torch.exp(-base * shape * follow_up)).mean())
        lo, hi = (base, hi) if frac < rho else (lo, base)
    return torch.distributions.Exponential(base * shape).sample()


def simulate(
    command: str, n: int, rho: float, seed: int, cif_fn: CifFn, horizon: float = TAU
) -> tuple[Tensor, Tensor, Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(n, P)
    t_event, cause = _event_times(x, cif_fn, constant=command == "constant")
    c = _censoring_times(x, t_event, rho, dependent=command == "depcens")
    end = c.clamp(max=horizon)
    y = torch.minimum(t_event, end)
    return x, y, torch.where(t_event <= end, cause, torch.zeros_like(cause))


def train(
    method: str, x: Tensor, y: Tensor, delta: Tensor, args: argparse.Namespace
) -> CRSoftNet:
    torch.manual_seed(args.seed)
    if method == "softcomp":
        model = CRSoftNet(
            input_dim=x.shape[1], num_causes=K, hidden_dim=32, num_blocks=1
        )
        model.fit(
            x,
            y,
            delta,
            epochs=args.epochs,
            lr=1e-3,
            weight_decay=3e-3,
            batch_size=256,
            n_aug=2,
            aug_weight=0.5,
            verbose=False,
            device="cpu",
        )
        return model
    model = JointSoftComp(input_dim=x.shape[1], num_causes=K)
    model.fit(
        x,
        y,
        delta,
        epochs=args.epochs,
        n_times=args.n_times,
        verbose=False,
        device="cpu",
    )
    return model


def predict(model: CRSoftNet, method: str, x: Tensor, times: Tensor) -> Tensor:
    cif, _ = model.predict_cif_survival_grid(x, times)
    if method == "softcomp":
        return enforce_cif_simplex(isotonic_project_cif(cif))
    return cif


def _horizon_bias(cif: Tensor, x: Tensor, times: Tensor, cif_fn: CifFn) -> dict:
    out = {}
    for h in HORIZONS:
        j = int(torch.argmin((times - h).abs()))
        truth, _ = cif_fn(x, times[j].expand(x.shape[0]))
        diff = cif[:, :, j] - truth
        out[f"t{h:g}"] = {
            "bias_per_cause": diff.mean(0).tolist(),
            "mean_true_per_cause": truth.mean(0).tolist(),
            "mean_abs_err": float(diff.abs().mean()),
        }
    return out


def run_simulation(args: argparse.Namespace) -> dict:
    params = case3_v5.generate_parameters(K=K, p=P)
    cif_fn = (
        constant_cif
        if args.command == "constant"
        else functools.partial(case3_v5.compute_cif, params=params)
    )
    x, y, delta = simulate(
        args.command, args.n_train, args.rho, 30_000 + args.seed, cif_fn
    )
    x_te, y_te, d_te = simulate(
        args.command, args.n_test, 0.0, 40_000 + args.seed, cif_fn, horizon=math.inf
    )
    times = torch.linspace(TAU / 100, TAU, 100)
    start = time.perf_counter()
    model = train(args.method, x, y, delta, args)
    train_time = time.perf_counter() - start
    cif = predict(model, args.method, x_te, times)
    metrics = compute_mse_accuracy(cif, x_te, times, cif_fn, K)
    metrics.update(evaluate_cif_metrics(cif, y_te, d_te, y_te, d_te, times, K))
    return {
        "censored_before_tau_train": float(((delta == 0) & (y < TAU)).float().mean()),
        "event_free_at_tau_train": float((y >= TAU).float().mean()),
        "metrics": metrics,
        "horizons": _horizon_bias(cif, x_te, times, cif_fn),
        "train_time_sec": train_time,
    }


def aalen_johansen(y: Tensor, delta: Tensor, times: Tensor) -> Tensor:
    """Nonparametric Aalen-Johansen CIF estimate (K, len(times)) without covariates."""
    event_times = torch.unique(y[delta > 0])
    at_risk = (y.unsqueeze(0) >= event_times.unsqueeze(1)).sum(1).double()
    d = torch.stack(
        [
            ((y.unsqueeze(0) == event_times.unsqueeze(1)) & (delta == k)).sum(1)
            for k in range(1, K + 1)
        ],
        dim=1,
    ).double()
    survival = torch.cumprod(1.0 - d.sum(1) / at_risk, dim=0)
    s_before = torch.cat([torch.ones(1, dtype=torch.float64), survival[:-1]])
    cif_jumps = (s_before / at_risk).unsqueeze(1) * d
    cif_path = cif_jumps.cumsum(0)
    index = torch.searchsorted(event_times, times, right=True) - 1
    padded = torch.cat([torch.zeros(1, K, dtype=torch.float64), cif_path])
    return padded[index + 1].T.float()


def run_aj_check(args: argparse.Namespace) -> dict:
    params = case3_v5.generate_parameters(K=K, p=P)
    cif_fn = functools.partial(case3_v5.compute_cif, params=params)
    x, y, delta = simulate("case3", args.n_train, args.rho, 30_000 + args.seed, cif_fn)
    times = torch.linspace(0.0, 20.0, 81)
    torch.manual_seed(123)
    x_pop = torch.randn(200_000, P)
    true_marginal = torch.stack(
        [cif_fn(x_pop, t.expand(x_pop.shape[0]))[0].mean(0) for t in times], dim=1
    )
    aj = aalen_johansen(y, delta, times)
    constant_input = torch.zeros(args.n_train, 1)
    model = train(args.method, constant_input, y, delta, args)
    fitted = predict(model, args.method, torch.zeros(1, 1), times)[0]
    return {
        "times": times.tolist(),
        "true_marginal": true_marginal.tolist(),
        "aalen_johansen": aj.tolist(),
        "fitted": fitted.tolist(),
        "max_abs_vs_aj": float((fitted - aj).abs().max()),
        "max_abs_vs_truth": float((fitted - true_marginal).abs().max()),
        "aj_max_abs_vs_truth": float((aj - true_marginal).abs().max()),
    }


def _mean_metric(records: list[dict], key: str) -> float:
    return sum(r["result"]["metrics"][key] for r in records) / len(records)


def _mean_bias(records: list[dict], horizon: str) -> list[float]:
    total = [0.0] * K
    for r in records:
        for k, value in enumerate(r["result"]["horizons"][horizon]["bias_per_cause"]):
            total[k] += value / len(records)
    return [round(value, 3) for value in total]


def summarize(out_dir: Path) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for path in sorted(out_dir.glob("*.json")):
        record = json.loads(path.read_text())
        groups[(record["command"], record["rho"], record["method"])].append(record)
    for (command, rho, method), records in sorted(groups.items()):
        if command == "aj-check":
            r = records[0]["result"]
            logger.info(
                "%-8s rho=%.1f %-8s max|fit-AJ|=%.4f max|fit-truth|=%.4f "
                "max|AJ-truth|=%.4f",
                command,
                rho,
                method,
                r["max_abs_vs_aj"],
                r["max_abs_vs_truth"],
                r["aj_max_abs_vs_truth"],
            )
            continue
        censored = sum(r["result"]["censored_before_tau_train"] for r in records)
        logger.info(
            "%-8s rho=%.1f %-8s n=%d cens<tau=%.2f MSE=%.5f Ctd=%.4f IBS=%.4f "
            "bias@t10=%s bias@t20=%s",
            command,
            rho,
            method,
            len(records),
            censored / len(records),
            _mean_metric(records, "MSE_overall"),
            _mean_metric(records, "Ctd_overall"),
            _mean_metric(records, "IBS_overall"),
            _mean_bias(records, "t10"),
            _mean_bias(records, "t20"),
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    if args.command == "summarize":
        summarize(args.out_dir)
        return
    torch.set_num_threads(args.threads)
    result = run_aj_check(args) if args.command == "aj-check" else run_simulation(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.command}_rho{args.rho:g}_{args.method}_seed{args.seed}.json"
    payload = {
        "command": args.command,
        "rho": args.rho,
        "method": args.method,
        "seed": args.seed,
        "result": result,
    }
    (args.out_dir / name).write_text(json.dumps(payload))
    logger.info("wrote %s", args.out_dir / name)
