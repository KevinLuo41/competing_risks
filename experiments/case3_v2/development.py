#!/usr/bin/env python3
"""Development-only Case III DGP and SoftComp configuration experiments.

This entry point uses subject seeds disjoint from the final Monte Carlo seeds.
It can run the four frozen baselines once and SoftComp configurations
separately, avoiding any tuning on the final 50 replicates.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

from ...crsoft_model import CRSoftNet
from ...data.case3_interaction import (
    compute_cif,
    compute_time_slope,
    generate_data,
    generate_parameters,
    T_MAX,
)
from ...data.utils import generate_test_observations
from ...evaluation import (
    build_evaluation_time_grid,
    compute_mse_accuracy,
    evaluate_cif_metrics,
    isotonic_project_cif,
)
from ...evaluation.simulation import compute_dist
from ..case2.run import _build_specs
from ..runner import ModelSpec

K = 3
P = 5


@dataclass(frozen=True)
class SoftCompConfig:
    hidden_dim: int = 32
    num_blocks: int = 1
    epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 3e-3
    n_aug: int = 2
    aug_weight: float = 0.5

    def label(self) -> str:
        return (
            f"h{self.hidden_dim}_b{self.num_blocks}_ep{self.epochs}_"
            f"lr{self.lr:g}_wd{self.weight_decay:g}_"
            f"aug{self.n_aug}_aw{self.aug_weight:g}"
        )


@dataclass(frozen=True)
class PreparedData:
    X_train_full: Tensor
    Y_train_full: Tensor
    Delta_train_full: Tensor
    T_train_true: Tensor
    X_train_fit: Tensor
    Y_train_fit: Tensor
    Delta_train_fit: Tensor
    X_val: Tensor
    Y_val: Tensor
    Delta_val: Tensor
    X_test: Tensor
    Y_test: Tensor
    Delta_test: Tensor
    eval_times: Tensor
    params: dict[str, Tensor]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--train-seed", type=int, default=30_000)
    parser.add_argument("--test-seed", type=int, default=40_000)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--dgp-seed", type=int, default=42)
    parser.add_argument("--interaction-range", type=float, default=0.6)
    parser.add_argument("--gamma-scale", type=float, default=1.0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--models", nargs="+", default=["SoftComp"])
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-3)
    parser.add_argument("--n-aug", type=int, default=2)
    parser.add_argument("--aug-weight", type=float, default=0.5)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="run the built-in SoftComp-only configuration grid",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _get_tensor(data: dict[str, object], name: str) -> Tensor:
    value = data[name]
    if not isinstance(value, Tensor):
        raise TypeError(f"expected tensor for {name}, got {type(value).__name__}")
    return value


def _prepare_data(args: argparse.Namespace) -> PreparedData:
    params = generate_parameters(
        K=K,
        p=P,
        seed=args.dgp_seed,
        interaction_range=args.interaction_range,
        gamma_scale=args.gamma_scale,
    )
    generated = generate_data(
        n=args.n_train,
        K=K,
        p=P,
        censor_rate=0.5,
        seed=args.train_seed,
        params=params,
    )
    generated_data: dict[str, object] = dict(generated)
    x_train = _get_tensor(generated_data, "X")
    y_train = _get_tensor(generated_data, "Y")
    delta_train = _get_tensor(generated_data, "Delta")
    t_train_true = _get_tensor(generated_data, "T_true")
    true_cif_fn = functools.partial(compute_cif, params=params)
    x_test, y_test, delta_test = generate_test_observations(
        args.n_test,
        P,
        true_cif_fn,
        censor_rate=0.5,
        seed=args.test_seed,
        t_max=T_MAX,
    )
    eval_times = build_evaluation_time_grid(y_test, delta_test, n_grid=100)
    n_val = int(args.n_train * 0.1)
    return PreparedData(
        X_train_full=x_train,
        Y_train_full=y_train,
        Delta_train_full=delta_train,
        T_train_true=t_train_true,
        X_train_fit=x_train[n_val:],
        Y_train_fit=y_train[n_val:],
        Delta_train_fit=delta_train[n_val:],
        X_val=x_train[:n_val],
        Y_val=y_train[:n_val],
        Delta_val=delta_train[:n_val],
        X_test=x_test,
        Y_test=y_test,
        Delta_test=delta_test,
        eval_times=eval_times,
        params=params,
    )


def _softcomp_spec(
    data: PreparedData,
    config: SoftCompConfig,
    model_seed: int,
) -> ModelSpec:
    def train() -> CRSoftNet:
        torch.manual_seed(model_seed)
        model = CRSoftNet(
            input_dim=P,
            num_causes=K,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
        )
        model.fit(
            data.X_train_full,
            data.Y_train_full,
            data.Delta_train_full,
            epochs=config.epochs,
            lr=config.lr,
            weight_decay=config.weight_decay,
            n_aug=config.n_aug,
            aug_weight=config.aug_weight,
            verbose=False,
            device="cpu",
        )
        return model

    return ModelSpec(
        name="SoftComp",
        cache_key="crsoft",
        cli_aliases=["crsoft", "softcomp"],
        train=train,
        predict=lambda model, x, times: model.predict_cif_grid(x, times),
        predict_survival=lambda model, x, times: model.predict_cif_survival_grid(
            x, times
        ),
        post_process=lambda cif: isotonic_project_cif(cif, data.eval_times),
    )


def _build_development_specs(
    data: PreparedData,
    config: SoftCompConfig,
    model_seed: int,
) -> list[ModelSpec]:
    specs = _build_specs(
        data.X_train_fit,
        data.Y_train_fit,
        data.Delta_train_fit,
        data.X_val,
        data.Y_val,
        data.Delta_val,
        data.X_train_full,
        data.Y_train_full,
        data.Delta_train_full,
        data.eval_times,
        device="cpu",
    )
    return [
        _softcomp_spec(data, config, model_seed) if spec.cache_key == "crsoft" else spec
        for spec in specs
    ]


def _softcomp_sweep(base: SoftCompConfig) -> list[SoftCompConfig]:
    candidates = [
        base,
        replace(base, n_aug=1),
        replace(base, n_aug=4),
        replace(base, weight_decay=1e-3),
        replace(base, weight_decay=1e-2),
        replace(base, hidden_dim=64),
        replace(base, num_blocks=2),
        replace(base, lr=5e-4),
    ]
    deduplicated: list[SoftCompConfig] = []
    labels: set[str] = set()
    for candidate in candidates:
        if candidate.label() not in labels:
            labels.add(candidate.label())
            deduplicated.append(candidate)
    return deduplicated


def _select_specs(specs: list[ModelSpec], requested: list[str]) -> list[ModelSpec]:
    selected: list[ModelSpec] = []
    for token in requested:
        spec = next(
            (candidate for candidate in specs if candidate.matches(token)), None
        )
        if spec is None:
            valid = ", ".join(candidate.name for candidate in specs)
            raise ValueError(f"unknown model {token!r}; valid models: {valid}")
        if spec not in selected:
            selected.append(spec)
    return selected


def _evaluate(
    cif: Tensor,
    survival: Tensor,
    data: PreparedData,
) -> dict[str, float]:
    true_cif_fn = functools.partial(compute_cif, params=data.params)
    metrics = compute_mse_accuracy(cif, data.X_test, data.eval_times, true_cif_fn, K)
    metrics.update(
        evaluate_cif_metrics(
            cif,
            data.Y_test,
            data.Delta_test,
            data.Y_train_full,
            data.Delta_train_full,
            data.eval_times,
            K,
        )
    )
    metrics.update(compute_dist(cif, survival))
    return metrics


def _run_specs(
    specs: list[ModelSpec],
    data: PreparedData,
    model_seed: int,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for spec in specs:
        torch.manual_seed(model_seed)
        start = time.perf_counter()
        model = spec.train()
        train_time = time.perf_counter() - start
        if spec.predict_survival is None:
            raise RuntimeError(f"{spec.name} does not expose native survival")
        start = time.perf_counter()
        cif, survival = spec.predict_survival(model, data.X_test, data.eval_times)
        predict_time = time.perf_counter() - start
        pav_time = 0.0
        if spec.cache_key == "crsoft":
            start = time.perf_counter()
            cif = spec.post_process(cif)
            pav_time = time.perf_counter() - start
        metrics = _evaluate(cif, survival, data)
        results[spec.name] = {
            "metrics": metrics,
            "timings": {
                "train_time_sec": train_time,
                "predict_time_sec": predict_time,
                "pav_time_sec": pav_time,
                "total_time_sec": train_time + predict_time + pav_time,
            },
        }
        print(
            f"{spec.name:<10} MSE={metrics['MSE_overall']:.6f} "
            f"Ctd={metrics['Ctd_overall']:.6f} "
            f"IBS={metrics['IBS_overall']:.6f} "
            f"time={train_time + predict_time + pav_time:.2f}s",
            flush=True,
        )
    return results


def _diagnostics(data: PreparedData) -> dict[str, object]:
    slope = compute_time_slope(data.X_test, data.params)
    sorted_slope = slope.sort().values
    quantiles = {}
    for percentile in (0, 5, 25, 50, 75, 95, 100):
        index = round((len(sorted_slope) - 1) * percentile / 100)
        quantiles[str(percentile)] = float(sorted_slope[index].item())
    event_at_limit = float((data.T_train_true >= T_MAX - 1e-3).float().mean().item())
    return {
        "time_slope_quantiles": quantiles,
        "train_censor_fraction": float(
            (data.Delta_train_full == 0).float().mean().item()
        ),
        "test_censor_fraction": float((data.Delta_test == 0).float().mean().item()),
        "train_event_time_at_inverse_cdf_limit_fraction": event_at_limit,
        "eval_time_min": float(data.eval_times[0].item()),
        "eval_time_max": float(data.eval_times[-1].item()),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def main() -> None:
    args = _parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    config = SoftCompConfig(
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_aug=args.n_aug,
        aug_weight=args.aug_weight,
    )
    data = _prepare_data(args)
    diagnostics = _diagnostics(data)
    print(f"DGP diagnostics: {json.dumps(diagnostics, sort_keys=True)}")
    if args.sweep:
        if args.models != ["SoftComp"]:
            raise ValueError("--sweep only supports the default --models SoftComp")
        sweep_results = {}
        for sweep_config in _softcomp_sweep(config):
            print(f"\nSoftComp configuration: {sweep_config.label()}", flush=True)
            result = _run_specs(
                [_softcomp_spec(data, sweep_config, args.model_seed)],
                data,
                args.model_seed,
            )
            sweep_results[sweep_config.label()] = {
                "config": asdict(sweep_config),
                **result["SoftComp"],
            }
        results: object = sweep_results
    else:
        specs = _select_specs(
            _build_development_specs(data, config, args.model_seed), args.models
        )
        results = _run_specs(specs, data, args.model_seed)
    payload = {
        "protocol": {
            "n_train": args.n_train,
            "n_test": args.n_test,
            "train_seed": args.train_seed,
            "test_seed": args.test_seed,
            "model_seed": args.model_seed,
            "dgp_seed": args.dgp_seed,
            "interaction_range": args.interaction_range,
            "gamma_scale": args.gamma_scale,
            "cpu_threads": args.cpu_threads,
            "softcomp": asdict(config),
            "softcomp_label": config.label(),
            "sweep": args.sweep,
        },
        "diagnostics": diagnostics,
        "results": results,
    }
    if args.output is not None:
        _write_json(args.output, payload)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
