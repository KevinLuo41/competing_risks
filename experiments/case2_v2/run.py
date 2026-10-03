#!/usr/bin/env python3
"""Repeatable Monte Carlo runner for the second-version Case II experiment."""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from ...data.case2 import compute_cif, generate_data, generate_parameters
from ...data.utils import generate_test_observations
from ...evaluation import (
    build_evaluation_time_grid,
    compute_mse_accuracy,
    evaluate_cif_metrics,
    get_output_dir,
)
from ...evaluation.simulation import compute_dist
from ..case2.run import _build_specs, K, N_TEST, N_TRAIN, P
from ..runner import ModelSpec

CASE_NAME = "case2_v2"
MODEL_SEED = 0
DGP_SEED = 42
TRAIN_SEED_BASE = 10_000
TEST_SEED_BASE = 20_000
PAV_ATOL = 1e-6


@dataclass(frozen=True)
class ExperimentConfig:
    n_replicates: int
    start_replicate: int
    end_replicate: int
    model_seed: int
    dgp_seed: int
    train_seed_base: int
    test_seed_base: int
    cpu_threads: int
    models: tuple[str, ...] | None


@dataclass(frozen=True)
class PreparedData:
    X_train_full: Tensor
    Y_train_full: Tensor
    Delta_train_full: Tensor
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
    parameters: dict[str, Tensor]
    true_cif_fn: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class ModelRun:
    model: object
    cif: Tensor
    survival: Tensor
    metrics: dict[str, float]
    timings: dict[str, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-replicates", type=int, default=50)
    parser.add_argument("--start-replicate", type=int, default=0)
    parser.add_argument("--end-replicate", type=int)
    parser.add_argument("--model-seed", type=int, default=MODEL_SEED)
    parser.add_argument("--dgp-seed", type=int, default=DGP_SEED)
    parser.add_argument("--train-seed-base", type=int, default=TRAIN_SEED_BASE)
    parser.add_argument("--test-seed-base", type=int, default=TEST_SEED_BASE)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument(
        "--models",
        nargs="+",
        help="run only these methods; intended for isolated smoke tests",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip replicate directories that already completed successfully",
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    end_replicate = (
        args.n_replicates if args.end_replicate is None else args.end_replicate
    )
    if args.n_replicates <= 0:
        raise ValueError("--n-replicates must be positive")
    if not 0 <= args.start_replicate < end_replicate <= args.n_replicates:
        raise ValueError(
            "replicate range must satisfy "
            "0 <= start-replicate < end-replicate <= n-replicates"
        )
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    return ExperimentConfig(
        n_replicates=args.n_replicates,
        start_replicate=args.start_replicate,
        end_replicate=end_replicate,
        model_seed=args.model_seed,
        dgp_seed=args.dgp_seed,
        train_seed_base=args.train_seed_base,
        test_seed_base=args.test_seed_base,
        cpu_threads=args.cpu_threads,
        models=tuple(args.models) if args.models else None,
    )


def _protocol_payload(config: ExperimentConfig) -> dict[str, object]:
    return {
        **asdict(config),
        "case": "Case II (Nonlinear)",
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "n_causes": K,
        "n_features": P,
        "censor_rate": 0.5,
        "evaluation_grid_points": 100,
        "evaluation_percentile_cap": 90.0,
        "device": "cpu",
        "pav": "required for all SoftComp evaluation and saved CIFs",
        "softcomp_loss": "ell_1 only",
        "model_hyperparameters": {
            "DeepHit": {
                "n_bins": 100,
                "hidden_dim": 64,
                "n_layers": 2,
                "epochs": 500,
                "patience": 50,
                "alpha": 0.2,
                "sigma": 0.1,
            },
            "DSM": {
                "mixtures": 6,
                "layers": [64, 64],
                "distribution": "Weibull",
                "epochs": 500,
                "patience": 50,
            },
            "cs-Cox": {"penalizer": 0.01},
            "NeuralFG": {
                "layers": [100, 100, 100],
                "layers_surv": [100],
                "epochs": 1000,
                "patience": 3,
                "weight_decay": 0.001,
            },
            "SoftComp": {
                "hidden_dim": 32,
                "num_blocks": 1,
                "epochs": 1000,
                "lr": 0.001,
                "weight_decay": 0.003,
                "n_aug": 2,
                "aug_weight": 0.5,
            },
        },
    }


def _environment_payload() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_version": str(torch.__version__),
        "torch_num_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    temp_path.replace(path)


def _write_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def _get_tensor(data: dict[str, object], name: str) -> Tensor:
    value = data[name]
    if not isinstance(value, Tensor):
        raise TypeError(f"expected tensor for {name}, got {type(value).__name__}")
    return value


def _prepare_data(
    replicate: int,
    config: ExperimentConfig,
    parameters: dict[str, Tensor],
) -> PreparedData:
    generated = generate_data(
        n=N_TRAIN,
        K=K,
        p=P,
        censor_rate=0.5,
        seed=config.train_seed_base + replicate,
        params=parameters,
    )
    train_data: dict[str, object] = dict(generated)
    X_train = _get_tensor(train_data, "X")
    Y_train = _get_tensor(train_data, "Y")
    Delta_train = _get_tensor(train_data, "Delta")
    n_val = int(N_TRAIN * 0.1)
    true_cif_fn = functools.partial(compute_cif, params=parameters)
    X_test, Y_test, Delta_test = generate_test_observations(
        N_TEST,
        P,
        true_cif_fn,
        censor_rate=0.5,
        seed=config.test_seed_base + replicate,
    )
    eval_times = build_evaluation_time_grid(Y_test, Delta_test, n_grid=100)
    return PreparedData(
        X_train_full=X_train,
        Y_train_full=Y_train,
        Delta_train_full=Delta_train,
        X_train_fit=X_train[n_val:],
        Y_train_fit=Y_train[n_val:],
        Delta_train_fit=Delta_train[n_val:],
        X_val=X_train[:n_val],
        Y_val=Y_train[:n_val],
        Delta_val=Delta_train[:n_val],
        X_test=X_test,
        Y_test=Y_test,
        Delta_test=Delta_test,
        eval_times=eval_times,
        parameters=parameters,
        true_cif_fn=true_cif_fn,
    )


def _build_replicate_specs(data: PreparedData) -> list[ModelSpec]:
    return _build_specs(
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


def _true_cif_grid(data: PreparedData) -> Tensor:
    n = data.X_test.shape[0]
    cif = torch.zeros(n, K, len(data.eval_times))
    with torch.no_grad():
        for index, eval_time in enumerate(data.eval_times):
            values, _ = data.true_cif_fn(data.X_test, eval_time.expand(n))
            cif[:, :, index] = values
    return cif


def _evaluate_predictions(
    cif: Tensor,
    survival: Tensor,
    data: PreparedData,
) -> dict[str, float]:
    metrics = compute_mse_accuracy(
        cif,
        data.X_test,
        data.eval_times,
        data.true_cif_fn,
        K,
    )
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
    if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
        raise RuntimeError("evaluation produced a non-finite metric")
    return metrics


def _validate_softcomp_cif(cif: Tensor) -> None:
    if not torch.isfinite(cif).all():
        raise RuntimeError("SoftComp PAV output contains non-finite values")
    minimum_increment = float((cif[:, :, 1:] - cif[:, :, :-1]).min().item())
    if minimum_increment < -PAV_ATOL:
        raise RuntimeError(
            f"SoftComp PAV output is non-monotone: min increment={minimum_increment}"
        )


def _display_name(spec: ModelSpec) -> str:
    return "SoftComp" if spec.cache_key == "crsoft" else spec.name


def _run_model(
    spec: ModelSpec,
    data: PreparedData,
    model_seed: int,
) -> ModelRun:
    torch.manual_seed(model_seed)
    train_start = time.perf_counter()
    model = spec.train()
    train_time = time.perf_counter() - train_start

    if spec.predict_survival is None:
        raise RuntimeError(f"{spec.name} does not expose native survival")
    predict_start = time.perf_counter()
    cif, survival = spec.predict_survival(model, data.X_test, data.eval_times)
    predict_time = time.perf_counter() - predict_start

    pav_time = 0.0
    if spec.cache_key == "crsoft":
        pav_start = time.perf_counter()
        cif = spec.post_process(cif)
        pav_time = time.perf_counter() - pav_start
        _validate_softcomp_cif(cif)

    metrics = _evaluate_predictions(cif, survival, data)
    timings = {
        "train_time_sec": train_time,
        "predict_time_sec": predict_time,
        "pav_time_sec": pav_time,
        "total_time_sec": train_time + predict_time + pav_time,
    }
    return ModelRun(model, cif, survival, metrics, timings)


def _data_manifest(
    replicate: int,
    config: ExperimentConfig,
    data: PreparedData,
) -> dict[str, object]:
    train_counts = [
        int((data.Delta_train_full == event).sum().item()) for event in range(K + 1)
    ]
    test_counts = [
        int((data.Delta_test == event).sum().item()) for event in range(K + 1)
    ]
    return {
        "replicate": replicate,
        "train_data_seed": config.train_seed_base + replicate,
        "test_data_seed": config.test_seed_base + replicate,
        "model_seed": config.model_seed,
        "dgp_seed": config.dgp_seed,
        "parameters": {
            name: value.tolist() for name, value in sorted(data.parameters.items())
        },
        "train_counts_censor_then_causes": train_counts,
        "test_counts_censor_then_causes": test_counts,
        "evaluation_time_count": len(data.eval_times),
        "evaluation_time_min": float(data.eval_times[0].item()),
        "evaluation_time_max": float(data.eval_times[-1].item()),
    }


def _save_softcomp(
    directory: Path,
    spec: ModelSpec,
    run: ModelRun,
    data: PreparedData,
    true_cif: Tensor,
    model_seed: int,
) -> None:
    checkpoint = spec.serialize(run.model)
    checkpoint["model_seed"] = model_seed
    _write_torch(directory / "softcomp_model.pt", checkpoint)
    _write_torch(
        directory / "softcomp_training_history.pt",
        {"loss": list(getattr(run.model, "_train_losses", []))},
    )
    plot_generator = torch.Generator().manual_seed(0)
    plot_indices = torch.randperm(data.X_test.shape[0], generator=plot_generator)[:6]
    _write_torch(
        directory / "softcomp_predictions.pt",
        {
            "eval_times": data.eval_times,
            "cif_pav": run.cif,
            "native_survival": run.survival,
            "true_cif": true_cif,
            "X_test": data.X_test,
            "Y_test": data.Y_test,
            "Delta_test": data.Delta_test,
            "plot_sample_indices": plot_indices,
        },
    )


def _save_model_checkpoint(
    directory: Path,
    spec: ModelSpec,
    model: object,
    model_seed: int,
) -> None:
    checkpoint = spec.serialize(model)
    checkpoint["model_seed"] = model_seed
    _write_torch(directory / "checkpoints" / f"{spec.cache_key}.pt", checkpoint)


def _select_specs(
    specs: Sequence[ModelSpec],
    requested: tuple[str, ...] | None,
) -> list[ModelSpec]:
    if requested is None:
        return list(specs)
    selected: set[str] = set()
    for token in requested:
        match = next((spec for spec in specs if spec.matches(token)), None)
        if match is None:
            valid = ", ".join(_display_name(spec) for spec in specs)
            raise ValueError(f"unknown model '{token}'; valid models: {valid}")
        selected.add(match.cache_key)
    return [spec for spec in specs if spec.cache_key in selected]


def _load_method_result(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    payload = torch.load(path, weights_only=False)
    return payload["metrics"], payload["timings"]


def _execute_models(
    directory: Path,
    data: PreparedData,
    config: ExperimentConfig,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    metrics: dict[str, dict[str, float]] = {}
    timings: dict[str, dict[str, float]] = {}
    true_cif = _true_cif_grid(data)
    specs = _select_specs(_build_replicate_specs(data), config.models)
    for spec in specs:
        name = _display_name(spec)
        method_path = directory / "methods" / f"{spec.cache_key}.pt"
        if method_path.exists():
            print(f"\n[{name}] loading completed partial result", flush=True)
            metrics[name], timings[name] = _load_method_result(method_path)
            continue
        print(f"\n[{name}]", flush=True)
        run = _run_model(spec, data, config.model_seed)
        metrics[name] = run.metrics
        timings[name] = run.timings
        _save_model_checkpoint(
            directory,
            spec,
            run.model,
            config.model_seed,
        )
        if spec.cache_key == "crsoft":
            _save_softcomp(
                directory,
                spec,
                run,
                data,
                true_cif,
                config.model_seed,
            )
        _write_torch(
            method_path,
            {"metrics": run.metrics, "timings": run.timings},
        )
    return metrics, timings


def _print_replicate_results(
    replicate: int,
    metrics: dict[str, dict[str, float]],
    timings: dict[str, dict[str, float]],
) -> None:
    print(f"\nReplicate {replicate:03d} results")
    print(
        f"{'Method':<12} {'MSE':>9} {'Ctd':>9} {'IBS':>9} {'Dist':>11} {'Time(s)':>11}"
    )
    for method, values in metrics.items():
        print(
            f"{method:<12} {values['MSE_overall']:>9.4f} "
            f"{values['Ctd_overall']:>9.4f} {values['IBS_overall']:>9.4f} "
            f"{values['Dist']:>11.4g} "
            f"{timings[method]['total_time_sec']:>11.2f}"
        )


def _run_replicate(
    replicate: int,
    output_dir: Path,
    config: ExperimentConfig,
    parameters: dict[str, Tensor],
    resume: bool,
) -> bool:
    replicates_dir = output_dir / "replicates"
    replicates_dir.mkdir(parents=True, exist_ok=True)
    final_dir = replicates_dir / f"rep_{replicate:03d}"
    if final_dir.exists():
        if resume and (final_dir / "complete.json").exists():
            print(f"Replicate {replicate:03d}: already complete, skipping")
            return False
        raise FileExistsError(f"replicate output already exists: {final_dir}")

    temp_dir = replicates_dir / f".rep_{replicate:03d}.in_progress"
    if temp_dir.exists():
        if not resume:
            raise FileExistsError(
                f"partial replicate output already exists; rerun with --resume: {temp_dir}"
            )
        print(f"Replicate {replicate:03d}: resuming partial result")
    else:
        temp_dir.mkdir()
    print(f"\n{'=' * 72}\nReplicate {replicate:03d}\n{'=' * 72}")
    data = _prepare_data(replicate, config, parameters)
    metrics, timings = _execute_models(temp_dir, data, config)
    _write_json(
        temp_dir / "data_manifest.json", _data_manifest(replicate, config, data)
    )
    _write_json(temp_dir / "metrics.json", metrics)
    _write_json(temp_dir / "timings.json", timings)
    _write_json(temp_dir / "complete.json", {"complete": True})
    temp_dir.rename(final_dir)
    _print_replicate_results(replicate, metrics, timings)
    return True


def _load_json(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def _long_rows(
    output_dir: Path,
    n_replicates: int,
    filename: str,
) -> tuple[list[dict], list[int]]:
    rows: list[dict] = []
    completed: list[int] = []
    for replicate in range(n_replicates):
        path = output_dir / "replicates" / f"rep_{replicate:03d}" / filename
        if not path.exists():
            continue
        completed.append(replicate)
        for method, values in _load_json(path).items():
            for metric, value in values.items():
                rows.append(
                    {
                        "replicate": replicate,
                        "method": method,
                        "metric": metric,
                        "value": value,
                    }
                )
    return rows, completed


def _summary_rows(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["metric"]))
        grouped.setdefault(key, []).append(float(row["value"]))
    summary: list[dict] = []
    for (method, metric), values in sorted(grouped.items()):
        summary.append(
            {
                "method": method,
                "metric": metric,
                "n": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else None,
            }
        )
    return summary


def _aggregate(output_dir: Path, config: ExperimentConfig) -> None:
    aggregate_dir = output_dir / "aggregate"
    metric_rows, metric_replicates = _long_rows(
        output_dir, config.n_replicates, "metrics.json"
    )
    timing_rows, timing_replicates = _long_rows(
        output_dir, config.n_replicates, "timings.json"
    )
    long_fields = ["replicate", "method", "metric", "value"]
    summary_fields = ["method", "metric", "n", "mean", "std"]
    _write_csv(aggregate_dir / "metrics_long.csv", long_fields, metric_rows)
    _write_csv(
        aggregate_dir / "metrics_summary.csv",
        summary_fields,
        _summary_rows(metric_rows),
    )
    _write_csv(
        aggregate_dir / "timings_long.csv",
        long_fields,
        timing_rows,
    )
    _write_csv(
        aggregate_dir / "timing_summary.csv",
        summary_fields,
        _summary_rows(timing_rows),
    )
    completed = sorted(set(metric_replicates) & set(timing_replicates))
    _write_json(
        output_dir / "manifest.json",
        {
            "completed_replicates": completed,
            "missing_replicates": [
                replicate
                for replicate in range(config.n_replicates)
                if replicate not in completed
            ],
            "environment": _environment_payload(),
        },
    )


def main() -> None:
    args = _parse_args()
    config = _build_config(args)
    output_dir = args.output_dir or get_output_dir(CASE_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.cpu_threads)
    _write_json(output_dir / "config.json", _protocol_payload(config))

    parameters = generate_parameters(K=K, p=P, seed=config.dgp_seed)
    for replicate in range(config.start_replicate, config.end_replicate):
        _run_replicate(replicate, output_dir, config, parameters, args.resume)
        _aggregate(output_dir, config)
    print(f"\nArtifacts: {output_dir}")


if __name__ == "__main__":
    main()
