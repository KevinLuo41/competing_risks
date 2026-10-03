#!/usr/bin/env python3
"""Development-gate and formal Monte Carlo runner for Case II v3."""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
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

from ...data import case2_v3
from ...data.utils import generate_test_observations
from ...evaluation import (
    build_evaluation_time_grid,
    compute_mse_accuracy,
    evaluate_cif_metrics,
    get_output_dir,
    isotonic_project_cif,
)
from ...evaluation.postprocess import enforce_cif_simplex
from ...evaluation.simulation import compute_dist
from ..runner import ModelSpec
from .run import (
    _build_specs,
    _select_specs,
    _training_history,
    K,
    P,
    PreparedData,
    SoftCompConfig,
)

DEVELOPMENT_MODELS = ("DeepHit", "DSM", "cs-Cox", "NeuralFG", "SoftComp")
ALL_MODELS = DEVELOPMENT_MODELS + ("Fine-Gray",)
METHOD_CACHE_KEYS = {
    "DeepHit": "deephit",
    "DSM": "dsm",
    "cs-Cox": "cs_cox",
    "NeuralFG": "neural_fg",
    "SoftComp": "crsoft",
    "Fine-Gray": "fine_gray",
}
REQUIRED_METHOD_ARTIFACTS = (
    "checkpoint.pt",
    "training_history.pt",
    "test_predictions.pt",
    "plot_predictions.pt",
    "metrics.json",
    "timings.json",
    "all_subject_probability_diagnostics.json",
    "complete.json",
)
# This original schedule remains in the hash payload so completed artifacts keep
# the same identity after the effective schedule was reduced to 10 paired runs.
ORIGINAL_FORMAL_REPLICATES_BY_METHOD = {
    "DeepHit": 50,
    "DSM": 10,
    "cs-Cox": 50,
    "NeuralFG": 50,
    "SoftComp": 50,
}
FORMAL_REPLICATES_BY_METHOD = {
    "DeepHit": 10,
    "DSM": 10,
    "cs-Cox": 10,
    "NeuralFG": 10,
    "SoftComp": 10,
    "Fine-Gray": 10,
}
MODEL_SEED = 0
DGP_SEED = 42
DEVELOPMENT_TRAIN_SEED_BASE = 30_000
DEVELOPMENT_TEST_SEED_BASE = 40_000
FORMAL_TRAIN_SEED_BASE = 50_000
FORMAL_TEST_SEED_BASE = 60_000
PLOT_COHORT_SEED = 70_000
PLOT_SUBJECTS = 5
PLOT_TIME_POINTS = 100
PLOT_TIME_MAX = 20.0
NEGATIVE_SURVIVAL_ATOL = 1e-6
FINE_GRAY_IMPLEMENTATION_VERSION = 1


@dataclass(frozen=True)
class ExperimentConfig:
    phase: str
    n_replicates: int
    start_replicate: int
    end_replicate: int
    train_seed_base: int
    test_seed_base: int
    model_seed: int
    dgp_seed: int
    cpu_threads: int
    models: tuple[str, ...]
    softcomp: SoftCompConfig


@dataclass(frozen=True)
class PlotCohort:
    X: Tensor
    times: Tensor
    true_cif: Tensor
    true_survival: Tensor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("development", "formal", "smoke"),
        default="development",
    )
    parser.add_argument("--n-replicates", type=int)
    parser.add_argument("--start-replicate", type=int, default=0)
    parser.add_argument("--end-replicate", type=int)
    parser.add_argument("--train-seed-base", type=int)
    parser.add_argument("--test-seed-base", type=int)
    parser.add_argument("--model-seed", type=int, default=MODEL_SEED)
    parser.add_argument("--dgp-seed", type=int, default=DGP_SEED)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gate-file", type=Path)
    parser.add_argument(
        "--gate-result",
        action="append",
        default=[],
        metavar="REPLICATE=PATH",
        help="import one development single-run JSON; may be repeated",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--n-aug", type=int, default=4)
    parser.add_argument("--aug-weight", type=float, default=0.5)
    return parser.parse_args()


def _phase_defaults(phase: str) -> tuple[int, int, int, str]:
    if phase == "development":
        return (
            4,
            DEVELOPMENT_TRAIN_SEED_BASE,
            DEVELOPMENT_TEST_SEED_BASE,
            "development_gate",
        )
    if phase == "formal":
        return (
            10,
            FORMAL_TRAIN_SEED_BASE,
            FORMAL_TEST_SEED_BASE,
            "formal_10",
        )
    return (1, 49_000, 59_000, "smoke")


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    default_n, default_train_seed, default_test_seed, _ = _phase_defaults(args.phase)
    n_replicates = args.n_replicates or default_n
    train_seed_base = args.train_seed_base or default_train_seed
    test_seed_base = args.test_seed_base or default_test_seed
    end_replicate = args.end_replicate or n_replicates
    default_models = DEVELOPMENT_MODELS if args.phase == "development" else ALL_MODELS
    models = tuple(args.models or default_models)
    if n_replicates <= 0:
        raise ValueError("--n-replicates must be positive")
    if not 0 <= args.start_replicate < end_replicate <= n_replicates:
        raise ValueError(
            "replicate range must satisfy "
            "0 <= start-replicate < end-replicate <= n-replicates"
        )
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    if args.phase == "development":
        expected = (4, DEVELOPMENT_TRAIN_SEED_BASE, DEVELOPMENT_TEST_SEED_BASE)
        actual = (n_replicates, train_seed_base, test_seed_base)
        if actual != expected or models != DEVELOPMENT_MODELS:
            raise ValueError(
                "development gate requires four paired replicates, seeds "
                "30000--30003/40000--40003, and the original five methods"
            )
    if args.phase == "formal":
        expected = (10, FORMAL_TRAIN_SEED_BASE, FORMAL_TEST_SEED_BASE)
        actual = (n_replicates, train_seed_base, test_seed_base)
        if actual != expected or models != ALL_MODELS:
            raise ValueError(
                "formal phase requires 10 paired replicates, seeds "
                "50000--50009/60000--60009, and all six methods"
            )
    return ExperimentConfig(
        phase=args.phase,
        n_replicates=n_replicates,
        start_replicate=args.start_replicate,
        end_replicate=end_replicate,
        train_seed_base=train_seed_base,
        test_seed_base=test_seed_base,
        model_seed=args.model_seed,
        dgp_seed=args.dgp_seed,
        cpu_threads=args.cpu_threads,
        models=models,
        softcomp=SoftCompConfig(
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            n_aug=args.n_aug,
            aug_weight=args.aug_weight,
        ),
    )


def expected_methods_for_replicate(
    config: ExperimentConfig,
    replicate: int,
) -> tuple[str, ...]:
    """Return the methods pre-registered for one replicate."""
    if replicate not in range(config.n_replicates):
        raise ValueError(
            f"replicate must be in 0--{config.n_replicates - 1}, got {replicate}"
        )
    if config.phase != "formal":
        return config.models
    return tuple(
        method
        for method in config.models
        if replicate < FORMAL_REPLICATES_BY_METHOD[method]
    )


def expected_method_replicate_counts(
    config: ExperimentConfig,
) -> dict[str, int]:
    """Return the number of expected completed replicates for each method."""
    counts = dict.fromkeys(config.models, 0)
    for replicate in range(config.n_replicates):
        for method in expected_methods_for_replicate(config, replicate):
            counts[method] += 1
    return counts


def _fine_gray_amendment_payload() -> dict[str, object]:
    return {
        "model": "classical Fine-Gray proportional subdistribution hazards",
        "implementation_version": FINE_GRAY_IMPLEMENTATION_VERSION,
        "n_causes": K,
        "n_features": P,
        "fit_subset": "stored 4500-subject fitting_indices",
        "feature_scaling": "center and population-standard-deviation scale",
        "censoring_weights": "single-group Kaplan-Meier G(t-)/G(T_i-)",
        "ties": "Breslow with cmprsk censoring-risk-set convention",
        "penalizer": 0.0,
        "max_iter": 100,
        "tolerance": 1e-7,
        "prediction": "right-continuous Breslow CIF step function",
        "joint_survival": "unclamped 1 - sum_k CIF_k",
    }


def fine_gray_amendment_hash() -> str:
    """Return the identity of the required sixth-method configuration."""
    encoded = json.dumps(
        _fine_gray_amendment_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _formal_schedule_metadata(config: ExperimentConfig) -> dict[str, object]:
    if config.phase != "formal":
        return {}
    return {
        "original_formal_replicates_by_method": dict(
            ORIGINAL_FORMAL_REPLICATES_BY_METHOD
        ),
        "effective_formal_replicates_by_method": (
            expected_method_replicate_counts(config)
        ),
        "schedule_amendment": {
            "date": "2026-08-15",
            "reason": (
                "After measuring the CPU cost, the formal schedule was reduced "
                "to the first 10 paired replicates for every method."
            ),
            "retained_replicates": list(range(config.n_replicates)),
            "selection_rule": (
                "Consecutive replicate indices 0--9 were fixed without "
                "outcome-based filtering."
            ),
            "configuration_hash_compatibility": (
                "The original schedule remains in the frozen hash payload so "
                "the completed model and DGP artifacts retain their identity."
            ),
        },
        "method_amendment": {
            "date": "2026-08-15",
            "added_method": "Fine-Gray",
            "required_replicates": list(range(config.n_replicates)),
            "reason": (
                "Fine-Gray is a required classical baseline and was added after "
                "the first five methods had completed."
            ),
            "paired_data_rule": (
                "Reuse the saved training/test subjects, censoring, split, and "
                "evaluation grid for replicates 0--9; do not regenerate data."
            ),
            "configuration_hash_compatibility": (
                "The original frozen protocol remains the base hash identity; "
                "this explicit method amendment records the sixth baseline."
            ),
            "configuration": _fine_gray_amendment_payload(),
            "configuration_hash": fine_gray_amendment_hash(),
        },
    }


def _frozen_protocol_payload(config: ExperimentConfig) -> dict[str, object]:
    return {
        "case": "Case II v3",
        "n_train": 5000,
        "n_test": 1000,
        "n_causes": K,
        "n_features": P,
        "censor_rate": 0.5,
        "dgp_seed": config.dgp_seed,
        "model_seed": config.model_seed,
        "formal_replicates_by_method": ORIGINAL_FORMAL_REPLICATES_BY_METHOD,
        "dgp": {
            "beta_distribution": "Uniform(0.1, 0.3), drawn once",
            "intercept_start": -5.1,
            "intercept_end": -4.9,
            "quadratic_weight": 1.5,
            "pairwise_weight": 1.0,
            "time_scale": 12.0,
        },
        "models": {
            "DeepHit": {
                "n_bins": 100,
                "hidden_dim": 64,
                "n_layers": 2,
                "dropout": 0.1,
                "lr": 0.001,
                "batch_size": 256,
                "epochs": 500,
                "patience": 50,
                "alpha": 0.2,
                "sigma": 0.1,
            },
            "DSM": {
                "mixtures": 6,
                "layers": [64, 64],
                "distribution": "Weibull",
                "gate_temperature": 1000,
                "dtype": "float64",
                "lr": 0.001,
                "weight_decay": 0.0,
                "batch_size": 256,
                "epochs": 500,
                "patience": 50,
            },
            "cs-Cox": {"penalizer": 0.01},
            "NeuralFG": {
                "layers": [100, 100, 100],
                "layers_surv": [100],
                "dropout": 0.0,
                "lr": 0.001,
                "weight_decay": 0.001,
                "batch_size": 100,
                "epochs": 1000,
                "patience": 3,
            },
            "SoftComp": asdict(config.softcomp),
        },
        "softcomp_loss": "ell_1 only",
        "softcomp_augmentation_execution": "one network call per augmented time",
        "softcomp_postprocessing": "PAV followed by global-time simplex rescaling",
        "dist": "mean(abs(sum(final CIF) + native survival - 1))",
        "negative_survival": "1 - sum(final CIF)",
        "evaluation_grid": "100 requested test event-time 0--90% quantiles",
        "plot_cohort": {
            "seed": PLOT_COHORT_SEED,
            "subjects": PLOT_SUBJECTS,
            "time_points": PLOT_TIME_POINTS,
            "time_min": 0.0,
            "time_max": PLOT_TIME_MAX,
        },
    }


def compute_configuration_hash(config: ExperimentConfig) -> str:
    """Return the hash of settings that must match across both experiment phases."""
    encoded = json.dumps(
        _frozen_protocol_payload(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def _prepare_data(
    replicate: int,
    config: ExperimentConfig,
    parameters: dict[str, Tensor],
) -> PreparedData:
    generated = case2_v3.generate_data(
        n=5000,
        K=K,
        p=P,
        censor_rate=0.5,
        seed=config.train_seed_base + replicate,
        params=parameters,
    )
    true_cif_fn = functools.partial(case2_v3.compute_cif, params=parameters)
    x_test, y_test, delta_test = generate_test_observations(
        1000,
        P,
        true_cif_fn,
        censor_rate=0.5,
        seed=config.test_seed_base + replicate,
        t_max=case2_v3.T_MAX,
    )
    eval_times = build_evaluation_time_grid(y_test, delta_test, n_grid=100)
    n_val = 500
    return PreparedData(
        X_train_full=generated["X"],
        Y_train_full=generated["Y"],
        Delta_train_full=generated["Delta"],
        T_train_true=generated["T_true"],
        X_train_fit=generated["X"][n_val:],
        Y_train_fit=generated["Y"][n_val:],
        Delta_train_fit=generated["Delta"][n_val:],
        X_val=generated["X"][:n_val],
        Y_val=generated["Y"][:n_val],
        Delta_val=generated["Delta"][:n_val],
        X_test=x_test,
        Y_test=y_test,
        Delta_test=delta_test,
        eval_times=eval_times,
        params=parameters,
        true_cif_fn=true_cif_fn,
        case_name="case2_v3",
        t_max=case2_v3.T_MAX,
    )


def _true_probability_grid(
    x: Tensor,
    times: Tensor,
    true_cif_fn: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    cif = torch.empty(x.shape[0], K, len(times))
    survival = torch.empty(x.shape[0], len(times))
    with torch.no_grad():
        for time_index, evaluation_time in enumerate(times):
            values, survival_values = true_cif_fn(
                x,
                evaluation_time.expand(x.shape[0]),
            )
            cif[:, :, time_index] = values
            survival[:, time_index] = survival_values
    return cif, survival


def make_plot_cohort(
    parameters: dict[str, Tensor],
    seed: int = PLOT_COHORT_SEED,
) -> PlotCohort:
    """Build the fixed subjects and grid shared by every fitted model."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(PLOT_SUBJECTS, P, generator=generator)
    times = torch.linspace(0.0, PLOT_TIME_MAX, PLOT_TIME_POINTS)
    true_cif_fn = functools.partial(case2_v3.compute_cif, params=parameters)
    true_cif, true_survival = _true_probability_grid(x, times, true_cif_fn)
    return PlotCohort(x, times, true_cif, true_survival)


def _finalize_cif(spec: ModelSpec, cif: Tensor, times: Tensor) -> Tensor:
    if spec.cache_key != "crsoft":
        return cif
    return enforce_cif_simplex(isotonic_project_cif(cif, times))


def _evaluate_predictions(
    cif: Tensor,
    native_survival: Tensor,
    data: PreparedData,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = compute_mse_accuracy(
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
    metrics.update(
        compute_dist(
            cif,
            native_survival,
            negative_survival_atol=NEGATIVE_SURVIVAL_ATOL,
        )
    )
    return metrics


def _probability_diagnostics(
    cif: Tensor,
    native_survival: Tensor,
) -> tuple[dict[str, float | int], Tensor, Tensor]:
    diagnostics = compute_dist(
        cif,
        native_survival,
        negative_survival_atol=NEGATIVE_SURVIVAL_ATOL,
    )
    implied_survival = 1.0 - cif.sum(dim=1)
    return (
        diagnostics,
        implied_survival < 0.0,
        implied_survival < -NEGATIVE_SURVIVAL_ATOL,
    )


def _method_is_complete(
    method_dir: Path,
    configuration_hash: str,
    method_configuration_hash: str | None = None,
) -> bool:
    complete_path = method_dir / "complete.json"
    if not complete_path.exists():
        return False
    payload = _read_json(complete_path)
    complete = (
        payload.get("complete") is True
        and payload.get("configuration_hash") == configuration_hash
    )
    if method_configuration_hash is not None:
        complete = complete and (
            payload.get("method_configuration_hash") == method_configuration_hash
        )
    return complete


def _replicate_is_complete(
    replicate_dir: Path,
    config: ExperimentConfig,
    replicate: int,
    configuration_hash: str,
) -> bool:
    complete_path = replicate_dir / "complete.json"
    metrics_path = replicate_dir / "metrics.json"
    if not complete_path.exists() or not metrics_path.exists():
        return False
    complete = _read_json(complete_path)
    if (
        complete.get("complete") is not True
        or complete.get("configuration_hash") != configuration_hash
    ):
        return False
    expected_methods = set(expected_methods_for_replicate(config, replicate))
    completed_methods = set(complete.get("completed_methods", []))
    recorded_expected_methods = set(complete.get("expected_methods", []))
    metric_methods = set(_read_json(metrics_path))
    if (
        completed_methods != expected_methods
        or recorded_expected_methods != expected_methods
        or metric_methods != expected_methods
    ):
        return False
    for method in expected_methods:
        method_dir = replicate_dir / "methods" / METHOD_CACHE_KEYS[method]
        method_configuration_hash = (
            fine_gray_amendment_hash() if method == "Fine-Gray" else None
        )
        if not _method_is_complete(
            method_dir,
            configuration_hash,
            method_configuration_hash,
        ):
            return False
        if any(
            not (method_dir / filename).is_file()
            for filename in REQUIRED_METHOD_ARTIFACTS
        ):
            return False
    return True


def _save_checkpoint(
    method_dir: Path,
    spec: ModelSpec,
    model: object,
    config: ExperimentConfig,
    configuration_hash: str,
) -> None:
    checkpoint = spec.serialize(model)
    checkpoint.update(
        {
            "model_seed": config.model_seed,
            "configuration_hash": configuration_hash,
        }
    )
    if spec.cache_key == "fine_gray":
        checkpoint["method_configuration_hash"] = fine_gray_amendment_hash()
    _write_torch(method_dir / "checkpoint.pt", checkpoint)
    history = _training_history(model)
    _write_torch(
        method_dir / "training_history.pt",
        {
            "applicable": spec.cache_key not in {"cs_cox", "fine_gray"},
            "loss": history,
            "epochs_completed": len(history),
        },
    )


def _save_method_predictions(
    method_dir: Path,
    test_cif: Tensor,
    test_native_survival: Tensor,
    plot_cif: Tensor,
    plot_native_survival: Tensor,
) -> None:
    _write_torch(
        method_dir / "test_predictions.pt",
        {
            "final_cif": test_cif.detach().cpu().float(),
            "native_survival": test_native_survival.detach().cpu().float(),
            "implied_survival": (1.0 - test_cif.sum(dim=1)).detach().cpu().float(),
        },
    )
    _write_torch(
        method_dir / "plot_predictions.pt",
        {
            "final_cif": plot_cif.detach().cpu().float(),
            "native_survival": plot_native_survival.detach().cpu().float(),
            "implied_survival": (1.0 - plot_cif.sum(dim=1)).detach().cpu().float(),
        },
    )


def _run_method(
    directory: Path,
    spec: ModelSpec,
    data: PreparedData,
    plot_cohort: PlotCohort,
    config: ExperimentConfig,
    configuration_hash: str,
) -> tuple[dict[str, float | int], dict[str, float], dict[str, float | int]]:
    method_dir = directory / "methods" / spec.cache_key
    method_configuration_hash = (
        fine_gray_amendment_hash() if spec.cache_key == "fine_gray" else None
    )
    if _method_is_complete(
        method_dir,
        configuration_hash,
        method_configuration_hash,
    ):
        print(f"[{spec.name}] loading completed partial result", flush=True)
        return (
            _read_json(method_dir / "metrics.json"),
            _read_json(method_dir / "timings.json"),
            _read_json(method_dir / "all_subject_probability_diagnostics.json"),
        )

    print(f"[{spec.name}] training", flush=True)
    torch.manual_seed(config.model_seed)
    train_start = time.perf_counter()
    model = spec.train()
    train_time = time.perf_counter() - train_start

    if spec.predict_survival is None:
        raise RuntimeError(f"{spec.name} does not expose native survival")
    predict_start = time.perf_counter()
    test_raw_cif, test_native_survival = spec.predict_survival(
        model,
        data.X_test,
        data.eval_times,
    )
    predict_time = time.perf_counter() - predict_start
    postprocess_start = time.perf_counter()
    test_cif = _finalize_cif(spec, test_raw_cif, data.eval_times)
    postprocess_time = time.perf_counter() - postprocess_start
    if spec.cache_key != "crsoft":
        postprocess_time = 0.0

    metrics = _evaluate_predictions(test_cif, test_native_survival, data)

    diagnostic_start = time.perf_counter()
    train_raw_cif, train_native_survival = spec.predict_survival(
        model,
        data.X_train_full,
        data.eval_times,
    )
    train_cif = _finalize_cif(spec, train_raw_cif, data.eval_times)
    all_cif = torch.cat([train_cif, test_cif], dim=0)
    all_native_survival = torch.cat(
        [train_native_survival, test_native_survival],
        dim=0,
    )
    all_diagnostics, negative_mask, below_tolerance_mask = _probability_diagnostics(
        all_cif,
        all_native_survival,
    )
    diagnostic_time = time.perf_counter() - diagnostic_start

    plot_start = time.perf_counter()
    plot_raw_cif, plot_native_survival = spec.predict_survival(
        model,
        plot_cohort.X,
        plot_cohort.times,
    )
    plot_cif = _finalize_cif(spec, plot_raw_cif, plot_cohort.times)
    plot_time = time.perf_counter() - plot_start

    timings = {
        "train_time_sec": train_time,
        "predict_time_sec": predict_time,
        "pav_time_sec": postprocess_time,
        "total_model_time_sec": train_time + predict_time + postprocess_time,
        "diagnostic_prediction_time_sec": diagnostic_time,
        "plot_prediction_time_sec": plot_time,
    }
    _save_checkpoint(method_dir, spec, model, config, configuration_hash)
    _save_method_predictions(
        method_dir,
        test_cif,
        test_native_survival,
        plot_cif,
        plot_native_survival,
    )
    if bool(negative_mask.any().item()):
        _write_torch(
            method_dir / "negative_survival_masks.pt",
            {
                "negative_mask": negative_mask.cpu(),
                "below_tolerance_mask": below_tolerance_mask.cpu(),
            },
        )
    _write_json(method_dir / "metrics.json", metrics)
    _write_json(
        method_dir / "all_subject_probability_diagnostics.json",
        all_diagnostics,
    )
    _write_json(method_dir / "timings.json", timings)
    complete_payload = {
        "complete": True,
        "configuration_hash": configuration_hash,
    }
    if method_configuration_hash is not None:
        complete_payload["method_configuration_hash"] = method_configuration_hash
    _write_json(method_dir / "complete.json", complete_payload)
    print(
        f"[{spec.name}] MSE={metrics['MSE_overall']:.6f} "
        f"Ctd={metrics['Ctd_overall']:.6f} "
        f"IBS={metrics['IBS_overall']:.6f} "
        f"Dist={metrics['Dist']:.6f} "
        f"negative subjects="
        f"{all_diagnostics['Implied_S_negative_subject_count']}/"
        f"{all_diagnostics['Implied_S_total_subject_count']} "
        f"train={train_time:.2f}s",
        flush=True,
    )
    return metrics, timings, all_diagnostics


def _replicate_manifest(
    replicate: int,
    data: PreparedData,
    config: ExperimentConfig,
    configuration_hash: str,
) -> dict[str, object]:
    return {
        "replicate": replicate,
        "train_seed": config.train_seed_base + replicate,
        "test_seed": config.test_seed_base + replicate,
        "model_seed": config.model_seed,
        "dgp_seed": config.dgp_seed,
        "configuration_hash": configuration_hash,
        "expected_methods": list(expected_methods_for_replicate(config, replicate)),
        "train_counts_censor_then_causes": [
            int((data.Delta_train_full == cause).sum().item()) for cause in range(K + 1)
        ],
        "test_counts_censor_then_causes": [
            int((data.Delta_test == cause).sum().item()) for cause in range(K + 1)
        ],
        "evaluation_time_count": len(data.eval_times),
        "evaluation_time_min": float(data.eval_times[0].item()),
        "evaluation_time_max": float(data.eval_times[-1].item()),
        "train_event_time_at_limit_fraction": float(
            (data.T_train_true >= data.t_max - 1e-3).float().mean().item()
        ),
    }


def _save_shared_data(
    directory: Path,
    data: PreparedData,
    true_cif: Tensor,
    true_survival: Tensor,
) -> None:
    _write_torch(
        directory / "shared_data.pt",
        {
            "X_train": data.X_train_full,
            "Y_train": data.Y_train_full,
            "Delta_train": data.Delta_train_full,
            "X_test": data.X_test,
            "Y_test": data.Y_test,
            "Delta_test": data.Delta_test,
            "eval_times": data.eval_times,
            "true_cif": true_cif,
            "true_survival": true_survival,
            "parameters": data.params,
            "validation_indices": torch.arange(500),
            "fitting_indices": torch.arange(500, 5000),
        },
    )


def _run_replicate(
    replicate: int,
    output_dir: Path,
    config: ExperimentConfig,
    parameters: dict[str, Tensor],
    plot_cohort: PlotCohort,
    configuration_hash: str,
    resume: bool,
) -> bool:
    replicates_dir = output_dir / "replicates"
    replicates_dir.mkdir(parents=True, exist_ok=True)
    final_dir = replicates_dir / f"rep_{replicate:03d}"
    if final_dir.exists():
        if resume and _replicate_is_complete(
            final_dir,
            config,
            replicate,
            configuration_hash,
        ):
            print(f"Replicate {replicate:03d}: already complete, skipping")
            return False
        raise FileExistsError(f"replicate output already exists: {final_dir}")

    working_dir = replicates_dir / f".rep_{replicate:03d}.in_progress"
    if working_dir.exists() and not resume:
        raise FileExistsError(
            f"partial replicate exists; rerun with --resume: {working_dir}"
        )
    working_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = working_dir / "replicate_manifest.json"
    if existing_manifest.exists():
        previous = _read_json(existing_manifest)
        if previous.get("configuration_hash") != configuration_hash:
            raise RuntimeError(f"configuration changed for {working_dir}")
    print(f"\n{'=' * 72}\nReplicate {replicate:03d}\n{'=' * 72}", flush=True)
    data = _prepare_data(replicate, config, parameters)
    true_cif, true_survival = _true_probability_grid(
        data.X_test,
        data.eval_times,
        data.true_cif_fn,
    )
    if not (working_dir / "shared_data.pt").exists():
        _save_shared_data(
            working_dir,
            data,
            true_cif,
            true_survival,
        )
    _write_json(
        working_dir / "replicate_manifest.json",
        _replicate_manifest(replicate, data, config, configuration_hash),
    )

    metrics: dict[str, dict[str, float | int]] = {}
    timings: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, dict[str, float | int]] = {}
    expected_methods = expected_methods_for_replicate(config, replicate)
    specs = _select_specs(
        _build_specs(data, config.softcomp, config.model_seed),
        list(expected_methods),
    )
    for spec in specs:
        method_metrics, method_timings, method_diagnostics = _run_method(
            working_dir,
            spec,
            data,
            plot_cohort,
            config,
            configuration_hash,
        )
        name = "SoftComp" if spec.cache_key == "crsoft" else spec.name
        metrics[name] = method_metrics
        timings[name] = method_timings
        diagnostics[name] = method_diagnostics

    if set(metrics) != set(expected_methods):
        raise RuntimeError(
            f"replicate {replicate:03d} completed {sorted(metrics)}, "
            f"expected {sorted(expected_methods)}"
        )

    _write_json(working_dir / "metrics.json", metrics)
    _write_json(working_dir / "timings.json", timings)
    _write_json(
        working_dir / "all_subject_probability_diagnostics.json",
        diagnostics,
    )
    _write_json(
        working_dir / "complete.json",
        {
            "complete": True,
            "configuration_hash": configuration_hash,
            "expected_methods": list(expected_methods),
            "completed_methods": list(metrics),
        },
    )
    working_dir.replace(final_dir)
    return True


def _completed_replicate_payloads(
    output_dir: Path,
    config: ExperimentConfig,
    configuration_hash: str,
) -> dict[int, dict[str, dict[str, float | int]]]:
    payloads = {}
    for replicate in range(config.n_replicates):
        replicate_dir = output_dir / "replicates" / f"rep_{replicate:03d}"
        if _replicate_is_complete(
            replicate_dir,
            config,
            replicate,
            configuration_hash,
        ):
            payloads[replicate] = _read_json(replicate_dir / "metrics.json")
    return payloads


def _long_rows(
    output_dir: Path,
    config: ExperimentConfig,
    configuration_hash: str,
    filename: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for replicate in range(config.n_replicates):
        replicate_dir = output_dir / "replicates" / f"rep_{replicate:03d}"
        if not _replicate_is_complete(
            replicate_dir,
            config,
            replicate,
            configuration_hash,
        ):
            continue
        path = replicate_dir / filename
        if not path.exists():
            continue
        expected_methods = set(expected_methods_for_replicate(config, replicate))
        for method, values in _read_json(path).items():
            if method not in expected_methods:
                continue
            for metric, value in values.items():
                rows.append(
                    {
                        "replicate": replicate,
                        "method": method,
                        "metric": metric,
                        "value": value,
                    }
                )
    return rows


def _summary_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["metric"]))
        grouped.setdefault(key, []).append(float(row["value"]))
    return [
        {
            "method": method,
            "metric": metric,
            "n": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
        }
        for (method, metric), values in sorted(grouped.items())
    ]


def development_gate_report(
    payloads: dict[int, dict[str, dict[str, float | int]]],
    configuration_hash: str,
) -> dict[str, object]:
    """Evaluate the pre-registered cs-Cox and SoftComp development gates."""
    complete = len(payloads) == 4 and all(
        set(method_metrics) == set(DEVELOPMENT_MODELS)
        for method_metrics in payloads.values()
    )
    if not complete:
        return {
            "passed": False,
            "complete": False,
            "configuration_hash": configuration_hash,
            "completed_replicates": sorted(payloads),
            "reason": "all five methods must complete all four development replicates",
        }

    core_metrics = ("MSE_overall", "Ctd_overall", "IBS_overall")
    means: dict[str, dict[str, float]] = {}
    ranks: dict[str, dict[str, int]] = {}
    for metric in core_metrics:
        means[metric] = {
            method: statistics.mean(
                float(payloads[replicate][method][metric])
                for replicate in sorted(payloads)
            )
            for method in DEVELOPMENT_MODELS
        }
        reverse = metric == "Ctd_overall"
        ordered = sorted(means[metric], key=means[metric].get, reverse=reverse)
        ranks[metric] = {method: rank for rank, method in enumerate(ordered, start=1)}

    cscox_ctd = [
        float(payloads[replicate]["cs-Cox"]["Ctd_overall"])
        for replicate in sorted(payloads)
    ]
    cscox_passed = statistics.mean(cscox_ctd) <= 0.55 and max(cscox_ctd) <= 0.60
    softcomp_passed = all(ranks[metric]["SoftComp"] <= 2 for metric in core_metrics)
    return {
        "passed": cscox_passed and softcomp_passed,
        "complete": True,
        "configuration_hash": configuration_hash,
        "completed_replicates": sorted(payloads),
        "core_metric_means": means,
        "core_metric_ranks": ranks,
        "cscox": {
            "ctd_values": cscox_ctd,
            "mean_ctd": statistics.mean(cscox_ctd),
            "max_ctd": max(cscox_ctd),
            "mean_threshold": 0.55,
            "maximum_threshold": 0.60,
            "passed": cscox_passed,
        },
        "softcomp": {
            "ranks": {metric: ranks[metric]["SoftComp"] for metric in core_metrics},
            "maximum_allowed_rank": 2,
            "passed": softcomp_passed,
        },
    }


def import_development_results(
    result_specs: Sequence[str],
    config: ExperimentConfig,
    configuration_hash: str,
) -> dict[str, object]:
    """Merge partial single-run JSON files and evaluate the development gate."""
    payloads: dict[int, dict[str, dict[str, float | int]]] = {}
    sources = []
    for result_spec in result_specs:
        try:
            replicate_text, path_text = result_spec.split("=", 1)
            replicate = int(replicate_text)
        except ValueError as error:
            raise ValueError(
                f"invalid --gate-result '{result_spec}'; expected REPLICATE=PATH"
            ) from error
        if replicate not in range(4):
            raise ValueError(f"development replicate must be 0--3, got {replicate}")
        path = Path(path_text)
        payload = _read_json(path)
        protocol = payload.get("protocol", {})
        expected_protocol = {
            "n_train": 5000,
            "n_test": 1000,
            "train_seed": config.train_seed_base + replicate,
            "test_seed": config.test_seed_base + replicate,
            "model_seed": config.model_seed,
            "dgp_seed": config.dgp_seed,
        }
        for name, expected in expected_protocol.items():
            if protocol.get(name) != expected:
                raise ValueError(
                    f"{path}: protocol {name}={protocol.get(name)!r}, "
                    f"expected {expected!r}"
                )
        if protocol.get("softcomp") != asdict(config.softcomp):
            raise ValueError(f"{path}: SoftComp configuration does not match")
        for method, method_payload in payload.get("results", {}).items():
            if method not in DEVELOPMENT_MODELS:
                raise ValueError(f"{path}: unexpected method {method}")
            replicate_metrics = payloads.setdefault(replicate, {})
            if method in replicate_metrics:
                raise ValueError(
                    f"duplicate result for replicate {replicate}, method {method}"
                )
            replicate_metrics[method] = method_payload["metrics"]
        sources.append(
            {
                "replicate": replicate,
                "path": str(path),
                "methods": sorted(payload.get("results", {})),
            }
        )
    report = development_gate_report(payloads, configuration_hash)
    report["sources"] = sources
    return report


def negative_survival_summary(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate exact implied-survival counts across paired replicates."""
    by_method: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    summaries = []
    for method, method_rows in sorted(by_method.items()):
        values: dict[str, list[float]] = {}
        for row in method_rows:
            values.setdefault(str(row["metric"]), []).append(float(row["value"]))

        replicate_count = len({int(row["replicate"]) for row in method_rows})

        negative_count = int(sum(values.get("Implied_S_negative_count", [])))
        total_count = int(sum(values.get("Implied_S_total_count", [])))
        negative_subject_count = int(
            sum(values.get("Implied_S_negative_subject_count", []))
        )
        total_subject_count = int(sum(values.get("Implied_S_total_subject_count", [])))
        tolerance_count = int(sum(values.get("Implied_S_below_tolerance_count", [])))
        tolerance_subject_count = int(
            sum(values.get("Implied_S_below_tolerance_subject_count", []))
        )
        minima = values.get("Implied_S_min", [])
        replicates_with_negative_survival = sum(
            count > 0 for count in values.get("Implied_S_negative_count", [])
        )
        summaries.append(
            {
                "method": method,
                "n": replicate_count,
                "negative_point_count": negative_count,
                "total_point_count": total_count,
                "negative_point_fraction": (
                    negative_count / total_count if total_count else 0.0
                ),
                "negative_subject_count": negative_subject_count,
                "total_subject_count": total_subject_count,
                "negative_subject_fraction": (
                    negative_subject_count / total_subject_count
                    if total_subject_count
                    else 0.0
                ),
                "tolerance_negative_point_count": tolerance_count,
                "tolerance_negative_subject_count": tolerance_subject_count,
                "replicates_with_negative_survival": (
                    replicates_with_negative_survival
                ),
                "replicate_fraction_with_negative_survival": (
                    replicates_with_negative_survival / replicate_count
                    if replicate_count
                    else 0.0
                ),
                "global_minimum_implied_survival": min(minima) if minima else 0.0,
            }
        )
    return summaries


def _aggregate_plot_predictions(
    output_dir: Path,
    config: ExperimentConfig,
    configuration_hash: str,
) -> None:
    aggregated: dict[str, object] = {}
    for method, cache_key in METHOD_CACHE_KEYS.items():
        predictions = []
        replicates = []
        for replicate in range(config.n_replicates):
            replicate_dir = output_dir / "replicates" / f"rep_{replicate:03d}"
            if method not in expected_methods_for_replicate(
                config, replicate
            ) or not _replicate_is_complete(
                replicate_dir,
                config,
                replicate,
                configuration_hash,
            ):
                continue
            path = replicate_dir / "methods" / cache_key / "plot_predictions.pt"
            if not path.exists():
                continue
            payload = torch.load(path, weights_only=False)
            predictions.append(payload["final_cif"].float())
            replicates.append(replicate)
        if not predictions:
            continue
        stacked = torch.stack(predictions)
        aggregated[method] = {
            "replicates": replicates,
            "final_cif": stacked,
            "mean": stacked.mean(dim=0),
            "std": (
                stacked.std(dim=0, unbiased=True)
                if len(predictions) > 1
                else torch.zeros_like(stacked[0])
            ),
        }
    if aggregated:
        _write_torch(
            output_dir / "aggregate" / "plot_predictions_all.pt",
            aggregated,
        )


def _aggregate(
    output_dir: Path,
    config: ExperimentConfig,
    configuration_hash: str,
) -> dict[str, object] | None:
    aggregate_dir = output_dir / "aggregate"
    metrics_rows = _long_rows(
        output_dir,
        config,
        configuration_hash,
        "metrics.json",
    )
    timing_rows = _long_rows(
        output_dir,
        config,
        configuration_hash,
        "timings.json",
    )
    diagnostic_rows = _long_rows(
        output_dir,
        config,
        configuration_hash,
        "all_subject_probability_diagnostics.json",
    )
    long_fields = ["replicate", "method", "metric", "value"]
    summary_fields = ["method", "metric", "n", "mean", "std"]
    _write_csv(aggregate_dir / "metrics_long.csv", long_fields, metrics_rows)
    _write_csv(
        aggregate_dir / "metrics_summary.csv",
        summary_fields,
        _summary_rows(metrics_rows),
    )
    _write_csv(aggregate_dir / "timings_long.csv", long_fields, timing_rows)
    _write_csv(
        aggregate_dir / "timing_summary.csv",
        summary_fields,
        _summary_rows(timing_rows),
    )
    _write_csv(
        aggregate_dir / "negative_survival_long.csv",
        long_fields,
        diagnostic_rows,
    )
    _write_csv(
        aggregate_dir / "negative_survival_metric_summary.csv",
        summary_fields,
        _summary_rows(diagnostic_rows),
    )
    negative_summary = negative_survival_summary(diagnostic_rows)
    if negative_summary:
        _write_csv(
            aggregate_dir / "negative_survival_summary.csv",
            list(negative_summary[0]),
            negative_summary,
        )
    _aggregate_plot_predictions(output_dir, config, configuration_hash)

    payloads = _completed_replicate_payloads(
        output_dir,
        config,
        configuration_hash,
    )
    completed = sorted(payloads)
    expected_replicates_by_method = {
        method: [
            replicate
            for replicate in range(config.n_replicates)
            if method in expected_methods_for_replicate(config, replicate)
        ]
        for method in config.models
    }
    completed_replicates_by_method = {
        method: [
            replicate
            for replicate, method_metrics in sorted(payloads.items())
            if method in method_metrics
        ]
        for method in config.models
    }
    manifest = {
        "phase": config.phase,
        "configuration_hash": configuration_hash,
        "formal_replicates_by_method": expected_method_replicate_counts(config),
        "completed_replicates": completed,
        "missing_replicates": [
            replicate
            for replicate in range(config.n_replicates)
            if replicate not in payloads
        ],
        "methods": {
            method: {
                "expected_count": len(expected_replicates_by_method[method]),
                "completed_count": len(completed_replicates_by_method[method]),
                "expected_replicates": expected_replicates_by_method[method],
                "completed_replicates": completed_replicates_by_method[method],
            }
            for method in config.models
        },
        "environment": _environment_payload(),
        **_formal_schedule_metadata(config),
    }
    _write_json(output_dir / "manifest.json", manifest)
    if config.phase != "development":
        return None
    gate_report = development_gate_report(payloads, configuration_hash)
    _write_json(aggregate_dir / "development_gate.json", gate_report)
    return gate_report


def validate_formal_gate(path: Path, configuration_hash: str) -> None:
    """Reject formal execution unless the matching development gate passed."""
    if not path.exists():
        raise FileNotFoundError(f"development gate file not found: {path}")
    gate = _read_json(path)
    if gate.get("passed") is not True:
        raise RuntimeError("development gate has not passed")
    if gate.get("configuration_hash") != configuration_hash:
        raise RuntimeError(
            "development gate configuration does not match formal configuration"
        )


def _default_output_dir(phase: str) -> Path:
    _, _, _, directory = _phase_defaults(phase)
    return get_output_dir("case2_v3") / directory


def _save_plot_cohort(
    output_dir: Path,
    cohort: PlotCohort,
    configuration_hash: str,
) -> None:
    path = output_dir / "plot_cohort.pt"
    if path.exists():
        payload = torch.load(path, weights_only=False)
        if payload.get("configuration_hash") != configuration_hash:
            raise RuntimeError("plot cohort configuration does not match this run")
        return
    _write_torch(
        path,
        {
            "seed": PLOT_COHORT_SEED,
            "configuration_hash": configuration_hash,
            "X": cohort.X,
            "times": cohort.times,
            "true_cif": cohort.true_cif,
            "true_survival": cohort.true_survival,
        },
    )


def main() -> None:
    args = _parse_args()
    config = _build_config(args)
    torch.set_num_threads(config.cpu_threads)
    output_dir = args.output_dir or _default_output_dir(config.phase)
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration_hash = compute_configuration_hash(config)
    config_path = output_dir / "config.json"
    if config_path.exists():
        previous_config = _read_json(config_path)
        if previous_config.get("configuration_hash") != configuration_hash:
            raise RuntimeError("output directory contains a different configuration")
    if config.phase == "formal":
        gate_file = args.gate_file or (
            _default_output_dir("development") / "aggregate" / "development_gate.json"
        )
        validate_formal_gate(gate_file, configuration_hash)

    _write_json(
        config_path,
        {
            "execution": asdict(config),
            "frozen_protocol": frozen_protocol_payload(config),
            "configuration_hash": configuration_hash,
            **_formal_schedule_metadata(config),
        },
    )
    if args.gate_result:
        if config.phase != "development":
            raise ValueError("--gate-result is only valid for development phase")
        report = import_development_results(
            args.gate_result,
            config,
            configuration_hash,
        )
        _write_json(output_dir / "aggregate" / "development_gate.json", report)
        print(
            f"Imported development gate passed={report['passed']} "
            f"complete={report['complete']}",
            flush=True,
        )
        return
    parameters = case2_v3.generate_parameters(K=K, p=P, seed=config.dgp_seed)
    plot_cohort = make_plot_cohort(parameters)
    _save_plot_cohort(output_dir, plot_cohort, configuration_hash)

    for replicate in range(config.start_replicate, config.end_replicate):
        _run_replicate(
            replicate,
            output_dir,
            config,
            parameters,
            plot_cohort,
            configuration_hash,
            args.resume,
        )
        gate_report = _aggregate(output_dir, config, configuration_hash)
        if gate_report is not None:
            print(
                f"Development gate passed={gate_report['passed']} "
                f"after {len(gate_report['completed_replicates'])}/4 replicates",
                flush=True,
            )
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
