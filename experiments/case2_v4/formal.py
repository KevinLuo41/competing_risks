#!/usr/bin/env python3
"""Development gate and 50-replicate formal runner for Case II v4."""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import logging
import os
import platform
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from ...data import case2_v4
from ...evaluation import compute_mse_accuracy, evaluate_cif_metrics, get_output_dir
from ...evaluation.simulation import compute_dist
from ..runner import ModelSpec
from .run import (
    build_specs,
    K,
    P,
    prepare_data,
    PreparedData,
    SoftCompConfig,
    training_history,
)

logger = logging.getLogger(__name__)

ALL_MODELS = ("DeepHit", "DSM", "cs-Cox", "NeuralFG", "SoftComp", "Fine-Gray")
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
REQUIRED_REPLICATE_ARTIFACTS = (
    "shared_data.pt",
    "metrics.json",
    "timings.json",
    "all_subject_probability_diagnostics.json",
    "replicate_manifest.json",
    "complete.json",
)

N_TRAIN = 5000
N_TEST = 1000
N_VALIDATION = 500
CENSOR_RATE = 0.5
MODEL_SEED = 0
DGP_SEED = 42
DEVELOPMENT_REPLICATES = 4
FORMAL_REPLICATES = 50
DEVELOPMENT_TRAIN_SEED_BASE = 110_000
DEVELOPMENT_TEST_SEED_BASE = 120_000
FORMAL_TRAIN_SEED_BASE = 130_000
FORMAL_TEST_SEED_BASE = 140_000
PLOT_COHORT_SEED = 150_000
PLOT_SUBJECTS = 5
PLOT_TIME_POINTS = 100
PLOT_TIME_MAX = 30.0
NEGATIVE_SURVIVAL_ATOL = 1e-6


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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-3)
    parser.add_argument("--n-aug", type=int, default=2)
    parser.add_argument("--aug-weight", type=float, default=0.5)
    return parser.parse_args()


def _phase_defaults(phase: str) -> tuple[int, int, int, str]:
    if phase == "development":
        return (
            DEVELOPMENT_REPLICATES,
            DEVELOPMENT_TRAIN_SEED_BASE,
            DEVELOPMENT_TEST_SEED_BASE,
            "development_gate",
        )
    if phase == "formal":
        return (
            FORMAL_REPLICATES,
            FORMAL_TRAIN_SEED_BASE,
            FORMAL_TEST_SEED_BASE,
            "formal_50",
        )
    return (1, 105_000, 115_000, "smoke")


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    default_n, default_train_seed, default_test_seed, _ = _phase_defaults(args.phase)
    n_replicates = default_n if args.n_replicates is None else args.n_replicates
    train_seed_base = (
        default_train_seed if args.train_seed_base is None else args.train_seed_base
    )
    test_seed_base = (
        default_test_seed if args.test_seed_base is None else args.test_seed_base
    )
    end_replicate = n_replicates if args.end_replicate is None else args.end_replicate
    models = tuple(args.models or ALL_MODELS)
    if n_replicates <= 0 or args.cpu_threads <= 0:
        raise ValueError("replicate and CPU-thread counts must be positive")
    if not 0 <= args.start_replicate < end_replicate <= n_replicates:
        raise ValueError("replicate range is outside the configured schedule")
    if models != ALL_MODELS:
        raise ValueError("the formal runner requires all six pre-registered methods")
    if args.phase in {"development", "formal"} and args.cpu_threads != 8:
        raise ValueError("development and formal phases require eight CPU threads")
    if args.phase == "development":
        expected = (
            DEVELOPMENT_REPLICATES,
            DEVELOPMENT_TRAIN_SEED_BASE,
            DEVELOPMENT_TEST_SEED_BASE,
            ALL_MODELS,
        )
        actual = (n_replicates, train_seed_base, test_seed_base, models)
        if actual != expected:
            raise ValueError("development requires the frozen four-replicate schedule")
    if args.phase == "formal":
        expected = (
            FORMAL_REPLICATES,
            FORMAL_TRAIN_SEED_BASE,
            FORMAL_TEST_SEED_BASE,
            ALL_MODELS,
        )
        actual = (n_replicates, train_seed_base, test_seed_base, models)
        if actual != expected:
            raise ValueError("formal requires all six methods on 50 replicates")
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


def _frozen_protocol_payload(config: ExperimentConfig) -> dict[str, object]:
    return {
        "case": "Case II v4",
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "n_validation": N_VALIDATION,
        "n_causes": K,
        "n_features": P,
        "censor_rate": CENSOR_RATE,
        "dgp_seed": config.dgp_seed,
        "model_seed": config.model_seed,
        "development_schedule": {
            "replicates": DEVELOPMENT_REPLICATES,
            "train_seed_base": DEVELOPMENT_TRAIN_SEED_BASE,
            "test_seed_base": DEVELOPMENT_TEST_SEED_BASE,
        },
        "formal_schedule": {
            "replicates": FORMAL_REPLICATES,
            "train_seed_base": FORMAL_TRAIN_SEED_BASE,
            "test_seed_base": FORMAL_TEST_SEED_BASE,
        },
        "formal_replicates_by_method": dict.fromkeys(ALL_MODELS, FORMAL_REPLICATES),
        "dgp": {
            "beta_distribution": "Uniform(0.05, 0.15), drawn once",
            "intercept": [-4.1, -4.0, -3.9],
            "quadratic_weight": 1.5,
            "pairwise_weight": 1.0,
            "time_scale": 12.0,
            "t_max": case2_v4.T_MAX,
        },
        "models": {
            "DeepHit": {
                "n_bins": 100,
                "hidden_dim": 64,
                "n_layers": 2,
                "dropout": 0.1,
                "lr": 1e-3,
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
                "lr": 1e-3,
                "batch_size": 256,
                "epochs": 500,
                "patience": 50,
            },
            "cs-Cox": {"penalizer": 0.01},
            "NeuralFG": {
                "layers": [100, 100, 100],
                "layers_surv": [100],
                "lr": 1e-3,
                "weight_decay": 1e-3,
                "batch_size": 100,
                "epochs": 1000,
                "patience": 3,
            },
            "SoftComp": asdict(config.softcomp),
            "Fine-Gray": {
                "penalizer": 0.0,
                "max_iter": 100,
                "tolerance": 1e-7,
                "feature_scaling": "training mean and population standard deviation",
                "censoring": "pooled Kaplan-Meier IPCW with all subjects at risk at ties",
                "ties": "Breslow target-event ties; strict-before competing stream",
                "prediction": "right-continuous baseline subdistribution hazard steps",
                "joint_survival": "unclamped 1 - sum of independently fitted cause CIFs",
            },
        },
        "softcomp_postprocessing": "PAV followed by global-time simplex rescaling",
        "evaluation_grid": "100 requested test event-time 0--90% quantiles",
        "negative_survival": "unclamped 1 - sum(final CIF)",
        "plot_cohort": {
            "seed": PLOT_COHORT_SEED,
            "subjects": PLOT_SUBJECTS,
            "time_points": PLOT_TIME_POINTS,
            "time_max": PLOT_TIME_MAX,
        },
    }


def compute_configuration_hash(config: ExperimentConfig) -> str:
    encoded = json.dumps(
        _frozen_protocol_payload(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_execution_hash(config: ExperimentConfig) -> str:
    payload = {
        "configuration_hash": compute_configuration_hash(config),
        "phase": config.phase,
        "n_replicates": config.n_replicates,
        "train_seed_base": config.train_seed_base,
        "test_seed_base": config.test_seed_base,
        "model_seed": config.model_seed,
        "dgp_seed": config.dgp_seed,
        "cpu_threads": config.cpu_threads,
        "models": config.models,
        "softcomp": asdict(config.softcomp),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _default_output_dir(phase: str) -> Path:
    _, _, _, name = _phase_defaults(phase)
    return get_output_dir("case2_v4") / name


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


def _true_probability_grid(
    x: Tensor,
    times: Tensor,
    parameters: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    cif = torch.empty(x.shape[0], K, len(times))
    survival = torch.empty(x.shape[0], len(times))
    with torch.no_grad():
        for index, evaluation_time in enumerate(times):
            values, survival_values = case2_v4.compute_cif(
                x,
                evaluation_time.expand(x.shape[0]),
                parameters,
            )
            cif[:, :, index] = values
            survival[:, index] = survival_values
    return cif, survival


def make_plot_cohort(parameters: dict[str, Tensor]) -> PlotCohort:
    generator = torch.Generator().manual_seed(PLOT_COHORT_SEED)
    x = torch.randn(PLOT_SUBJECTS, P, generator=generator)
    times = torch.linspace(0.0, PLOT_TIME_MAX, PLOT_TIME_POINTS)
    true_cif, true_survival = _true_probability_grid(x, times, parameters)
    return PlotCohort(x, times, true_cif, true_survival)


def _save_plot_cohort(
    output_dir: Path,
    plot_cohort: PlotCohort,
    execution_hash: str,
) -> None:
    _write_torch(
        output_dir / "plot_cohort.pt",
        {
            "execution_hash": execution_hash,
            "X": plot_cohort.X,
            "times": plot_cohort.times,
            "true_cif": plot_cohort.true_cif,
            "true_survival": plot_cohort.true_survival,
        },
    )


def _load_or_create_plot_cohort(
    output_dir: Path,
    parameters: dict[str, Tensor],
    execution_hash: str,
) -> PlotCohort:
    path = output_dir / "plot_cohort.pt"
    if not path.exists():
        cohort = make_plot_cohort(parameters)
        _save_plot_cohort(output_dir, cohort, execution_hash)
        return cohort
    payload = torch.load(path, weights_only=False)
    if payload.get("execution_hash") != execution_hash:
        raise RuntimeError("plot cohort execution hash mismatch")
    return PlotCohort(
        X=payload["X"],
        times=payload["times"],
        true_cif=payload["true_cif"],
        true_survival=payload["true_survival"],
    )


def _prepare_data(
    replicate: int,
    config: ExperimentConfig,
    parameters: dict[str, Tensor],
) -> PreparedData:
    return prepare_data(
        n_train=N_TRAIN,
        n_test=N_TEST,
        train_seed=config.train_seed_base + replicate,
        test_seed=config.test_seed_base + replicate,
        dgp_seed=config.dgp_seed,
        parameters=parameters,
    )


def _save_shared_data(
    replicate_dir: Path,
    data: PreparedData,
    execution_hash: str,
) -> None:
    _write_torch(
        replicate_dir / "shared_data.pt",
        {
            "execution_hash": execution_hash,
            "X_train": data.X_train_full,
            "Y_train": data.Y_train_full,
            "Delta_train": data.Delta_train_full,
            "T_train_true": data.T_train_true,
            "X_test": data.X_test,
            "Y_test": data.Y_test,
            "Delta_test": data.Delta_test,
            "eval_times": data.eval_times,
            "parameters": data.params,
            "validation_indices": torch.arange(N_VALIDATION),
            "fitting_indices": torch.arange(N_VALIDATION, N_TRAIN),
        },
    )


def _load_shared_data(replicate_dir: Path, execution_hash: str) -> PreparedData:
    payload = torch.load(replicate_dir / "shared_data.pt", weights_only=False)
    if payload.get("execution_hash") != execution_hash:
        raise RuntimeError("shared data execution hash mismatch")
    x_train = payload["X_train"]
    y_train = payload["Y_train"]
    delta_train = payload["Delta_train"]
    fitting_indices = payload["fitting_indices"].long()
    validation_indices = payload["validation_indices"].long()
    parameters = payload["parameters"]
    true_cif_fn = functools.partial(case2_v4.compute_cif, params=parameters)
    return PreparedData(
        X_train_full=x_train,
        Y_train_full=y_train,
        Delta_train_full=delta_train,
        T_train_true=payload["T_train_true"],
        X_train_fit=x_train[fitting_indices],
        Y_train_fit=y_train[fitting_indices],
        Delta_train_fit=delta_train[fitting_indices],
        X_val=x_train[validation_indices],
        Y_val=y_train[validation_indices],
        Delta_val=delta_train[validation_indices],
        X_test=payload["X_test"],
        Y_test=payload["Y_test"],
        Delta_test=payload["Delta_test"],
        eval_times=payload["eval_times"],
        params=parameters,
        true_cif_fn=true_cif_fn,
        case_name="case2_v4",
        t_max=case2_v4.T_MAX,
    )


def _linear_r_squared(data: PreparedData) -> Tensor:
    midpoint = data.X_train_full.shape[0] // 2
    x_fit = data.X_train_full[:midpoint]
    x_test = data.X_train_full[midpoint:]
    score_fit = case2_v4.compute_static_logits(x_fit, data.params)
    score_test = case2_v4.compute_static_logits(x_test, data.params)
    fit_design = torch.cat([torch.ones(len(x_fit), 1), x_fit], dim=1)
    test_design = torch.cat([torch.ones(len(x_test), 1), x_test], dim=1)
    coefficients = torch.linalg.lstsq(fit_design, score_fit).solution
    residual = score_test - test_design @ coefficients
    centered = score_test - score_test.mean(dim=0, keepdim=True)
    return 1.0 - residual.square().sum(dim=0) / centered.square().sum(dim=0)


def _data_diagnostics(data: PreparedData) -> dict[str, object]:
    r_squared = _linear_r_squared(data)
    train_counts = [
        int((data.Delta_train_full == event).sum().item()) for event in range(K + 1)
    ]
    test_counts = [
        int((data.Delta_test == event).sum().item()) for event in range(K + 1)
    ]
    return {
        "train_counts_censor_then_causes": train_counts,
        "test_counts_censor_then_causes": test_counts,
        "train_censor_fraction": train_counts[0] / N_TRAIN,
        "test_censor_fraction": test_counts[0] / N_TEST,
        "event_time_at_limit_fraction": float(
            (data.T_train_true >= case2_v4.T_MAX - 1e-3).float().mean().item()
        ),
        "static_score_linear_r2": [float(value) for value in r_squared],
        "static_score_linear_r2_mean": float(r_squared.mean().item()),
        "static_score_linear_r2_max": float(r_squared.max().item()),
        "evaluation_time_count": len(data.eval_times),
        "evaluation_time_min": float(data.eval_times[0].item()),
        "evaluation_time_max": float(data.eval_times[-1].item()),
    }


def truth_diagnostics(parameters: dict[str, Tensor]) -> dict[str, float]:
    generator = torch.Generator().manual_seed(160_000)
    x = torch.randn(128, P, generator=generator)
    times = torch.linspace(0.0, case2_v4.T_MAX, 301)
    cif, survival = _true_probability_grid(x, times, parameters)
    increments = cif[:, :, 1:] - cif[:, :, :-1]
    conservation = cif.sum(dim=1) + survival - 1.0
    return {
        "minimum_cif": float(cif.min().item()),
        "minimum_survival": float(survival.min().item()),
        "minimum_cif_increment": float(increments.min().item()),
        "maximum_conservation_error": float(conservation.abs().max().item()),
        "maximum_cif_at_zero": float(cif[:, :, 0].abs().max().item()),
        "maximum_survival_error_at_zero": float(
            (survival[:, 0] - 1.0).abs().max().item()
        ),
    }


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


def _method_is_complete(method_dir: Path, execution_hash: str) -> bool:
    complete_path = method_dir / "complete.json"
    if not complete_path.exists():
        return False
    complete = _read_json(complete_path)
    if not (
        complete.get("complete") is True
        and complete.get("execution_hash") == execution_hash
    ):
        return False
    return all(
        (method_dir / filename).is_file() for filename in REQUIRED_METHOD_ARTIFACTS
    )


def _replicate_is_complete(
    replicate_dir: Path,
    replicate: int,
    config: ExperimentConfig,
    execution_hash: str,
) -> bool:
    if not all(
        (replicate_dir / filename).is_file()
        for filename in REQUIRED_REPLICATE_ARTIFACTS
    ):
        return False
    complete_path = replicate_dir / "complete.json"
    metrics_path = replicate_dir / "metrics.json"
    complete = _read_json(complete_path)
    manifest = _read_json(replicate_dir / "replicate_manifest.json")
    if (
        complete.get("complete") is not True
        or complete.get("execution_hash") != execution_hash
        or set(complete.get("completed_methods", [])) != set(ALL_MODELS)
        or set(_read_json(metrics_path)) != set(ALL_MODELS)
        or manifest.get("execution_hash") != execution_hash
        or manifest.get("replicate") != replicate
        or manifest.get("train_seed") != config.train_seed_base + replicate
        or manifest.get("test_seed") != config.test_seed_base + replicate
        or manifest.get("model_seed") != config.model_seed
        or manifest.get("dgp_seed") != config.dgp_seed
    ):
        return False
    return all(
        _method_is_complete(
            replicate_dir / "methods" / METHOD_CACHE_KEYS[method],
            execution_hash,
        )
        for method in ALL_MODELS
    )


def _save_checkpoint(
    method_dir: Path,
    spec: ModelSpec,
    model: object,
    config: ExperimentConfig,
    configuration_hash: str,
    execution_hash: str,
) -> None:
    checkpoint = spec.serialize(model)
    checkpoint.update(
        {
            "model_seed": config.model_seed,
            "configuration_hash": configuration_hash,
            "execution_hash": execution_hash,
        }
    )
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


def _save_predictions(
    method_dir: Path,
    test_cif: Tensor,
    test_survival: Tensor,
    plot_cif: Tensor,
    plot_survival: Tensor,
) -> None:
    _write_torch(
        method_dir / "test_predictions.pt",
        {
            "final_cif": test_cif.detach().cpu().float(),
            "native_survival": test_survival.detach().cpu().float(),
            "implied_survival": (1.0 - test_cif.sum(dim=1)).detach().cpu().float(),
        },
    )
    _write_torch(
        method_dir / "plot_predictions.pt",
        {
            "final_cif": plot_cif.detach().cpu().float(),
            "native_survival": plot_survival.detach().cpu().float(),
            "implied_survival": (1.0 - plot_cif.sum(dim=1)).detach().cpu().float(),
        },
    )


def _run_method(
    replicate_dir: Path,
    spec: ModelSpec,
    data: PreparedData,
    plot_cohort: PlotCohort,
    config: ExperimentConfig,
    configuration_hash: str,
    execution_hash: str,
) -> tuple[dict[str, float | int], dict[str, float], dict[str, float | int]]:
    method_dir = replicate_dir / "methods" / spec.cache_key
    if _method_is_complete(method_dir, execution_hash):
        logger.info("[%s] loading completed partial result", spec.name)
        return (
            _read_json(method_dir / "metrics.json"),
            _read_json(method_dir / "timings.json"),
            _read_json(method_dir / "all_subject_probability_diagnostics.json"),
        )
    logger.info("[%s] training", spec.name)
    torch.manual_seed(config.model_seed)
    train_start = time.perf_counter()
    model = spec.train()
    train_time = time.perf_counter() - train_start
    if spec.predict_survival is None:
        raise RuntimeError(f"{spec.name} does not expose native survival")
    predict_start = time.perf_counter()
    test_raw_cif, test_survival = spec.predict_survival(
        model,
        data.X_test,
        data.eval_times,
    )
    predict_time = time.perf_counter() - predict_start
    postprocess_start = time.perf_counter()
    test_cif = spec.post_process(test_raw_cif)
    postprocess_time = time.perf_counter() - postprocess_start
    if spec.cache_key != "crsoft":
        postprocess_time = 0.0
    metrics = _evaluate_predictions(test_cif, test_survival, data)
    diagnostic_start = time.perf_counter()
    train_raw_cif, train_survival = spec.predict_survival(
        model,
        data.X_train_full,
        data.eval_times,
    )
    train_cif = spec.post_process(train_raw_cif)
    all_cif = torch.cat([train_cif, test_cif], dim=0)
    all_survival = torch.cat([train_survival, test_survival], dim=0)
    diagnostics, negative_mask, below_tolerance_mask = _probability_diagnostics(
        all_cif,
        all_survival,
    )
    diagnostic_time = time.perf_counter() - diagnostic_start
    plot_start = time.perf_counter()
    plot_raw_cif, plot_survival = spec.predict_survival(
        model,
        plot_cohort.X,
        plot_cohort.times,
    )
    plot_cif = spec.post_process(plot_raw_cif)
    plot_time = time.perf_counter() - plot_start
    timings = {
        "train_time_sec": train_time,
        "predict_time_sec": predict_time,
        "pav_time_sec": postprocess_time,
        "total_model_time_sec": train_time + predict_time + postprocess_time,
        "diagnostic_prediction_time_sec": diagnostic_time,
        "plot_prediction_time_sec": plot_time,
    }
    _save_checkpoint(
        method_dir,
        spec,
        model,
        config,
        configuration_hash,
        execution_hash,
    )
    _save_predictions(method_dir, test_cif, test_survival, plot_cif, plot_survival)
    if bool(negative_mask.any().item()):
        _write_torch(
            method_dir / "negative_survival_masks.pt",
            {
                "negative_mask": negative_mask.cpu(),
                "below_tolerance_mask": below_tolerance_mask.cpu(),
            },
        )
    _write_json(method_dir / "metrics.json", metrics)
    _write_json(method_dir / "timings.json", timings)
    _write_json(
        method_dir / "all_subject_probability_diagnostics.json",
        diagnostics,
    )
    _write_json(
        method_dir / "complete.json",
        {"complete": True, "execution_hash": execution_hash},
    )
    logger.info(
        "[%s] MSE=%.6f Ctd=%.6f IBS=%.6f train=%.2fs",
        spec.name,
        metrics["MSE_overall"],
        metrics["Ctd_overall"],
        metrics["IBS_overall"],
        train_time,
    )
    return metrics, timings, diagnostics


def _replicate_manifest(
    replicate: int,
    data: PreparedData,
    config: ExperimentConfig,
    configuration_hash: str,
    execution_hash: str,
) -> dict[str, object]:
    return {
        "replicate": replicate,
        "configuration_hash": configuration_hash,
        "execution_hash": execution_hash,
        "train_seed": config.train_seed_base + replicate,
        "test_seed": config.test_seed_base + replicate,
        "model_seed": config.model_seed,
        "dgp_seed": config.dgp_seed,
        "expected_methods": list(ALL_MODELS),
        "data_diagnostics": _data_diagnostics(data),
    }


def _run_replicate(
    replicate: int,
    output_dir: Path,
    config: ExperimentConfig,
    parameters: dict[str, Tensor],
    plot_cohort: PlotCohort,
    configuration_hash: str,
    execution_hash: str,
    resume: bool,
) -> bool:
    final_dir = output_dir / "replicates" / f"rep_{replicate:03d}"
    if final_dir.exists():
        if resume and _replicate_is_complete(
            final_dir,
            replicate,
            config,
            execution_hash,
        ):
            logger.info("Replicate %03d already complete", replicate)
            return False
        raise FileExistsError(
            f"replicate output is incomplete or incompatible: {final_dir}"
        )
    temp_dir = output_dir / "replicates" / f".rep_{replicate:03d}.in_progress"
    if temp_dir.exists() and not resume:
        raise FileExistsError(f"partial replicate requires --resume: {temp_dir}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    shared_data_path = temp_dir / "shared_data.pt"
    if shared_data_path.exists():
        data = _load_shared_data(temp_dir, execution_hash)
    else:
        data = _prepare_data(replicate, config, parameters)
        _save_shared_data(temp_dir, data, execution_hash)
    logger.info("Replicate %03d", replicate)
    metrics: dict[str, dict[str, float | int]] = {}
    timings: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, dict[str, float | int]] = {}
    for spec in build_specs(data, config.softcomp, config.model_seed):
        method_metrics, method_timings, method_diagnostics = _run_method(
            temp_dir,
            spec,
            data,
            plot_cohort,
            config,
            configuration_hash,
            execution_hash,
        )
        metrics[spec.name] = method_metrics
        timings[spec.name] = method_timings
        diagnostics[spec.name] = method_diagnostics
    _write_json(temp_dir / "metrics.json", metrics)
    _write_json(temp_dir / "timings.json", timings)
    _write_json(temp_dir / "all_subject_probability_diagnostics.json", diagnostics)
    _write_json(
        temp_dir / "replicate_manifest.json",
        _replicate_manifest(
            replicate,
            data,
            config,
            configuration_hash,
            execution_hash,
        ),
    )
    _write_json(
        temp_dir / "complete.json",
        {
            "complete": True,
            "configuration_hash": configuration_hash,
            "execution_hash": execution_hash,
            "completed_methods": list(ALL_MODELS),
        },
    )
    temp_dir.replace(final_dir)
    return True


def _long_rows(
    output_dir: Path,
    config: ExperimentConfig,
    filename: str,
    execution_hash: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for replicate in range(config.n_replicates):
        replicate_dir = output_dir / "replicates" / f"rep_{replicate:03d}"
        if not _replicate_is_complete(
            replicate_dir,
            replicate,
            config,
            execution_hash,
        ):
            continue
        for method, values in _read_json(replicate_dir / filename).items():
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


def negative_survival_summary(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, dict[int, dict[str, float]]] = {}
    for row in rows:
        method = str(row["method"])
        replicate = int(row["replicate"])
        metric = str(row["metric"])
        grouped.setdefault(method, {}).setdefault(replicate, {})[metric] = float(
            row["value"]
        )
    summaries = []
    for method, replicate_values in sorted(grouped.items()):
        values = list(replicate_values.values())
        negative_count = int(sum(v["Implied_S_negative_count"] for v in values))
        total_count = int(sum(v["Implied_S_total_count"] for v in values))
        negative_subjects = int(
            sum(v["Implied_S_negative_subject_count"] for v in values)
        )
        total_subjects = int(sum(v["Implied_S_total_subject_count"] for v in values))
        tolerance_count = int(sum(v["Implied_S_below_tolerance_count"] for v in values))
        tolerance_subjects = int(
            sum(v["Implied_S_below_tolerance_subject_count"] for v in values)
        )
        replicates_with_negative = sum(
            v["Implied_S_negative_count"] > 0 for v in values
        )
        summaries.append(
            {
                "method": method,
                "n": len(values),
                "negative_point_count": negative_count,
                "total_point_count": total_count,
                "negative_point_fraction": (
                    negative_count / total_count if total_count else 0.0
                ),
                "negative_subject_count": negative_subjects,
                "total_subject_count": total_subjects,
                "negative_subject_fraction": (
                    negative_subjects / total_subjects if total_subjects else 0.0
                ),
                "tolerance_negative_point_count": tolerance_count,
                "tolerance_negative_subject_count": tolerance_subjects,
                "replicates_with_negative_survival": replicates_with_negative,
                "global_minimum_implied_survival": min(
                    v["Implied_S_min"] for v in values
                ),
            }
        )
    return summaries


def _aggregate_plot_predictions(
    output_dir: Path,
    config: ExperimentConfig,
    execution_hash: str,
) -> None:
    aggregated: dict[str, object] = {}
    for method, cache_key in METHOD_CACHE_KEYS.items():
        predictions = []
        replicates = []
        for replicate in range(config.n_replicates):
            replicate_dir = output_dir / "replicates" / f"rep_{replicate:03d}"
            if not _replicate_is_complete(
                replicate_dir,
                replicate,
                config,
                execution_hash,
            ):
                continue
            payload = torch.load(
                replicate_dir / "methods" / cache_key / "plot_predictions.pt",
                weights_only=False,
            )
            predictions.append(payload["final_cif"].float())
            replicates.append(replicate)
        if predictions:
            stacked = torch.stack(predictions)
            aggregated[method] = {
                "replicates": replicates,
                "final_cif": stacked,
                "mean": stacked.mean(dim=0),
                "std": stacked.std(dim=0, unbiased=len(predictions) > 1),
            }
    _write_torch(output_dir / "aggregate" / "plot_predictions_all.pt", aggregated)


def _aggregate(
    output_dir: Path,
    config: ExperimentConfig,
    configuration_hash: str,
    execution_hash: str,
) -> None:
    aggregate_dir = output_dir / "aggregate"
    metrics = _long_rows(output_dir, config, "metrics.json", execution_hash)
    timings = _long_rows(output_dir, config, "timings.json", execution_hash)
    diagnostics = _long_rows(
        output_dir,
        config,
        "all_subject_probability_diagnostics.json",
        execution_hash,
    )
    long_fields = ["replicate", "method", "metric", "value"]
    summary_fields = ["method", "metric", "n", "mean", "std"]
    _write_csv(aggregate_dir / "metrics_long.csv", long_fields, metrics)
    _write_csv(
        aggregate_dir / "metrics_summary.csv",
        summary_fields,
        _summary_rows(metrics),
    )
    _write_csv(aggregate_dir / "timings_long.csv", long_fields, timings)
    _write_csv(
        aggregate_dir / "timing_summary.csv",
        summary_fields,
        _summary_rows(timings),
    )
    _write_csv(
        aggregate_dir / "negative_survival_long.csv",
        long_fields,
        diagnostics,
    )
    negative_summary = negative_survival_summary(diagnostics)
    if negative_summary:
        _write_csv(
            aggregate_dir / "negative_survival_summary.csv",
            list(negative_summary[0]),
            negative_summary,
        )
    _aggregate_plot_predictions(output_dir, config, execution_hash)
    completed = [
        replicate
        for replicate in range(config.n_replicates)
        if _replicate_is_complete(
            output_dir / "replicates" / f"rep_{replicate:03d}",
            replicate,
            config,
            execution_hash,
        )
    ]
    _write_json(
        output_dir / "manifest.json",
        {
            "configuration_hash": configuration_hash,
            "execution_hash": execution_hash,
            "completed_replicates": completed,
            "missing_replicates": [
                replicate
                for replicate in range(config.n_replicates)
                if replicate not in completed
            ],
            "methods": {
                method: {
                    "expected_count": config.n_replicates,
                    "completed_count": len(completed),
                    "completed_replicates": completed,
                }
                for method in ALL_MODELS
            },
            "environment": _environment_payload(),
        },
    )


def _development_payloads(
    output_dir: Path,
    config: ExperimentConfig,
    execution_hash: str,
) -> tuple[dict[int, dict[str, dict[str, float | int]]], dict[int, dict]]:
    metrics = {}
    manifests = {}
    for replicate in range(DEVELOPMENT_REPLICATES):
        replicate_dir = output_dir / "replicates" / f"rep_{replicate:03d}"
        if not _replicate_is_complete(
            replicate_dir,
            replicate,
            config,
            execution_hash,
        ):
            continue
        metrics[replicate] = _read_json(replicate_dir / "metrics.json")
        manifests[replicate] = _read_json(replicate_dir / "replicate_manifest.json")
    return metrics, manifests


def evaluate_development_gate(
    metrics: dict[int, dict[str, dict[str, float | int]]],
    manifests: dict[int, dict],
    configuration_hash: str,
    truth: dict[str, float],
) -> dict[str, object]:
    """Evaluate the frozen Case II v4 development criteria."""
    complete = (
        len(metrics) == DEVELOPMENT_REPLICATES
        and len(manifests) == DEVELOPMENT_REPLICATES
        and all(set(values) == set(ALL_MODELS) for values in metrics.values())
    )
    if not complete:
        return {"passed": False, "complete": False}
    core_metrics = ("MSE_overall", "Ctd_overall", "IBS_overall")
    means = {
        metric: {
            method: statistics.mean(
                float(metrics[replicate][method][metric])
                for replicate in range(DEVELOPMENT_REPLICATES)
            )
            for method in ALL_MODELS
        }
        for metric in core_metrics
    }
    ranks = {}
    for metric, values in means.items():
        reverse = metric == "Ctd_overall"
        ordered = sorted(values, key=values.get, reverse=reverse)
        ranks[metric] = {method: ordered.index(method) + 1 for method in ordered}
    cscox = [
        float(metrics[replicate]["cs-Cox"]["Ctd_overall"])
        for replicate in range(DEVELOPMENT_REPLICATES)
    ]
    data_checks = []
    for replicate in range(DEVELOPMENT_REPLICATES):
        diagnostics = manifests[replicate]["data_diagnostics"]
        train_counts = diagnostics["train_counts_censor_then_causes"]
        test_counts = diagnostics["test_counts_censor_then_causes"]
        data_checks.append(
            diagnostics["static_score_linear_r2_mean"] < 0.05
            and diagnostics["static_score_linear_r2_max"] < 0.10
            and diagnostics["event_time_at_limit_fraction"] == 0.0
            and 0.45 <= diagnostics["train_censor_fraction"] <= 0.55
            and 0.45 <= diagnostics["test_censor_fraction"] <= 0.55
            and min(train_counts[1:]) >= 500
            and min(test_counts[1:]) >= 80
        )
    truth_passed = (
        truth["minimum_cif"] >= 0.0
        and truth["minimum_survival"] >= 0.0
        and truth["minimum_cif_increment"] >= -1e-6
        and truth["maximum_conservation_error"] <= 1e-6
        and truth["maximum_cif_at_zero"] <= 1e-7
        and truth["maximum_survival_error_at_zero"] <= 1e-7
    )
    cscox_passed = statistics.mean(cscox) <= 0.55 and max(cscox) <= 0.60
    softcomp_passed = all(ranks[metric]["SoftComp"] <= 2 for metric in core_metrics)
    return {
        "passed": truth_passed
        and all(data_checks)
        and cscox_passed
        and softcomp_passed,
        "complete": True,
        "configuration_hash": configuration_hash,
        "truth": {"passed": truth_passed, "diagnostics": truth},
        "data": {"passed": all(data_checks), "replicate_checks": data_checks},
        "cscox": {
            "passed": cscox_passed,
            "values": cscox,
            "mean": statistics.mean(cscox),
            "maximum": max(cscox),
        },
        "softcomp": {
            "passed": softcomp_passed,
            "ranks": {metric: ranks[metric]["SoftComp"] for metric in core_metrics},
        },
        "means": means,
    }


def development_gate_report(
    output_dir: Path,
    config: ExperimentConfig,
    configuration_hash: str,
    execution_hash: str,
    truth: dict[str, float],
) -> dict[str, object]:
    metrics, manifests = _development_payloads(output_dir, config, execution_hash)
    return evaluate_development_gate(metrics, manifests, configuration_hash, truth)


def validate_formal_gate(path: Path, configuration_hash: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"formal phase requires a development gate: {path}")
    gate = _read_json(path)
    if gate.get("configuration_hash") != configuration_hash:
        raise RuntimeError("development gate configuration hash mismatch")
    if gate.get("complete") is not True or gate.get("passed") is not True:
        raise RuntimeError("development gate did not pass")


def _write_config(
    output_dir: Path,
    config: ExperimentConfig,
    configuration_hash: str,
    execution_hash: str,
    truth: dict[str, float],
) -> None:
    path = output_dir / "config.json"
    execution = {
        "phase": config.phase,
        "n_replicates": config.n_replicates,
        "train_seed_base": config.train_seed_base,
        "test_seed_base": config.test_seed_base,
        "model_seed": config.model_seed,
        "dgp_seed": config.dgp_seed,
        "cpu_threads": config.cpu_threads,
        "models": config.models,
        "softcomp": asdict(config.softcomp),
    }
    payload = {
        "execution": execution,
        "last_invocation": {
            "start_replicate": config.start_replicate,
            "end_replicate": config.end_replicate,
        },
        "frozen_protocol": _frozen_protocol_payload(config),
        "configuration_hash": configuration_hash,
        "execution_hash": execution_hash,
        "truth_diagnostics": truth,
    }
    if path.exists():
        existing = _read_json(path)
        if (
            existing.get("configuration_hash") != configuration_hash
            or existing.get("execution_hash") != execution_hash
        ):
            raise RuntimeError("output directory contains a different execution")
    _write_json(path, payload)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    config = build_config(args)
    torch.set_num_threads(config.cpu_threads)
    output_dir = args.output_dir or _default_output_dir(config.phase)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = case2_v4.generate_parameters(K=K, p=P, seed=config.dgp_seed)
    configuration_hash = compute_configuration_hash(config)
    execution_hash = compute_execution_hash(config)
    truth = truth_diagnostics(parameters)
    if config.phase == "formal":
        gate_file = args.gate_file or (
            get_output_dir("case2_v4") / "development_gate" / "gate.json"
        )
        validate_formal_gate(gate_file, configuration_hash)
    _write_config(
        output_dir,
        config,
        configuration_hash,
        execution_hash,
        truth,
    )
    plot_cohort = _load_or_create_plot_cohort(
        output_dir,
        parameters,
        execution_hash,
    )
    for replicate in range(config.start_replicate, config.end_replicate):
        _run_replicate(
            replicate,
            output_dir,
            config,
            parameters,
            plot_cohort,
            configuration_hash,
            execution_hash,
            args.resume,
        )
        _aggregate(
            output_dir,
            config,
            configuration_hash,
            execution_hash,
        )
    if config.phase == "development":
        gate = development_gate_report(
            output_dir,
            config,
            configuration_hash,
            execution_hash,
            truth,
        )
        _write_json(output_dir / "gate.json", gate)
        logger.info("Development gate passed=%s", gate["passed"])
    logger.info("Artifacts: %s", output_dir)


if __name__ == "__main__":
    main()
