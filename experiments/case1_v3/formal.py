#!/usr/bin/env python3
"""Development gate and 50-replicate formal runner for Case I v3."""

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

from ...data import case1_v3
from ...evaluation import compute_mse_accuracy, evaluate_cif_metrics, get_output_dir
from ...evaluation.simulation import compute_dist
from ..runner import ModelSpec
from .run import (
    build_specs,
    CRITICAL_EVALUATION_TIMES,
    data_diagnostics as run_data_diagnostics,
    EVALUATION_PERCENTILE_CAP,
    K,
    P,
    prepare_data,
    PreparedData,
    SoftCompConfig,
    training_history,
    truth as truth_grid,
)

logger: logging.Logger = logging.getLogger(__name__)

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
DEVELOPMENT_TRAIN_SEED_BASE = 511_000
DEVELOPMENT_TEST_SEED_BASE = 521_000
FORMAL_TRAIN_SEED_BASE = 530_000
FORMAL_TEST_SEED_BASE = 540_000
PLOT_COHORT_SEED = 550_000
TRUTH_DIAGNOSTIC_SEED = 560_000
PLOT_SUBJECTS = 5
PLOT_TIME_POINTS = 100
PLOT_TIME_MAX = 24.0
NEGATIVE_SURVIVAL_ATOL = 1e-6
BETA = (
    (-0.7, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (0.7, 0.0, 0.0, 0.0, 0.0),
)
INTERCEPTS = (-16.0, -15.78, -16.0)
TIME_SLOPE = 1.15
MAX_STATIC_AFFINE_ERROR = 1e-7
MAX_AFFINE_MU_RECONSTRUCTION_ERROR = 2e-5
MIN_AFFINE_MU_R2 = 1.0 - 1e-6
MAX_INITIAL_EVENT_PROBABILITY_MEAN = 2e-6
MAX_INITIAL_EVENT_PROBABILITY_P99 = 5e-6
MAX_INITIAL_EVENT_PROBABILITY_MAX = 1e-4
MAX_ATOM_COUNT_DISCREPANCY_FLOOR = 3.0
ATOM_COUNT_STANDARD_DEVIATION_MULTIPLIER = 5.0
MIN_TRAIN_EVENTS_PER_CAUSE = 400
MIN_TEST_EVENTS_PER_CAUSE = 60
MIN_FINE_GRAY_ACO_POINTS_PER_REPLICATE = 500
MIN_FINE_GRAY_ACO_SUBJECTS_PER_REPLICATE = 150
MAX_CSCOX_MEAN_CTD = 0.63
MAX_CSCOX_REPLICATE_CTD = 0.65
MIN_CSCOX_MEAN_RANK = 2
MAX_FINE_GRAY_MINIMUM_IMPLIED_SURVIVAL = -0.05


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


@dataclass(frozen=True)
class AffineFitDiagnostics:
    r_squared_by_cause: list[float]
    r_squared_min: float
    r_squared_min_nonconstant: float
    constant_cause_count: int
    nonconstant_cause_count: int
    constant_cause_max_abs_error: float
    static_max_abs_error: float

    def as_payload(self) -> dict[str, object]:
        return {
            "affine_mu_r2_by_cause": self.r_squared_by_cause,
            "affine_mu_r2_min": self.r_squared_min,
            "affine_mu_r2_min_nonconstant": self.r_squared_min_nonconstant,
            "affine_mu_constant_cause_count": self.constant_cause_count,
            "affine_mu_nonconstant_cause_count": self.nonconstant_cause_count,
            "affine_mu_constant_cause_max_abs_error": (
                self.constant_cause_max_abs_error
            ),
            "static_affine_max_abs_error": self.static_max_abs_error,
        }


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
    parser.add_argument("--softcomp-optimizer", default="lbfgs")
    parser.add_argument(
        "--softcomp-standardize-x",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--softcomp-initial-alpha", type=float, default=1.0)
    parser.add_argument("--softcomp-max-iter", type=int, default=100)
    parser.add_argument("--softcomp-lr", type=float, default=1.0)
    parser.add_argument("--softcomp-l2-penalty", type=float, default=0.0)
    parser.add_argument("--softcomp-tolerance-grad", type=float, default=1e-7)
    parser.add_argument("--softcomp-tolerance-change", type=float, default=1e-9)
    parser.add_argument("--softcomp-history-size", type=int, default=50)
    parser.add_argument("--softcomp-atom-tolerance", type=float, default=1e-8)
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
    return (1, 205_000, 215_000, "smoke")


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
            optimizer=args.softcomp_optimizer,
            standardize_x=args.softcomp_standardize_x,
            initial_alpha=args.softcomp_initial_alpha,
            max_iter=args.softcomp_max_iter,
            lr=args.softcomp_lr,
            l2_penalty=args.softcomp_l2_penalty,
            tolerance_grad=args.softcomp_tolerance_grad,
            tolerance_change=args.softcomp_tolerance_change,
            history_size=args.softcomp_history_size,
            atom_tolerance=args.softcomp_atom_tolerance,
        ),
    )


def generate_frozen_parameters(dgp_seed: int = DGP_SEED) -> dict[str, Tensor]:
    """Construct the exact pre-registered Case I v3 truth parameters."""
    return case1_v3.generate_parameters(
        K=K,
        p=P,
        seed=dgp_seed,
        beta_values=tuple(value for row in BETA for value in row),
        intercepts=INTERCEPTS,
        alpha=TIME_SLOPE,
    )


def _frozen_protocol_payload(config: ExperimentConfig) -> dict[str, object]:
    return {
        "case": "Case I v3",
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
        "development_gate_amendment": {
            "type": "one-time replacement after the initial gate failed",
            "superseded_schedule": {
                "replicates": DEVELOPMENT_REPLICATES,
                "train_seed_base": 510_000,
                "test_seed_base": 520_000,
            },
            "superseded_cscox_gate": {
                "maximum_mean_ctd": 0.625,
                "maximum_replicate_ctd": 0.65,
                "minimum_mean_ctd_rank": 3,
            },
            "replacement_cscox_gate": {
                "maximum_mean_ctd": MAX_CSCOX_MEAN_CTD,
                "maximum_replicate_ctd": MAX_CSCOX_REPLICATE_CTD,
                "minimum_mean_ctd_rank": MIN_CSCOX_MEAN_RANK,
            },
            "formal_schedule_changed": False,
            "failure_reason_record": "Case1_v3_experiment_plan.md",
        },
        "formal_schedule": {
            "replicates": FORMAL_REPLICATES,
            "train_seed_base": FORMAL_TRAIN_SEED_BASE,
            "test_seed_base": FORMAL_TEST_SEED_BASE,
        },
        "formal_replicates_by_method": dict.fromkeys(ALL_MODELS, FORMAL_REPLICATES),
        "dgp": {
            "beta": [list(row) for row in BETA],
            "intercepts": list(INTERCEPTS),
            "time_slope": TIME_SLOPE,
            "linear_predictor": ("mu_k(x,t)=intercept_k+beta_k^T x+alpha*t, t>=0"),
            "probabilities": (
                "F_k=exp(mu_k)/(1+sum_j exp(mu_j)); S=1/(1+sum_j exp(mu_j))"
            ),
            "time_zero_atom": "sum_k F_k(0|x), sampled by generalized inverse",
            "t_max": case1_v3.T_MAX,
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
        "evaluation_grid": (
            f"100 requested test event-time 0--{EVALUATION_PERCENTILE_CAP}% "
            "quantiles, unioned with four fixed critical times"
        ),
        "negative_survival": "unclamped 1 - sum(final CIF)",
        "negative_survival_atol": NEGATIVE_SURVIVAL_ATOL,
        "development_gate": {
            "censor_fraction_range": [0.45, 0.55],
            "minimum_train_events_per_cause": MIN_TRAIN_EVENTS_PER_CAUSE,
            "minimum_test_events_per_cause": MIN_TEST_EVENTS_PER_CAUSE,
            "maximum_affine_mu_reconstruction_error": (
                MAX_AFFINE_MU_RECONSTRUCTION_ERROR
            ),
            "minimum_nonconstant_affine_mu_r2": MIN_AFFINE_MU_R2,
            "constant_affine_mu_r2_definition": (
                "1.0 when the frozen constant beta row has exact residual at most "
                f"{MAX_STATIC_AFFINE_ERROR}"
            ),
            "maximum_static_affine_error": MAX_STATIC_AFFINE_ERROR,
            "maximum_initial_event_probability_mean": (
                MAX_INITIAL_EVENT_PROBABILITY_MEAN
            ),
            "maximum_initial_event_probability_p99": (
                MAX_INITIAL_EVENT_PROBABILITY_P99
            ),
            "maximum_initial_event_probability_max": (
                MAX_INITIAL_EVENT_PROBABILITY_MAX
            ),
            "atom_count_standard_deviation_multiplier": (
                ATOM_COUNT_STANDARD_DEVIATION_MULTIPLIER
            ),
            "atom_count_discrepancy_floor": MAX_ATOM_COUNT_DISCREPANCY_FLOOR,
            "evaluation_grid_must_cover": list(CRITICAL_EVALUATION_TIMES),
            "maximum_cscox_mean_ctd": MAX_CSCOX_MEAN_CTD,
            "maximum_cscox_replicate_ctd": MAX_CSCOX_REPLICATE_CTD,
            "minimum_cscox_mean_ctd_rank": MIN_CSCOX_MEAN_RANK,
            "maximum_softcomp_rank": 2,
            "fine_gray_all_cause_fits_must_converge": True,
            "minimum_fine_gray_aco_points_per_replicate": (
                MIN_FINE_GRAY_ACO_POINTS_PER_REPLICATE
            ),
            "minimum_fine_gray_aco_subjects_per_replicate": (
                MIN_FINE_GRAY_ACO_SUBJECTS_PER_REPLICATE
            ),
            "maximum_fine_gray_minimum_implied_survival": (
                MAX_FINE_GRAY_MINIMUM_IMPLIED_SURVIVAL
            ),
            "truth_diagnostic_seed": TRUTH_DIAGNOSTIC_SEED,
        },
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
    return get_output_dir("case1_v3") / name


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
            values, survival_values = case1_v3.compute_cif(
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
    true_cif_fn = functools.partial(case1_v3.compute_cif, params=parameters)
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
        case_name="case1_v3",
        t_max=case1_v3.T_MAX,
    )


def _affine_fit_diagnostics(
    x: Tensor,
    static_logits: Tensor,
    parameters: dict[str, Tensor],
) -> AffineFitDiagnostics:
    expected = parameters["intercept"].unsqueeze(0) + x @ parameters["beta"].T
    exact_errors = (static_logits - expected).abs().amax(dim=0)
    constant_causes = parameters["beta"].abs().amax(dim=1) <= 1e-12
    midpoint = len(x) // 2
    fit_design = torch.cat([x.new_ones((midpoint, 1)), x[:midpoint]], dim=1).double()
    test_design = torch.cat(
        [x.new_ones((len(x) - midpoint, 1)), x[midpoint:]], dim=1
    ).double()
    fit_target = static_logits[:midpoint].double()
    test_target = static_logits[midpoint:].double()
    coefficients = torch.linalg.lstsq(fit_design, fit_target).solution
    residual_sum_squares = (
        (test_target - test_design @ coefficients).square().sum(dim=0)
    )
    centered = test_target - test_target.mean(dim=0, keepdim=True)
    total_sum_squares = centered.square().sum(dim=0)
    fitted_r_squared = 1.0 - residual_sum_squares / total_sum_squares.clamp(min=1e-12)
    exact_constant_fit = exact_errors <= MAX_STATIC_AFFINE_ERROR
    r_squared = torch.where(
        constant_causes,
        exact_constant_fit.to(fitted_r_squared.dtype),
        fitted_r_squared,
    )
    nonconstant_r_squared = r_squared[~constant_causes]
    constant_errors = exact_errors[constant_causes]
    return AffineFitDiagnostics(
        r_squared_by_cause=[float(value) for value in r_squared],
        r_squared_min=float(r_squared.min().item()),
        r_squared_min_nonconstant=float(nonconstant_r_squared.min().item()),
        constant_cause_count=int(constant_causes.sum().item()),
        nonconstant_cause_count=int((~constant_causes).sum().item()),
        constant_cause_max_abs_error=float(constant_errors.max().item()),
        static_max_abs_error=float(exact_errors.max().item()),
    )


def _data_diagnostics(data: PreparedData) -> dict[str, object]:
    train_counts = [
        int((data.Delta_train_full == event).sum().item()) for event in range(K + 1)
    ]
    test_counts = [
        int((data.Delta_test == event).sum().item()) for event in range(K + 1)
    ]
    train_event_counts = train_counts[1:]
    test_event_counts = test_counts[1:]
    cause_probabilities = case1_v3.compute_cause_probabilities(
        data.X_train_full, data.params
    )
    initial_probability = case1_v3.compute_initial_event_probability(
        data.X_train_full,
        data.params,
    )
    observed_atom_count = int((data.T_train_true == 0.0).sum().item())
    observed_atom_fraction = observed_atom_count / len(data.T_train_true)
    expected_atom_fraction = float(initial_probability.mean().item())
    expected_atom_count = float(initial_probability.sum().item())
    atom_count_standard_deviation = float(
        (initial_probability * (1.0 - initial_probability)).sum().sqrt().item()
    )
    allowed_atom_count_discrepancy = max(
        MAX_ATOM_COUNT_DISCREPANCY_FLOOR,
        ATOM_COUNT_STANDARD_DEVIATION_MULTIPLIER * atom_count_standard_deviation,
    )
    static_logits = case1_v3.compute_static_logits(data.X_train_full, data.params)
    affine_diagnostics = _affine_fit_diagnostics(
        data.X_train_full,
        static_logits,
        data.params,
    )
    critical_time_errors = [
        float((data.eval_times - critical_time).abs().min().item())
        for critical_time in CRITICAL_EVALUATION_TIMES
    ]
    diagnostics = run_data_diagnostics(data)
    diagnostics.update(
        {
            "train_censor_fraction": train_counts[0] / len(data.Delta_train_full),
            "test_censor_fraction": test_counts[0] / len(data.Delta_test),
            "train_observed_cause_min_max_ratio": min(train_event_counts)
            / max(train_event_counts),
            "test_observed_cause_min_max_ratio": min(test_event_counts)
            / max(test_event_counts),
            "cause_probability_marginal_means": [
                float(value) for value in cause_probabilities.mean(dim=0)
            ],
            "cause_probability_global_min": float(cause_probabilities.min().item()),
            "cause_probability_global_max": float(cause_probabilities.max().item()),
            "event_time_at_limit_fraction": float(
                (data.T_train_true >= case1_v3.T_MAX - 1e-3).float().mean().item()
            ),
            "initial_event_probability_mean": expected_atom_fraction,
            "initial_event_probability_p99": float(
                torch.quantile(initial_probability, 0.99).item()
            ),
            "initial_event_probability_max": float(initial_probability.max().item()),
            "time_zero_atom_count": observed_atom_count,
            "time_zero_atom_fraction": observed_atom_fraction,
            "expected_time_zero_atom_count": expected_atom_count,
            "time_zero_atom_count_discrepancy": abs(
                observed_atom_count - expected_atom_count
            ),
            "time_zero_atom_count_allowed_discrepancy": (
                allowed_atom_count_discrepancy
            ),
            **affine_diagnostics.as_payload(),
            "affine_mu_reconstruction_max_abs_error": (
                _affine_mu_reconstruction_error(
                    data.X_train_full[:128],
                    data.params,
                    torch.tensor(CRITICAL_EVALUATION_TIMES),
                )
            ),
            "evaluation_time_count": len(data.eval_times),
            "evaluation_time_min": float(data.eval_times[0].item()),
            "evaluation_time_max": float(data.eval_times[-1].item()),
            "evaluation_grid_critical_time_abs_errors": critical_time_errors,
            "evaluation_grid_critical_time_max_abs_error": max(critical_time_errors),
        }
    )
    return diagnostics


def _affine_mu_reconstruction_error(
    x: Tensor,
    parameters: dict[str, Tensor],
    times: Tensor,
) -> float:
    x64 = x.double()
    parameters64 = {name: value.double() for name, value in parameters.items()}
    times64 = times.double()
    cif, survival = truth_grid(x64, times64, parameters64)
    reconstructed = torch.log(cif) - torch.log(survival).unsqueeze(1)
    expected = case1_v3.compute_static_logits(x64, parameters64).unsqueeze(
        2
    ) + parameters64["alpha"] * times64.reshape(1, 1, -1)
    nonsaturated = (survival.unsqueeze(1) > 1e-6) & (cif > 0.0)
    if not bool(nonsaturated.any().item()):
        raise RuntimeError("affine reconstruction has no nonsaturated truth points")
    return float((reconstructed - expected)[nonsaturated].abs().max().item())


def truth_diagnostics(parameters: dict[str, Tensor]) -> dict[str, float]:
    generator = torch.Generator().manual_seed(TRUTH_DIAGNOSTIC_SEED)
    x = torch.randn(4096, P, generator=generator)
    times = torch.linspace(0.0, case1_v3.T_MAX, 301)
    cif, survival = _true_probability_grid(x, times, parameters)
    increments = cif[:, :, 1:] - cif[:, :, :-1]
    conservation = cif.sum(dim=1) + survival - 1.0
    initial_probability = case1_v3.compute_initial_event_probability(x, parameters)
    static_logits = case1_v3.compute_static_logits(x, parameters)
    affine_diagnostics = _affine_fit_diagnostics(
        x,
        static_logits,
        parameters,
    )
    diagnostics = {
        "minimum_cif": float(cif.min().item()),
        "minimum_survival": float(survival.min().item()),
        "minimum_cif_increment": float(increments.min().item()),
        "maximum_conservation_error": float(conservation.abs().max().item()),
        "maximum_time_zero_mass_error": float(
            (cif[:, :, 0].sum(dim=1) - initial_probability).abs().max().item()
        ),
        "initial_event_probability_mean": float(initial_probability.mean().item()),
        "initial_event_probability_p99": float(
            torch.quantile(initial_probability, 0.99).item()
        ),
        "initial_event_probability_max": float(initial_probability.max().item()),
        "static_affine_max_abs_error": affine_diagnostics.static_max_abs_error,
        "affine_mu_r2_min": affine_diagnostics.r_squared_min,
        "affine_mu_r2_min_nonconstant": (affine_diagnostics.r_squared_min_nonconstant),
        "affine_mu_constant_cause_count": float(
            affine_diagnostics.constant_cause_count
        ),
        "affine_mu_nonconstant_cause_count": float(
            affine_diagnostics.nonconstant_cause_count
        ),
        "affine_mu_constant_cause_max_abs_error": (
            affine_diagnostics.constant_cause_max_abs_error
        ),
        "affine_mu_reconstruction_max_abs_error": (
            _affine_mu_reconstruction_error(
                x[:256],
                parameters,
                torch.tensor((0.0, *CRITICAL_EVALUATION_TIMES)),
            )
        ),
    }
    return diagnostics


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
    fine_gray_convergence = manifest.get("fine_gray_convergence")
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
        or not isinstance(fine_gray_convergence, dict)
        or fine_gray_convergence.get("cause_fit_count") != K
        or fine_gray_convergence.get("all_converged") is not True
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
    history = training_history(model)
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


def _fine_gray_convergence_payload(replicate_dir: Path) -> dict[str, object]:
    checkpoint = torch.load(
        replicate_dir / "methods" / "fine_gray" / "checkpoint.pt",
        weights_only=False,
    )
    cause_models = checkpoint.get("cause_models")
    if not isinstance(cause_models, list):
        raise TypeError("Fine-Gray checkpoint must contain a cause_models list")
    if any(not isinstance(model, dict) for model in cause_models):
        raise TypeError("Fine-Gray cause models must be dictionaries")
    converged = [bool(model.get("converged")) for model in cause_models]
    iterations = [int(model.get("iterations", 0)) for model in cause_models]
    score_norms = [float(model["score_norm"]) for model in cause_models]
    return {
        "cause_fit_count": len(cause_models),
        "converged_count": sum(converged),
        "all_converged": len(cause_models) == K and all(converged),
        "iterations": iterations,
        "maximum_score_norm": max(score_norms, default=0.0),
    }


def _replicate_manifest(
    replicate_dir: Path,
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
        "fine_gray_convergence": _fine_gray_convergence_payload(replicate_dir),
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
            temp_dir,
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
    """Evaluate the frozen Case I v3 development criteria."""
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
        ordered = sorted(
            values,
            key=values.get,
            reverse=metric == "Ctd_overall",
        )
        ranks[metric] = {method: rank for rank, method in enumerate(ordered, start=1)}

    cscox_values = [
        float(metrics[replicate]["cs-Cox"]["Ctd_overall"])
        for replicate in range(DEVELOPMENT_REPLICATES)
    ]
    fine_gray_aco = [
        {
            "point_count": int(
                metrics[replicate]["Fine-Gray"]["Implied_S_below_tolerance_count"]
            ),
            "subject_count": int(
                metrics[replicate]["Fine-Gray"][
                    "Implied_S_below_tolerance_subject_count"
                ]
            ),
            "minimum_implied_survival": float(
                metrics[replicate]["Fine-Gray"]["Implied_S_min"]
            ),
        }
        for replicate in range(DEVELOPMENT_REPLICATES)
    ]

    data_checks = []
    affine_checks = []
    atom_checks = []
    fine_gray_convergence_checks = []
    for replicate in range(DEVELOPMENT_REPLICATES):
        diagnostics = manifests[replicate]["data_diagnostics"]
        convergence = manifests[replicate]["fine_gray_convergence"]
        train_counts = diagnostics["train_counts_censor_then_causes"]
        test_counts = diagnostics["test_counts_censor_then_causes"]
        data_checks.append(
            diagnostics["event_time_at_limit_fraction"] == 0.0
            and 0.45 <= diagnostics["train_censor_fraction"] <= 0.55
            and 0.45 <= diagnostics["test_censor_fraction"] <= 0.55
            and min(train_counts[1:]) >= MIN_TRAIN_EVENTS_PER_CAUSE
            and min(test_counts[1:]) >= MIN_TEST_EVENTS_PER_CAUSE
            and diagnostics["evaluation_grid_critical_time_max_abs_error"] <= 1e-6
            and diagnostics["evaluation_time_max"] >= max(CRITICAL_EVALUATION_TIMES)
        )
        affine_checks.append(
            diagnostics["affine_mu_r2_min_nonconstant"] >= MIN_AFFINE_MU_R2
            and diagnostics["affine_mu_constant_cause_count"] == 1
            and diagnostics["affine_mu_nonconstant_cause_count"] == K - 1
            and diagnostics["affine_mu_constant_cause_max_abs_error"]
            <= MAX_STATIC_AFFINE_ERROR
            and diagnostics["static_affine_max_abs_error"] <= MAX_STATIC_AFFINE_ERROR
            and diagnostics["affine_mu_reconstruction_max_abs_error"]
            <= MAX_AFFINE_MU_RECONSTRUCTION_ERROR
        )
        atom_checks.append(
            0.0
            < diagnostics["initial_event_probability_mean"]
            <= MAX_INITIAL_EVENT_PROBABILITY_MEAN
            and diagnostics["initial_event_probability_p99"]
            <= MAX_INITIAL_EVENT_PROBABILITY_P99
            and diagnostics["initial_event_probability_max"]
            <= MAX_INITIAL_EVENT_PROBABILITY_MAX
            and diagnostics["time_zero_atom_count_discrepancy"]
            <= diagnostics["time_zero_atom_count_allowed_discrepancy"]
        )
        fine_gray_convergence_checks.append(
            convergence["cause_fit_count"] == K
            and convergence["converged_count"] == K
            and convergence["all_converged"] is True
        )

    truth_passed = (
        truth["minimum_cif"] >= 0.0
        and truth["minimum_survival"] >= 0.0
        and truth["minimum_cif_increment"] >= -1e-6
        and truth["maximum_conservation_error"] <= 1e-6
        and truth["maximum_time_zero_mass_error"] <= 1e-6
        and truth["affine_mu_r2_min_nonconstant"] >= MIN_AFFINE_MU_R2
        and truth["affine_mu_constant_cause_count"] == 1
        and truth["affine_mu_nonconstant_cause_count"] == K - 1
        and truth["affine_mu_constant_cause_max_abs_error"] <= MAX_STATIC_AFFINE_ERROR
        and truth["static_affine_max_abs_error"] <= MAX_STATIC_AFFINE_ERROR
        and truth["affine_mu_reconstruction_max_abs_error"]
        <= MAX_AFFINE_MU_RECONSTRUCTION_ERROR
        and 0.0
        < truth["initial_event_probability_mean"]
        <= MAX_INITIAL_EVENT_PROBABILITY_MEAN
        and truth["initial_event_probability_p99"] <= MAX_INITIAL_EVENT_PROBABILITY_P99
        and truth["initial_event_probability_max"] <= MAX_INITIAL_EVENT_PROBABILITY_MAX
    )
    cscox_rank = ranks["Ctd_overall"]["cs-Cox"]
    cscox_passed = (
        statistics.mean(cscox_values) <= MAX_CSCOX_MEAN_CTD
        and max(cscox_values) <= MAX_CSCOX_REPLICATE_CTD
        and cscox_rank >= MIN_CSCOX_MEAN_RANK
    )
    softcomp_passed = all(ranks[metric]["SoftComp"] <= 2 for metric in core_metrics)
    fine_gray_convergence_passed = all(fine_gray_convergence_checks)
    fine_gray_aco_passed = all(
        values["point_count"] >= MIN_FINE_GRAY_ACO_POINTS_PER_REPLICATE
        and values["subject_count"] >= MIN_FINE_GRAY_ACO_SUBJECTS_PER_REPLICATE
        and values["minimum_implied_survival"] <= MAX_FINE_GRAY_MINIMUM_IMPLIED_SURVIVAL
        for values in fine_gray_aco
    )
    passed = (
        truth_passed
        and all(data_checks)
        and all(affine_checks)
        and all(atom_checks)
        and cscox_passed
        and softcomp_passed
        and fine_gray_convergence_passed
        and fine_gray_aco_passed
    )
    return {
        "passed": passed,
        "complete": True,
        "configuration_hash": configuration_hash,
        "truth": {"passed": truth_passed, "diagnostics": truth},
        "data": {
            "passed": all(data_checks),
            "replicate_checks": data_checks,
            "minimum_train_events_per_cause": MIN_TRAIN_EVENTS_PER_CAUSE,
            "minimum_test_events_per_cause": MIN_TEST_EVENTS_PER_CAUSE,
            "evaluation_grid_must_cover": list(CRITICAL_EVALUATION_TIMES),
        },
        "affine_truth": {
            "passed": all(affine_checks),
            "replicate_checks": affine_checks,
            "minimum_nonconstant_r_squared": MIN_AFFINE_MU_R2,
            "constant_cause_exact_fit_r_squared": 1.0,
            "maximum_static_affine_error": MAX_STATIC_AFFINE_ERROR,
            "maximum_reconstruction_error": (MAX_AFFINE_MU_RECONSTRUCTION_ERROR),
        },
        "time_zero_atom": {
            "passed": all(atom_checks),
            "replicate_checks": atom_checks,
            "maximum_probability_mean": MAX_INITIAL_EVENT_PROBABILITY_MEAN,
            "maximum_probability_p99": MAX_INITIAL_EVENT_PROBABILITY_P99,
            "maximum_probability": MAX_INITIAL_EVENT_PROBABILITY_MAX,
            "count_discrepancy_floor": MAX_ATOM_COUNT_DISCREPANCY_FLOOR,
            "standard_deviation_multiplier": (ATOM_COUNT_STANDARD_DEVIATION_MULTIPLIER),
        },
        "cscox": {
            "passed": cscox_passed,
            "values": cscox_values,
            "mean": statistics.mean(cscox_values),
            "maximum": max(cscox_values),
            "mean_ctd_rank": cscox_rank,
            "maximum_allowed_mean": MAX_CSCOX_MEAN_CTD,
            "maximum_allowed_replicate": MAX_CSCOX_REPLICATE_CTD,
            "minimum_required_rank": MIN_CSCOX_MEAN_RANK,
        },
        "softcomp": {
            "passed": softcomp_passed,
            "ranks": {metric: ranks[metric]["SoftComp"] for metric in core_metrics},
        },
        "fine_gray_convergence": {
            "passed": fine_gray_convergence_passed,
            "replicate_checks": fine_gray_convergence_checks,
        },
        "fine_gray_aco": {
            "passed": fine_gray_aco_passed,
            "replicates": fine_gray_aco,
            "negative_survival_atol": NEGATIVE_SURVIVAL_ATOL,
            "minimum_points_per_replicate": (MIN_FINE_GRAY_ACO_POINTS_PER_REPLICATE),
            "minimum_subjects_per_replicate": (
                MIN_FINE_GRAY_ACO_SUBJECTS_PER_REPLICATE
            ),
            "minimum_required_overflow": (MAX_FINE_GRAY_MINIMUM_IMPLIED_SURVIVAL),
            "scope": "test subjects on each replicate's primary evaluation grid",
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
    parameters = generate_frozen_parameters(config.dgp_seed)
    configuration_hash = compute_configuration_hash(config)
    execution_hash = compute_execution_hash(config)
    truth = truth_diagnostics(parameters)
    if config.phase == "formal":
        gate_file = args.gate_file or (
            get_output_dir("case1_v3") / "development_gate" / "gate.json"
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

