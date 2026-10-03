#!/usr/bin/env python3
"""Run one standard-scale Case II v4 experiment."""

from __future__ import annotations

import argparse
import functools
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from ...baseline_models import CsCox, DeepHit, DSM, FineGray, NeuralFG
from ...crsoft_model import CRSoftNet
from ...data import case2_v4
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

K = 3
P = 3
ALL_MODELS = ("DeepHit", "DSM", "cs-Cox", "NeuralFG", "SoftComp", "Fine-Gray")


@dataclass(frozen=True)
class SoftCompConfig:
    hidden_dim: int = 32
    num_blocks: int = 1
    epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 3e-3
    n_aug: int = 2
    aug_weight: float = 0.5


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
    true_cif_fn: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]
    case_name: str
    t_max: float


def _record_training_history(model: object, history: list[float]) -> None:
    vars(model)["_training_history"] = history


def training_history(model: object) -> list[float]:
    """Return a model's serializable training-loss history, when applicable."""
    history = vars(model).get("_training_history", [])
    return [float(value) for value in history]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--train-seed", type=int, default=30_000)
    parser.add_argument("--test-seed", type=int, default=40_000)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--dgp-seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS))
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-3)
    parser.add_argument("--n-aug", type=int, default=2)
    parser.add_argument("--aug-weight", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        help="optionally save test-set survival curves for plotting",
    )
    return parser.parse_args()


def prepare_data(
    *,
    n_train: int,
    n_test: int,
    train_seed: int,
    test_seed: int,
    dgp_seed: int = 42,
    parameters: dict[str, Tensor] | None = None,
    censor_rate: float = 0.5,
) -> PreparedData:
    """Generate one paired Case II v4 train/test data set."""
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must be positive")
    params = (
        case2_v4.generate_parameters(K=K, p=P, seed=dgp_seed)
        if parameters is None
        else parameters
    )
    generated = case2_v4.generate_data(
        n=n_train,
        K=K,
        p=P,
        censor_rate=censor_rate,
        seed=train_seed,
        params=params,
    )
    true_cif_fn = functools.partial(case2_v4.compute_cif, params=params)
    x_test, y_test, delta_test = generate_test_observations(
        n_test,
        P,
        true_cif_fn,
        censor_rate=censor_rate,
        seed=test_seed,
        t_max=case2_v4.T_MAX,
    )
    eval_times = build_evaluation_time_grid(y_test, delta_test, n_grid=100)
    n_val = int(n_train * 0.1)
    if n_val == 0:
        raise ValueError("n_train must leave at least one validation subject")
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
        params=params,
        true_cif_fn=true_cif_fn,
        case_name="case2_v4",
        t_max=case2_v4.T_MAX,
    )


def _deep_hit_spec(data: PreparedData) -> ModelSpec:
    def train() -> DeepHit:
        model = DeepHit(
            n_bins=100,
            n_causes=K,
            hidden_dim=64,
            n_layers=2,
            lr=1e-3,
            batch_size=256,
            epochs=500,
            patience=50,
            alpha=0.2,
            sigma=0.1,
        )
        history = model.fit(
            data.X_train_fit,
            data.Y_train_fit,
            data.Delta_train_fit,
            data.X_val,
            data.Y_val,
            data.Delta_val,
            verbose=False,
            device="cpu",
        )
        _record_training_history(model, history)
        return model

    return ModelSpec(
        name="DeepHit",
        cache_key="deephit",
        train=train,
        serialize=lambda model: {
            "state_dict": model.net.state_dict(),
            "bin_edges": model.bin_edges,
            "n_bins": model.n_bins,
            "n_causes": model.n_causes,
            "hidden_dim": model.hidden_dim,
            "n_layers": model.n_layers,
        },
        load_ckpt=DeepHit.load_from_checkpoint,
        predict=lambda model, x, times: model.predict_cif(x, times),
        predict_survival=lambda model, x, times: model.predict_cif_survival(x, times),
    )


def _dsm_spec(data: PreparedData) -> ModelSpec:
    def train() -> DSM:
        model = DSM(
            n_causes=K,
            n_features=P,
            k=6,
            layers=[64, 64],
            distribution="Weibull",
        )
        history = model.fit(
            data.X_train_fit,
            data.Y_train_fit,
            data.Delta_train_fit,
            data.X_val,
            data.Y_val,
            data.Delta_val,
            epochs=500,
            lr=1e-3,
            batch_size=256,
            patience=50,
            device="cpu",
        )
        _record_training_history(model, history)
        return model

    return ModelSpec(
        name="DSM",
        cache_key="dsm",
        train=train,
        serialize=lambda model: {
            "state_dict": model.net.state_dict(),
            "n_causes": model.n_causes,
            "n_features": model.n_features,
            "k": model.k,
            "layers": model.layers,
            "distribution": model.distribution,
        },
        load_ckpt=DSM.load_from_checkpoint,
        predict=lambda model, x, times: model.predict_cif(x, times),
        predict_survival=lambda model, x, times: model.predict_cif_survival(x, times),
    )


def _cs_cox_spec(data: PreparedData) -> ModelSpec:
    def train() -> CsCox:
        model = CsCox(n_causes=K)
        model.fit(data.X_train_fit, data.Y_train_fit, data.Delta_train_fit)
        return model

    return ModelSpec(
        name="cs-Cox",
        cache_key="cs_cox",
        train=train,
        serialize=lambda model: {
            "models": model.models,
            "baseline_hazards": model.baseline_hazards,
            "n_causes": model.n_causes,
            "penalizer": model.penalizer,
        },
        load_ckpt=CsCox.load_from_checkpoint,
        predict=lambda model, x, times: model.predict_cif(x, times),
        predict_survival=lambda model, x, times: model.predict_cif_survival(x, times),
    )


def _fine_gray_spec(data: PreparedData) -> ModelSpec:
    def train() -> FineGray:
        model = FineGray(n_causes=K)
        model.fit(data.X_train_fit, data.Y_train_fit, data.Delta_train_fit)
        return model

    return ModelSpec(
        name="Fine-Gray",
        cache_key="fine_gray",
        cli_aliases=["fg", "finegray"],
        train=train,
        serialize=lambda model: model.to_checkpoint(),
        load_ckpt=FineGray.load_from_checkpoint,
        predict=lambda model, x, times: model.predict_cif(x, times),
        predict_survival=lambda model, x, times: model.predict_cif_survival(x, times),
    )


def _neural_fg_spec(data: PreparedData) -> ModelSpec:
    def train() -> NeuralFG:
        model = NeuralFG(
            n_causes=K,
            layers=[100, 100, 100],
            layers_surv=[100],
            lr=1e-3,
            batch_size=100,
            epochs=1000,
            patience=3,
            weight_decay=1e-3,
        )
        history = model.fit(
            data.X_train_fit,
            data.Y_train_fit,
            data.Delta_train_fit,
            data.X_val,
            data.Y_val,
            data.Delta_val,
            verbose=False,
            device="cpu",
        )
        _record_training_history(model, history)
        return model

    return ModelSpec(
        name="NeuralFG",
        cache_key="neural_fg",
        train=train,
        serialize=lambda model: {
            "state_dict": model.net.state_dict(),
            "n_causes": model.n_causes,
            "layers": model.layers,
            "layers_surv": model.layers_surv,
            "_time_max": model._time_max,
        },
        load_ckpt=NeuralFG.load_from_checkpoint,
        predict=lambda model, x, times: model.predict_cif(x, times),
        predict_survival=lambda model, x, times: model.predict_cif_survival(x, times),
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
        history = model.fit(
            data.X_train_fit,
            data.Y_train_fit,
            data.Delta_train_fit,
            epochs=config.epochs,
            lr=config.lr,
            weight_decay=config.weight_decay,
            n_aug=config.n_aug,
            aug_weight=config.aug_weight,
            verbose=False,
            device="cpu",
        )
        _record_training_history(model, history)
        return model

    return ModelSpec(
        name="SoftComp",
        cache_key="crsoft",
        train=train,
        serialize=lambda model: {
            "state_dict": model.state_dict(),
            "num_causes": model.num_causes,
            "input_dim": P,
            "hidden_dim": config.hidden_dim,
            "num_blocks": config.num_blocks,
        },
        load_ckpt=CRSoftNet.load_from_checkpoint,
        predict=lambda model, x, times: model.predict_cif_grid(x, times),
        predict_survival=lambda model, x, times: model.predict_cif_survival_grid(
            x, times
        ),
        post_process=lambda cif: enforce_cif_simplex(isotonic_project_cif(cif)),
    )


def build_specs(
    data: PreparedData,
    config: SoftCompConfig,
    model_seed: int,
) -> list[ModelSpec]:
    """Build all six pre-registered Case II v4 method specifications."""
    return [
        _deep_hit_spec(data),
        _dsm_spec(data),
        _cs_cox_spec(data),
        _neural_fg_spec(data),
        _softcomp_spec(data, config, model_seed),
        _fine_gray_spec(data),
    ]


def _select_specs(specs: list[ModelSpec], requested: list[str]) -> list[ModelSpec]:
    selected = []
    for token in requested:
        spec = next(
            (candidate for candidate in specs if candidate.matches(token)), None
        )
        if spec is None:
            raise ValueError(f"unknown model: {token}")
        if spec not in selected:
            selected.append(spec)
    return selected


def _evaluate(
    cif: Tensor, survival: Tensor, data: PreparedData
) -> dict[str, float | int]:
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
    return metrics


def _all_subject_probability_diagnostics(
    spec: ModelSpec,
    model: object,
    test_cif: Tensor,
    test_survival: Tensor,
    data: PreparedData,
) -> tuple[dict[str, float | int], float]:
    if spec.predict_survival is None:
        raise RuntimeError(f"{spec.name} does not expose native survival")
    start = time.perf_counter()
    train_cif, train_survival = spec.predict_survival(
        model, data.X_train_full, data.eval_times
    )
    if spec.cache_key == "crsoft":
        train_cif = spec.post_process(train_cif)
    all_cif = torch.cat([train_cif, test_cif], dim=0)
    all_survival = torch.cat([train_survival, test_survival], dim=0)
    diagnostics = compute_dist(all_cif, all_survival)
    return diagnostics, time.perf_counter() - start


def _true_survival_grid(data: PreparedData) -> Tensor:
    survival = torch.empty(data.X_test.shape[0], data.eval_times.shape[0])
    with torch.no_grad():
        for time_index, evaluation_time in enumerate(data.eval_times):
            _, values = data.true_cif_fn(
                data.X_test,
                evaluation_time.expand(data.X_test.shape[0]),
            )
            survival[:, time_index] = values
    return survival


def _save_survival_predictions(
    output_dir: Path,
    spec: ModelSpec,
    cif: Tensor,
    native_survival: Tensor,
    data: PreparedData,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": spec.name,
            "X_test": data.X_test.detach().cpu(),
            "eval_times": data.eval_times.detach().cpu(),
            "true_survival": _true_survival_grid(data).detach().cpu(),
            "implied_survival": (1.0 - cif.sum(dim=1)).detach().cpu(),
            "native_survival": native_survival.detach().cpu(),
        },
        output_dir / f"{spec.cache_key}.pt",
    )


def _run_specs(
    specs: list[ModelSpec],
    data: PreparedData,
    model_seed: int,
    prediction_dir: Path | None,
) -> dict[str, dict[str, object]]:
    results = {}
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
        raw_metrics = _evaluate(cif, survival, data)
        pav_time = 0.0
        if spec.cache_key == "crsoft":
            start = time.perf_counter()
            cif = spec.post_process(cif)
            pav_time = time.perf_counter() - start
        metrics = _evaluate(cif, survival, data)
        if prediction_dir is not None:
            _save_survival_predictions(prediction_dir, spec, cif, survival, data)
        all_diagnostics, diagnostic_time = _all_subject_probability_diagnostics(
            spec, model, cif, survival, data
        )
        total_time = train_time + predict_time + pav_time
        results[spec.name] = {
            "metrics": metrics,
            "raw_metrics": raw_metrics,
            "all_subject_probability_diagnostics": all_diagnostics,
            "timings": {
                "train_time_sec": train_time,
                "predict_time_sec": predict_time,
                "pav_time_sec": pav_time,
                "total_time_sec": total_time,
                "diagnostic_prediction_time_sec": diagnostic_time,
            },
        }
        print(
            f"{spec.name:<10} MSE={metrics['MSE_overall']:.6f} "
            f"Ctd={metrics['Ctd_overall']:.6f} "
            f"IBS={metrics['IBS_overall']:.6f} "
            f"Dist={metrics['Dist']:.6f} "
            f"all neg subjects="
            f"{all_diagnostics['Implied_S_negative_subject_count']}/"
            f"{all_diagnostics['Implied_S_total_subject_count']} "
            f"time={total_time:.2f}s",
            flush=True,
        )
    return results


def _linear_r_squared(data: PreparedData) -> Tensor:
    train_scores = case2_v4.compute_static_logits(data.X_train_full, data.params)
    test_scores = case2_v4.compute_static_logits(data.X_test, data.params)
    train_design = torch.cat(
        [torch.ones(data.X_train_full.shape[0], 1), data.X_train_full], dim=1
    )
    test_design = torch.cat([torch.ones(data.X_test.shape[0], 1), data.X_test], dim=1)
    coefficients = torch.linalg.lstsq(train_design, train_scores).solution
    residual = test_scores - test_design @ coefficients
    total = test_scores - test_scores.mean(dim=0, keepdim=True)
    return 1.0 - residual.square().sum(dim=0) / total.square().sum(dim=0).clamp(
        min=1e-12
    )


def _diagnostics(data: PreparedData) -> dict[str, object]:
    linear_r_squared = _linear_r_squared(data)
    diagnostics: dict[str, object] = {
        "case": data.case_name,
        "K": K,
        "P": P,
        "n_train_fit": int(data.X_train_fit.shape[0]),
        "beta_min": float(data.params["beta"].min().item()),
        "beta_mean": float(data.params["beta"].mean().item()),
        "beta_max": float(data.params["beta"].max().item()),
        "time_scale": float(data.params["time_scale"].item()),
        "train_counts_censor_then_causes": [
            int((data.Delta_train_full == cause).sum().item()) for cause in range(K + 1)
        ],
        "test_counts_censor_then_causes": [
            int((data.Delta_test == cause).sum().item()) for cause in range(K + 1)
        ],
        "train_event_time_at_limit_fraction": float(
            (data.T_train_true >= data.t_max - 1e-3).float().mean().item()
        ),
        "eval_time_min": float(data.eval_times[0].item()),
        "eval_time_max": float(data.eval_times[-1].item()),
        "static_score_linear_r2_min": float(linear_r_squared.min().item()),
        "static_score_linear_r2_mean": float(linear_r_squared.mean().item()),
        "static_score_linear_r2_max": float(linear_r_squared.max().item()),
    }
    for name in ("quadratic_weight", "pairwise_weight", "radial_weight"):
        value = data.params.get(name)
        if value is not None:
            diagnostics[name] = float(value.item())
    return diagnostics


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
    data = prepare_data(
        n_train=args.n_train,
        n_test=args.n_test,
        train_seed=args.train_seed,
        test_seed=args.test_seed,
        dgp_seed=args.dgp_seed,
    )
    diagnostics = _diagnostics(data)
    print(f"DGP diagnostics: {json.dumps(diagnostics, sort_keys=True)}")
    specs = _select_specs(build_specs(data, config, args.model_seed), args.models)
    results = _run_specs(specs, data, args.model_seed, args.prediction_dir)
    payload = {
        "protocol": {
            "case": "case2_v4",
            "n_train": args.n_train,
            "n_test": args.n_test,
            "train_seed": args.train_seed,
            "test_seed": args.test_seed,
            "model_seed": args.model_seed,
            "dgp_seed": args.dgp_seed,
            "cpu_threads": args.cpu_threads,
            "models": args.models,
            "softcomp": asdict(config),
            "softcomp_postprocessing": (
                "PAV followed by global-time simplex rescaling"
            ),
        },
        "diagnostics": diagnostics,
        "results": results,
    }
    output = args.output or (get_output_dir("case2_v4") / "single_run.json")
    _write_json(output, payload)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
