#!/usr/bin/env python3
"""Run one standard-scale Case I v3 experiment."""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...baseline_models import CsCox, DeepHit, DSM, FineGray, NeuralFG
from ...data import case1_v3
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

logger: logging.Logger = logging.getLogger(__name__)

K = case1_v3.K
P = case1_v3.P
ALL_MODELS = ("DeepHit", "DSM", "cs-Cox", "NeuralFG", "SoftComp", "Fine-Gray")
EVALUATION_PERCENTILE_CAP = 97.5
CRITICAL_EVALUATION_TIMES = (1.5, 8.0, 16.0, 24.0)


@dataclass(frozen=True)
class SoftCompConfig:
    optimizer: str = "lbfgs"
    standardize_x: bool = True
    initial_alpha: float = 1.0
    max_iter: int = 100
    lr: float = 1.0
    l2_penalty: float = 0.0
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    history_size: int = 50
    atom_tolerance: float = 1e-8


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


class AffineExactSoftComp(nn.Module):
    """Case I SoftComp with affine logits and the exact event-time likelihood."""

    def __init__(
        self,
        input_dim: int,
        num_causes: int,
        *,
        initial_alpha: float = 1.0,
        atom_tolerance: float = 1e-8,
        alpha_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_causes <= 0:
            raise ValueError("input_dim and num_causes must be positive")
        if initial_alpha <= alpha_epsilon:
            raise ValueError("initial_alpha must exceed alpha_epsilon")
        if atom_tolerance < 0.0:
            raise ValueError("atom_tolerance must be nonnegative")
        self.input_dim = input_dim
        self.num_causes = num_causes
        self.atom_tolerance = atom_tolerance
        self.alpha_epsilon = alpha_epsilon
        self.intercept = nn.Parameter(torch.zeros(num_causes))
        self.beta = nn.Parameter(torch.zeros(num_causes, input_dim))
        raw_alpha = _inverse_softplus(initial_alpha - alpha_epsilon)
        self.raw_alpha = nn.Parameter(torch.tensor(raw_alpha))
        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_scale", torch.ones(input_dim))

    @property
    def alpha(self) -> Tensor:
        return F.softplus(self.raw_alpha) + self.alpha_epsilon

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _standardize(self, x: Tensor) -> Tensor:
        return (x - self.x_mean) / self.x_scale

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x must have shape (n, {self.input_dim})")
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        if t.shape != (x.shape[0],):
            raise ValueError("t must be scalar or contain one value per subject")
        return (
            self.intercept.unsqueeze(0)
            + self._standardize(x) @ self.beta.T
            + self.alpha * t.unsqueeze(1)
        )

    def predict_cif(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        expanded_time = t.expand(x.shape[0]) if t.dim() == 0 else t
        nonnegative_time = expanded_time.clamp(min=0.0)
        logits = self(x, nonnegative_time)
        full_logits = torch.cat(
            [torch.zeros(len(x), 1, device=x.device, dtype=x.dtype), logits],
            dim=1,
        )
        probabilities = torch.softmax(full_logits, dim=1)
        cif = probabilities[:, 1:]
        survival = probabilities[:, 0]
        before_support = expanded_time < 0.0
        cif = torch.where(before_support.unsqueeze(1), torch.zeros_like(cif), cif)
        survival = torch.where(before_support, torch.ones_like(survival), survival)
        return cif, survival

    def exact_negative_log_likelihood(
        self,
        x: Tensor,
        y: Tensor,
        delta: Tensor,
        *,
        reduction: str = "mean",
    ) -> Tensor:
        """Evaluate the atom-aware exact likelihood for the affine CIF model."""
        if y.shape != (len(x),) or delta.shape != (len(x),):
            raise ValueError("y and delta must contain one value per subject")
        if bool((y < 0.0).any().item()):
            raise ValueError("event and censoring times must be nonnegative")
        delta_long = delta.long()
        if bool((delta != delta_long).any().item()) or bool(
            ((delta_long < 0) | (delta_long > self.num_causes)).any().item()
        ):
            raise ValueError("delta must contain integers from 0 through num_causes")
        event = delta_long > 0
        atom = event & (y <= self.atom_tolerance)
        likelihood_time = torch.where(atom, torch.zeros_like(y), y)
        logits = self(x, likelihood_time)
        full_logits = torch.cat(
            [torch.zeros(len(x), 1, device=x.device, dtype=x.dtype), logits],
            dim=1,
        )
        log_probabilities = F.log_softmax(full_logits, dim=1)
        log_survival = log_probabilities[:, 0]
        selected_log_probability = log_probabilities.gather(
            1, delta_long.unsqueeze(1)
        ).squeeze(1)
        log_likelihood = torch.where(event, selected_log_probability, log_survival)
        continuous_event = event & ~atom
        continuous_adjustment = torch.log(self.alpha) + log_survival
        log_likelihood = torch.where(
            continuous_event,
            log_likelihood + continuous_adjustment,
            log_likelihood,
        )
        losses = -log_likelihood
        if reduction == "none":
            return losses
        if reduction == "sum":
            return losses.sum()
        if reduction == "mean":
            return losses.mean()
        raise ValueError("reduction must be one of: none, sum, mean")

    def _objective(
        self,
        x: Tensor,
        y: Tensor,
        delta: Tensor,
        l2_penalty: float,
    ) -> Tensor:
        loss = self.exact_negative_log_likelihood(x, y, delta)
        if l2_penalty > 0.0:
            loss = loss + l2_penalty * self.beta.square().mean()
        return loss

    def _set_standardization(self, x: Tensor, enabled: bool) -> None:
        with torch.no_grad():
            if enabled:
                self.x_mean.copy_(x.mean(dim=0))
                self.x_scale.copy_(x.std(dim=0, unbiased=False).clamp(min=1e-6))
            else:
                self.x_mean.zero_()
                self.x_scale.fill_(1.0)

    def _initialize_parameters(self, y: Tensor, delta: Tensor, alpha: float) -> None:
        reference_time = float(y.median().item())
        censor_count = float((delta == 0).sum().item()) + 0.5
        event_counts = torch.stack(
            [(delta == cause).sum() for cause in range(1, self.num_causes + 1)]
        ).to(dtype=self.intercept.dtype)
        initial_intercept = (
            torch.log((event_counts + 0.5) / censor_count) - alpha * reference_time
        )
        with torch.no_grad():
            self.intercept.copy_(initial_intercept)
            self.beta.zero_()
            self.raw_alpha.fill_(_inverse_softplus(alpha - self.alpha_epsilon))

    @staticmethod
    def _validate_config(config: SoftCompConfig) -> None:
        if config.optimizer != "lbfgs":
            raise ValueError("optimizer must be lbfgs")
        if config.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        positive_values = (
            config.initial_alpha,
            config.lr,
            config.tolerance_grad,
            config.tolerance_change,
        )
        if any(value <= 0.0 or not math.isfinite(value) for value in positive_values):
            raise ValueError("positive optimizer settings must be finite")
        if config.l2_penalty < 0.0 or config.history_size <= 0:
            raise ValueError("regularization and history settings are invalid")

    def fit(
        self,
        x: Tensor,
        y: Tensor,
        delta: Tensor,
        config: SoftCompConfig,
        *,
        device: torch.device | str = "cpu",
    ) -> list[float]:
        """Fit the 19-parameter affine model by deterministic full-batch MLE."""
        self._validate_config(config)
        resolved_device = torch.device(device)
        self.to(resolved_device)
        parameter_dtype = self.intercept.dtype
        x = x.to(device=resolved_device, dtype=parameter_dtype)
        y = y.to(device=resolved_device, dtype=parameter_dtype)
        delta = delta.to(device=resolved_device)
        self._set_standardization(x, config.standardize_x)
        self._initialize_parameters(y, delta, config.initial_alpha)
        history = [float(self._objective(x, y, delta, config.l2_penalty).item())]
        lbfgs = torch.optim.LBFGS(
            self.parameters(),
            lr=config.lr,
            max_iter=config.max_iter,
            tolerance_grad=config.tolerance_grad,
            tolerance_change=config.tolerance_change,
            history_size=config.history_size,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss = self._objective(x, y, delta, config.l2_penalty)
            loss.backward()
            history.append(float(loss.item()))
            return loss

        lbfgs.step(closure)
        final_loss = float(self._objective(x, y, delta, config.l2_penalty).item())
        if not math.isfinite(final_loss):
            raise RuntimeError(
                "SoftComp exact-likelihood optimization became nonfinite"
            )
        history.append(final_loss)
        self.eval()
        return history

    @torch.no_grad()
    def predict_cif_grid(self, x: Tensor, times: Tensor) -> Tensor:
        cif, _ = self.predict_cif_survival_grid(x, times)
        return cif

    @torch.no_grad()
    def predict_cif_survival_grid(
        self,
        x: Tensor,
        times: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if times.dim() != 1:
            raise ValueError("times must be one-dimensional")
        self.eval()
        device = self.intercept.device
        dtype = self.intercept.dtype
        x = x.to(device=device, dtype=dtype)
        times = times.to(device=device, dtype=dtype)
        static_logits = self.intercept.unsqueeze(0) + self._standardize(x) @ self.beta.T
        logits = static_logits.unsqueeze(1) + self.alpha * times.reshape(1, -1, 1)
        full_logits = torch.cat(
            [
                torch.zeros(len(x), len(times), 1, device=device, dtype=dtype),
                logits,
            ],
            dim=2,
        )
        probabilities = torch.softmax(full_logits, dim=2)
        cif = probabilities[:, :, 1:].permute(0, 2, 1)
        survival = probabilities[:, :, 0]
        before_support = times < 0.0
        cif[:, :, before_support] = 0.0
        survival[:, before_support] = 1.0
        return cif.cpu(), survival.cpu()

    def to_checkpoint(self) -> dict[str, object]:
        return {
            "model_type": "case1_v3_affine_exact_softcomp",
            "state_dict": {
                name: value.detach().cpu() for name, value in self.state_dict().items()
            },
            "input_dim": self.input_dim,
            "num_causes": self.num_causes,
            "atom_tolerance": self.atom_tolerance,
            "alpha_epsilon": self.alpha_epsilon,
        }

    @classmethod
    def load_from_checkpoint(cls, data: dict) -> AffineExactSoftComp:
        if data.get("model_type") != "case1_v3_affine_exact_softcomp":
            raise ValueError("checkpoint is not a Case I affine exact SoftComp model")
        model = cls(
            input_dim=int(data["input_dim"]),
            num_causes=int(data["num_causes"]),
            atom_tolerance=float(data["atom_tolerance"]),
            alpha_epsilon=float(data["alpha_epsilon"]),
        )
        model.load_state_dict(data["state_dict"])
        model.eval()
        return model


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


def truth(
    x: Tensor,
    times: Tensor,
    params: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """Evaluate the exact CIF and survival surfaces on a common time grid."""
    if times.dim() != 1:
        raise ValueError("times must be one-dimensional")
    cif = torch.empty(x.shape[0], K, times.shape[0], dtype=x.dtype)
    survival = torch.empty(x.shape[0], times.shape[0], dtype=x.dtype)
    with torch.no_grad():
        for index, evaluation_time in enumerate(times):
            values, survival_values = case1_v3.compute_cif(
                x,
                evaluation_time.expand(x.shape[0]),
                params,
            )
            cif[:, :, index] = values
            survival[:, index] = survival_values
    return cif, survival


def prepare_data(
    *,
    n_train: int,
    n_test: int,
    train_seed: int,
    test_seed: int,
    dgp_seed: int = 42,
    parameters: dict[str, Tensor] | None = None,
) -> PreparedData:
    """Generate one paired Case I v3 train/test data set."""
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must be positive")
    params = (
        case1_v3.generate_parameters(K=K, p=P, seed=dgp_seed)
        if parameters is None
        else parameters
    )
    generated = case1_v3.generate_data(
        n=n_train,
        K=K,
        p=P,
        censor_rate=0.5,
        seed=train_seed,
        params=params,
    )
    generated_test = case1_v3.generate_data(
        n=n_test,
        K=K,
        p=P,
        censor_rate=0.5,
        seed=test_seed,
        params=params,
    )
    true_cif_fn = functools.partial(case1_v3.compute_cif, params=params)
    x_test = generated_test["X"]
    y_test = generated_test["Y"]
    delta_test = generated_test["Delta"]
    if not all(isinstance(value, Tensor) for value in (x_test, y_test, delta_test)):
        raise TypeError("Case I v3 test generator returned an invalid tensor payload")
    quantile_times = build_evaluation_time_grid(
        y_test,
        delta_test,
        n_grid=100,
        percentile_cap=EVALUATION_PERCENTILE_CAP,
    )
    critical_times = torch.tensor(CRITICAL_EVALUATION_TIMES, dtype=quantile_times.dtype)
    eval_times = torch.cat([quantile_times, critical_times]).unique().sort().values
    n_val = int(n_train * 0.1)
    if n_val == 0:
        raise ValueError("n_train must leave at least one validation subject")
    x_train = generated["X"]
    y_train = generated["Y"]
    delta_train = generated["Delta"]
    t_train = generated["T_true"]
    if not all(
        isinstance(value, Tensor) for value in (x_train, y_train, delta_train, t_train)
    ):
        raise TypeError("Case I v3 data generator returned an invalid tensor payload")
    return PreparedData(
        X_train_full=x_train,
        Y_train_full=y_train,
        Delta_train_full=delta_train,
        T_train_true=t_train,
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
        true_cif_fn=true_cif_fn,
        case_name="case1_v3",
        t_max=case1_v3.T_MAX,
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
    def train() -> AffineExactSoftComp:
        torch.manual_seed(model_seed)
        model = AffineExactSoftComp(
            input_dim=P,
            num_causes=K,
            initial_alpha=config.initial_alpha,
            atom_tolerance=config.atom_tolerance,
        )
        history = model.fit(
            data.X_train_fit,
            data.Y_train_fit,
            data.Delta_train_fit,
            config,
            device="cpu",
        )
        _record_training_history(model, history)
        return model

    return ModelSpec(
        name="SoftComp",
        cache_key="crsoft",
        train=train,
        serialize=lambda model: {
            **model.to_checkpoint(),
            "training_config": asdict(config),
        },
        load_ckpt=AffineExactSoftComp.load_from_checkpoint,
        predict=lambda model, x, times: model.predict_cif_grid(x, times),
        predict_survival=lambda model, x, times: model.predict_cif_survival_grid(
            x, times
        ),
        post_process=lambda cif: enforce_cif_simplex(isotonic_project_cif(cif)),
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


def build_specs(
    data: PreparedData,
    config: SoftCompConfig,
    model_seed: int,
) -> list[ModelSpec]:
    """Build all six pre-registered Case I v3 method specifications."""
    return [
        _deep_hit_spec(data),
        _dsm_spec(data),
        _cs_cox_spec(data),
        _neural_fg_spec(data),
        _softcomp_spec(data, config, model_seed),
        _fine_gray_spec(data),
    ]


def _select_specs(
    specs: Sequence[ModelSpec],
    requested: Sequence[str],
) -> list[ModelSpec]:
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
    cif: Tensor,
    survival: Tensor,
    data: PreparedData,
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
    train_cif = spec.post_process(train_cif)
    all_cif = torch.cat([train_cif, test_cif], dim=0)
    all_survival = torch.cat([train_survival, test_survival], dim=0)
    return compute_dist(all_cif, all_survival), time.perf_counter() - start


def _save_predictions(
    output_dir: Path,
    spec: ModelSpec,
    cif: Tensor,
    native_survival: Tensor,
    data: PreparedData,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    true_cif, true_survival = truth(data.X_test, data.eval_times, data.params)
    torch.save(
        {
            "model_name": spec.name,
            "X_test": data.X_test.detach().cpu(),
            "Y_test": data.Y_test.detach().cpu(),
            "Delta_test": data.Delta_test.detach().cpu(),
            "eval_times": data.eval_times.detach().cpu(),
            "true_cif": true_cif.detach().cpu(),
            "true_survival": true_survival.detach().cpu(),
            "cif": cif.detach().cpu(),
            "implied_survival": (1.0 - cif.sum(dim=1)).detach().cpu(),
            "native_survival": native_survival.detach().cpu(),
        },
        output_dir / f"{spec.cache_key}.pt",
    )


def _run_specs(
    specs: Sequence[ModelSpec],
    data: PreparedData,
    model_seed: int,
    prediction_dir: Path | None,
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
        raw_cif, survival = spec.predict_survival(model, data.X_test, data.eval_times)
        predict_time = time.perf_counter() - start
        raw_metrics = _evaluate(raw_cif, survival, data)
        start = time.perf_counter()
        cif = spec.post_process(raw_cif)
        postprocess_time = time.perf_counter() - start
        if spec.cache_key != "crsoft":
            postprocess_time = 0.0
        metrics = _evaluate(cif, survival, data)
        if prediction_dir is not None:
            _save_predictions(prediction_dir, spec, cif, survival, data)
        all_diagnostics, diagnostic_time = _all_subject_probability_diagnostics(
            spec,
            model,
            cif,
            survival,
            data,
        )
        total_time = train_time + predict_time + postprocess_time
        results[spec.name] = {
            "metrics": metrics,
            "raw_metrics": raw_metrics,
            "all_subject_probability_diagnostics": all_diagnostics,
            "timings": {
                "train_time_sec": train_time,
                "predict_time_sec": predict_time,
                "pav_time_sec": postprocess_time,
                "total_time_sec": total_time,
                "diagnostic_prediction_time_sec": diagnostic_time,
            },
        }
        logger.info(
            "%s MSE=%.6f Ctd=%.6f IBS=%.6f Dist=%.6f ACO=%d time=%.2fs",
            spec.name,
            metrics["MSE_overall"],
            metrics["Ctd_overall"],
            metrics["IBS_overall"],
            metrics["Dist"],
            all_diagnostics["Implied_S_below_tolerance_count"],
            total_time,
        )
    return results


def _heldout_affine_r_squared(x: Tensor, target: Tensor) -> Tensor:
    if target.dim() == 1:
        target = target.unsqueeze(1)
    midpoint = len(x) // 2
    fit_design = torch.cat([torch.ones(midpoint, 1), x[:midpoint]], dim=1)
    test_design = torch.cat([torch.ones(len(x) - midpoint, 1), x[midpoint:]], dim=1)
    coefficients = torch.linalg.lstsq(fit_design, target[:midpoint]).solution
    residual = target[midpoint:] - test_design @ coefficients
    centered = target[midpoint:] - target[midpoint:].mean(dim=0, keepdim=True)
    residual_sum = residual.square().sum(dim=0)
    centered_sum = centered.square().sum(dim=0)
    return torch.where(
        centered_sum <= 1e-10,
        (residual_sum <= 1e-10).to(target.dtype),
        1.0 - residual_sum / centered_sum,
    )


def data_diagnostics(data: PreparedData) -> dict[str, object]:
    """Summarize data balance, the time-zero atom, and affine truth checks."""
    logits = case1_v3.compute_static_logits(data.X_train_full, data.params)
    initial_cif, initial_survival = case1_v3.compute_cif(
        data.X_train_full,
        torch.zeros(data.X_train_full.shape[0]),
        data.params,
    )
    cause_probabilities = torch.softmax(logits, dim=1)
    linear_r_squared = _heldout_affine_r_squared(data.X_train_full, logits)
    diagnostics: dict[str, object] = {
        "case": data.case_name,
        "K": K,
        "p": P,
        "n_train_fit": int(data.X_train_fit.shape[0]),
        "train_counts_censor_then_causes": [
            int((data.Delta_train_full == cause).sum().item()) for cause in range(K + 1)
        ],
        "test_counts_censor_then_causes": [
            int((data.Delta_test == cause).sum().item()) for cause in range(K + 1)
        ],
        "train_event_time_zero_fraction": float(
            (data.T_train_true <= 1e-7).float().mean().item()
        ),
        "eval_time_min": float(data.eval_times[0].item()),
        "eval_time_max": float(data.eval_times[-1].item()),
        "cause_probability_min": float(cause_probabilities.min().item()),
        "cause_probability_max": float(cause_probabilities.max().item()),
        "initial_event_probability_min": float(initial_cif.sum(dim=1).min().item()),
        "initial_event_probability_mean": float(initial_cif.sum(dim=1).mean().item()),
        "initial_event_probability_max": float(initial_cif.sum(dim=1).max().item()),
        "initial_survival_min": float(initial_survival.min().item()),
        "affine_mu_r2_min": float(linear_r_squared.min().item()),
        "affine_mu_r2_mean": float(linear_r_squared.mean().item()),
        "affine_mu_r2_max": float(linear_r_squared.max().item()),
        "beta": data.params["beta"].tolist(),
        "intercept": data.params["intercept"].tolist(),
        "alpha": float(data.params["alpha"].item()),
    }
    return diagnostics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--train-seed", type=int, default=110_000)
    parser.add_argument("--test-seed", type=int, default=120_000)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--dgp-seed", type=int, default=42)
    parser.add_argument(
        "--beta-values",
        nargs=K,
        type=float,
        default=list(case1_v3.DEFAULT_BETA_VALUES),
    )
    parser.add_argument(
        "--intercepts",
        nargs=K,
        type=float,
        default=list(case1_v3.DEFAULT_INTERCEPTS),
    )
    parser.add_argument("--beta-flat", nargs=K * P, type=float)
    parser.add_argument("--alpha", type=float, default=1.15)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS))
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prediction-dir", type=Path)
    return parser.parse_args()



def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    config = SoftCompConfig(
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
    )
    parameters = case1_v3.generate_parameters(
        K=K,
        p=P,
        seed=args.dgp_seed,
        beta_values=args.beta_flat if args.beta_flat is not None else args.beta_values,
        intercepts=args.intercepts,
        alpha=args.alpha,
    )
    data = prepare_data(
        n_train=args.n_train,
        n_test=args.n_test,
        train_seed=args.train_seed,
        test_seed=args.test_seed,
        dgp_seed=args.dgp_seed,
        parameters=parameters,
    )
    diagnostics = data_diagnostics(data)
    logger.info("DGP diagnostics: %s", json.dumps(diagnostics, sort_keys=True))
    specs = _select_specs(build_specs(data, config, args.model_seed), args.models)
    results = _run_specs(specs, data, args.model_seed, args.prediction_dir)
    payload = {
        "protocol": {
            "case": "case1_v3",
            "n_train": args.n_train,
            "n_test": args.n_test,
            "train_seed": args.train_seed,
            "test_seed": args.test_seed,
            "model_seed": args.model_seed,
            "dgp_seed": args.dgp_seed,
            "dgp_parameters": {
                "beta": parameters["beta"].tolist(),
                "intercept": parameters["intercept"].tolist(),
                "alpha": float(parameters["alpha"].item()),
            },
            "cpu_threads": args.cpu_threads,
            "models": args.models,
            "evaluation_percentile_cap": EVALUATION_PERCENTILE_CAP,
            "critical_evaluation_times": CRITICAL_EVALUATION_TIMES,
            "softcomp": asdict(config),
            "softcomp_postprocessing": (
                "PAV followed by global-time simplex rescaling"
            ),
        },
        "diagnostics": diagnostics,
        "results": results,
    }
    output = args.output or (get_output_dir("case1_v3") / "single_run.json")
    _write_json(output, payload)
    logger.info("Saved %s", output)


if __name__ == "__main__":
    main()

