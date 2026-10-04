#!/usr/bin/env python3
"""Classical Fine-Gray proportional subdistribution hazards regression.

The implementation fits one model per cause with the Fine-Gray weighted
partial likelihood. Subjects with an earlier competing event remain in the
risk set with weight ``G(t-) / G(T_i-)``, where ``G`` is the Kaplan-Meier
estimate of the censoring survival function.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CauseModel:
    coefficients: np.ndarray
    event_times: np.ndarray
    baseline_cumulative_hazard: np.ndarray
    converged: bool
    iterations: int
    score_norm: float


def _training_arrays(
    X_train: Tensor,
    Y_train: Tensor,
    Delta_train: Tensor,
    n_causes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(X_train.detach().cpu(), dtype=np.float64)
    times = np.asarray(Y_train.detach().cpu(), dtype=np.float64)
    events = np.asarray(Delta_train.detach().cpu(), dtype=np.int64)
    if x.ndim != 2 or times.ndim != 1 or events.ndim != 1:
        raise ValueError("X must be 2-D and Y/Delta must be 1-D")
    if len(x) != len(times) or len(times) != len(events):
        raise ValueError("X, Y, and Delta must contain the same number of rows")
    if len(x) == 0 or x.shape[1] == 0:
        raise ValueError("Fine-Gray requires at least one row and one feature")
    if not np.isfinite(x).all() or not np.isfinite(times).all():
        raise ValueError("X and Y must contain only finite values")
    if (times < 0).any():
        raise ValueError("observed times must be non-negative")
    if ((events < 0) | (events > n_causes)).any():
        raise ValueError(f"Delta values must be in 0..{n_causes}")
    return x, times, events


def _censoring_survival_before(
    times: np.ndarray,
    events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support, inverse = np.unique(times, return_inverse=True)
    survival_before = np.empty(len(support), dtype=np.float64)
    survival = 1.0
    for index, time_value in enumerate(support):
        survival_before[index] = survival
        censored = np.count_nonzero((times == time_value) & (events == 0))
        if censored == 0:
            continue
        # Match cmprsk/comprisk: every observation tied at this time remains
        # in the censoring KM risk set before the censoring jump is applied.
        at_risk = np.count_nonzero(times >= time_value)
        if at_risk < censored:
            raise RuntimeError("invalid censoring risk set")
        survival *= 1.0 - censored / at_risk
    return support, survival_before, survival_before[inverse]


def _risk_weights(
    event_times: np.ndarray,
    observed_times: np.ndarray,
    events: np.ndarray,
    cause: int,
) -> np.ndarray:
    support, survival_before, subject_survival = _censoring_survival_before(
        observed_times,
        events,
    )
    event_indices = np.searchsorted(support, event_times)
    event_survival = survival_before[event_indices]
    weights = (observed_times[None, :] >= event_times[:, None]).astype(np.float64)
    competing = (
        (observed_times[None, :] < event_times[:, None])
        & (events[None, :] != 0)
        & (events[None, :] != cause)
    )
    if np.any(competing & (subject_survival[None, :] <= 0)):
        raise ValueError("censoring survival reached zero before a competing event")
    ratios = np.divide(
        event_survival[:, None],
        subject_survival[None, :],
        out=np.zeros_like(weights),
        where=subject_survival[None, :] > 0,
    )
    weights[competing] = ratios[competing]
    return weights


def _event_summaries(
    design: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    cause: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    event_mask = events == cause
    if not event_mask.any():
        raise ValueError(f"cause {cause} has no observed events")
    event_times, inverse = np.unique(times[event_mask], return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    design_sums = np.zeros((len(event_times), design.shape[1]), dtype=np.float64)
    np.add.at(design_sums, inverse, design[event_mask])
    return event_times, counts, design_sums


def _weighted_risk_terms(
    coefficients: np.ndarray,
    design: np.ndarray,
    risk_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    linear_predictor = design @ coefficients
    offset = float(linear_predictor.max(initial=0.0))
    relative_risk = np.exp(linear_predictor - offset)
    weighted_risk = risk_weights * relative_risk[None, :]
    denominators = weighted_risk.sum(axis=1)
    if (denominators <= 0).any():
        raise RuntimeError("Fine-Gray risk set has zero total weight")
    return weighted_risk, np.log(denominators) + offset


def _objective(
    coefficients: np.ndarray,
    design: np.ndarray,
    risk_weights: np.ndarray,
    event_counts: np.ndarray,
    event_design_sums: np.ndarray,
    penalizer: float,
) -> float:
    _, log_denominators = _weighted_risk_terms(
        coefficients,
        design,
        risk_weights,
    )
    event_term = float((event_design_sums @ coefficients).sum())
    penalty = 0.5 * penalizer * float(coefficients @ coefficients)
    return event_term - float(event_counts @ log_denominators) - penalty


def _score_information(
    coefficients: np.ndarray,
    design: np.ndarray,
    risk_weights: np.ndarray,
    event_counts: np.ndarray,
    event_design_sums: np.ndarray,
    penalizer: float,
) -> tuple[np.ndarray, np.ndarray]:
    weighted_risk, _ = _weighted_risk_terms(coefficients, design, risk_weights)
    denominators = weighted_risk.sum(axis=1)
    means = weighted_risk @ design / denominators[:, None]
    second_moments = (
        np.einsum(
            "mn,ni,nj->mij",
            weighted_risk,
            design,
            design,
            optimize=True,
        )
        / denominators[:, None, None]
    )
    covariance = second_moments - np.einsum("mi,mj->mij", means, means)
    score = (event_design_sums - event_counts[:, None] * means).sum(axis=0)
    score -= penalizer * coefficients
    information = np.einsum("m,mij->ij", event_counts, covariance)
    information += penalizer * np.eye(design.shape[1])
    return score, information


def _newton_step(information: np.ndarray, score: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(information, score)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(information, score, rcond=None)[0]


def _backtracking_update(
    coefficients: np.ndarray,
    step: np.ndarray,
    objective: float,
    objective_fn: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, float, float]:
    scale = 1.0
    for _ in range(25):
        candidate = coefficients + scale * step
        candidate_objective = objective_fn(candidate)
        if np.isfinite(candidate_objective) and candidate_objective >= objective:
            return candidate, float(candidate_objective), scale
        scale *= 0.5
    return coefficients, objective, 0.0


def _optimize_coefficients(
    design: np.ndarray,
    risk_weights: np.ndarray,
    event_counts: np.ndarray,
    event_design_sums: np.ndarray,
    penalizer: float,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, bool, int]:
    def objective_fn(value: np.ndarray) -> float:
        return _objective(
            value,
            design,
            risk_weights,
            event_counts,
            event_design_sums,
            penalizer,
        )

    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    objective = objective_fn(coefficients)
    converged = False
    iterations = 0
    for iteration in range(1, max_iter + 1):
        score, information = _score_information(
            coefficients,
            design,
            risk_weights,
            event_counts,
            event_design_sums,
            penalizer,
        )
        if np.linalg.norm(score, ord=np.inf) <= tolerance:
            converged = True
            break
        step = _newton_step(information, score)
        iterations = iteration
        step_tolerance = tolerance * (1.0 + np.linalg.norm(coefficients))
        if np.linalg.norm(step) <= step_tolerance:
            coefficients = coefficients + step
            converged = True
            break
        coefficients, objective, scale = _backtracking_update(
            coefficients, step, objective, objective_fn
        )
        if scale == 0.0:
            break
    return coefficients, converged, iterations


def _fit_cause(
    design: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    cause: int,
    penalizer: float,
    max_iter: int,
    tolerance: float,
) -> _CauseModel:
    event_times, event_counts, event_design_sums = _event_summaries(
        design, times, events, cause
    )
    risk_weights = _risk_weights(event_times, times, events, cause)
    coefficients, converged, iterations = _optimize_coefficients(
        design,
        risk_weights,
        event_counts,
        event_design_sums,
        penalizer,
        max_iter,
        tolerance,
    )
    score, _ = _score_information(
        coefficients,
        design,
        risk_weights,
        event_counts,
        event_design_sums,
        penalizer,
    )
    _, log_denominators = _weighted_risk_terms(coefficients, design, risk_weights)
    increments = event_counts * np.exp(-log_denominators)
    return _CauseModel(
        coefficients=coefficients,
        event_times=event_times,
        baseline_cumulative_hazard=np.cumsum(increments),
        converged=converged,
        iterations=iterations,
        score_norm=float(np.linalg.norm(score, ord=np.inf)),
    )


class FineGray:
    """One classical Fine-Gray model per competing event cause.

    Args:
        n_causes: number of competing event types.
        penalizer: L2 penalty applied to each cause-specific coefficient vector.
        max_iter: maximum Newton iterations per cause.
        tolerance: convergence tolerance for score and coefficient updates.
    """

    def __init__(
        self,
        n_causes: int,
        penalizer: float = 0.0,
        max_iter: int = 100,
        tolerance: float = 1e-7,
    ) -> None:
        if n_causes <= 0:
            raise ValueError("n_causes must be positive")
        if not np.isfinite(penalizer) or penalizer < 0:
            raise ValueError("penalizer must be non-negative")
        if max_iter <= 0 or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("max_iter and tolerance must be positive")
        self.n_causes = n_causes
        self.penalizer = penalizer
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.models: list[_CauseModel] = []
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None

    @classmethod
    def load_from_checkpoint(cls, data: dict) -> FineGray:
        model = cls(
            n_causes=data["n_causes"],
            penalizer=data.get("penalizer", 0.0),
            max_iter=data.get("max_iter", 100),
            tolerance=data.get("tolerance", 1e-7),
        )
        model.feature_mean = np.asarray(data["feature_mean"], dtype=np.float64)
        model.feature_scale = np.asarray(data["feature_scale"], dtype=np.float64)
        model.models = [
            _CauseModel(
                coefficients=np.asarray(item["coefficients"], dtype=np.float64),
                event_times=np.asarray(item["event_times"], dtype=np.float64),
                baseline_cumulative_hazard=np.asarray(
                    item["baseline_cumulative_hazard"], dtype=np.float64
                ),
                converged=bool(item["converged"]),
                iterations=int(item["iterations"]),
                score_norm=float(item.get("score_norm", np.nan)),
            )
            for item in data["cause_models"]
        ]
        return model

    def to_checkpoint(self) -> dict[str, object]:
        """Return a dependency-free checkpoint containing fitted arrays."""
        self._require_fitted()
        return {
            "n_causes": self.n_causes,
            "penalizer": self.penalizer,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "cause_models": [
                {
                    "coefficients": cause_model.coefficients,
                    "event_times": cause_model.event_times,
                    "baseline_cumulative_hazard": (
                        cause_model.baseline_cumulative_hazard
                    ),
                    "converged": cause_model.converged,
                    "iterations": cause_model.iterations,
                    "score_norm": cause_model.score_norm,
                }
                for cause_model in self.models
            ],
        }

    def fit(
        self,
        X_train: Tensor,
        Y_train: Tensor,
        Delta_train: Tensor,
    ) -> None:
        """Fit a proportional subdistribution-hazards model for each cause."""
        x, times, events = _training_arrays(
            X_train, Y_train, Delta_train, self.n_causes
        )
        self.feature_mean = x.mean(axis=0)
        self.feature_scale = x.std(axis=0)
        self.feature_scale[self.feature_scale < 1e-12] = 1.0
        design = (x - self.feature_mean) / self.feature_scale
        self.models = []
        for cause in range(1, self.n_causes + 1):
            cause_model = _fit_cause(
                design,
                times,
                events,
                cause,
                self.penalizer,
                self.max_iter,
                self.tolerance,
            )
            self.models.append(cause_model)
            event_count = int((events == cause).sum())
            if not cause_model.converged:
                raise RuntimeError(
                    f"Fine-Gray failed to converge for cause {cause} after "
                    f"{cause_model.iterations} iterations"
                )
            logger.info(
                "Fine-Gray cause %d fitted (%d events, iterations=%d, "
                "score_norm=%.3e, converged=%s)",
                cause,
                event_count,
                cause_model.iterations,
                cause_model.score_norm,
                cause_model.converged,
            )

    def predict_cif(self, X_test: Tensor, times: Tensor) -> Tensor:
        """Predict cause-specific cumulative incidence functions."""
        self._require_fitted()
        assert self.feature_mean is not None
        assert self.feature_scale is not None
        x = np.asarray(X_test.detach().cpu(), dtype=np.float64)
        evaluation_times = np.asarray(times.detach().cpu(), dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.feature_mean):
            raise ValueError("X_test has the wrong feature dimension")
        if not np.isfinite(x).all():
            raise ValueError("X_test must contain only finite values")
        if evaluation_times.ndim != 1 or not np.isfinite(evaluation_times).all():
            raise ValueError("times must be a finite one-dimensional tensor")
        design = (x - self.feature_mean) / self.feature_scale
        cif = np.zeros((len(x), self.n_causes, len(evaluation_times)))
        for cause_index, cause_model in enumerate(self.models):
            indices = (
                np.searchsorted(cause_model.event_times, evaluation_times, side="right")
                - 1
            )
            baseline = np.zeros(len(evaluation_times), dtype=np.float64)
            present = indices >= 0
            baseline[present] = cause_model.baseline_cumulative_hazard[indices[present]]
            hazard_ratio = np.exp(
                np.clip(design @ cause_model.coefficients, -50.0, 50.0)
            )
            cif[:, cause_index, :] = -np.expm1(
                -hazard_ratio[:, None] * baseline[None, :]
            )
        return torch.tensor(cif, dtype=torch.float32)

    def predict_cif_survival(
        self,
        X_test: Tensor,
        times: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return CIFs and their implied, potentially negative, survival.

        Separate Fine-Gray regressions do not define a joint native overall
        survival model. Returning ``1 - sum_k CIF_k`` preserves that limitation
        and allows the existing negative-survival diagnostics to expose it.
        """
        cif = self.predict_cif(X_test, times)
        return cif, 1.0 - cif.sum(dim=1)

    def _require_fitted(self) -> None:
        if (
            len(self.models) != self.n_causes
            or self.feature_mean is None
            or self.feature_scale is None
        ):
            raise RuntimeError("FineGray is not fitted")
