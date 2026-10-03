#!/usr/bin/env python3
"""Case I v3 data generation with a strictly affine log-odds model.

Let ``X ~ N_5(0, I_5)`` and, for nonnegative time, define

    mu_k(x, t) = intercept_k + beta_k^T x + alpha t.

The true probabilities follow the competing-risks softmax link

    F_k(t | x) = exp(mu_k(x, t)) / [1 + sum_j exp(mu_j(x, t))],
    S(t | x) = 1 / [1 + sum_j exp(mu_j(x, t))].

The shared positive time slope makes every cause-specific CIF monotone.  A
finite affine logit necessarily gives a positive event mass at time zero.  We
treat that mass as an explicit, legal atom rather than silently approximating
``F_k(0 | x)`` by zero.  For negative times the CDF is zero and survival is
one, so the distribution is right-continuous at the atom.

Write ``eta_k(x) = intercept_k + beta_k^T x`` and
``A(x) = sum_k exp(eta_k(x))``.  The total event CDF is

    F(t | x) = A(x) exp(alpha t) / [1 + A(x) exp(alpha t)],  t >= 0.

Consequently, event times have the analytic generalized inverse

    T = 0,                                           U <= F(0 | x),
    T = [logit(U) - log A(x)] / alpha,                U > F(0 | x).

The frozen coefficient matrix uses only the first covariate,

    beta_1 = (-0.70, 0, 0, 0, 0),
    beta_2 = ( 0.00, 0, 0, 0, 0),
    beta_3 = ( 0.70, 0, 0, 0, 0).

Thus the second cause has a constant covariate score.  ``generate_parameters``
also accepts either three first-coordinate coefficients or a complete flat
``K * p`` coefficient matrix for development-only command-line probes.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence

import torch
from torch import Tensor

from .utils import assert_monotone_cif

K = 3
P = 5
T_MAX = 60.0

DEFAULT_INTERCEPTS = (-16.0, -15.78, -16.0)
DEFAULT_BETA_VALUES = (-0.7, 0.0, 0.7)


def _expand_time(t: Tensor, n_subjects: int) -> Tensor:
    if t.dim() == 0:
        return t.expand(n_subjects)
    if t.dim() != 1 or len(t) != n_subjects:
        raise ValueError("t must be scalar or contain one value per subject")
    return t


def compute_static_logits(x: Tensor, params: dict[str, Tensor]) -> Tensor:
    """Return the affine cause scores ``intercept + x @ beta.T``."""
    return params["intercept"].unsqueeze(0) + x @ params["beta"].T


def compute_cause_probabilities(x: Tensor, params: dict[str, Tensor]) -> Tensor:
    """Return the time-invariant conditional probabilities of each cause."""
    return torch.softmax(compute_static_logits(x, params), dim=1)


def compute_initial_event_probability(
    x: Tensor,
    params: dict[str, Tensor],
) -> Tensor:
    """Return the total point mass ``sum_k F_k(0 | x)``."""
    log_total_weight = torch.logsumexp(compute_static_logits(x, params), dim=1)
    return torch.sigmoid(log_total_weight)


def compute_cif(
    x: Tensor,
    t: Tensor,
    params: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """Compute Case I v3 CIF and survival, including the time-zero atom."""
    expanded_time = _expand_time(t, x.shape[0])
    nonnegative_time = expanded_time.clamp(min=0.0)
    static_logits = compute_static_logits(x, params)
    cause_probabilities = torch.softmax(static_logits, dim=1)
    log_total_odds = torch.logsumexp(static_logits, dim=1) + (
        nonnegative_time * params["alpha"]
    )
    total_event_probability = torch.sigmoid(log_total_odds)
    cif = cause_probabilities * total_event_probability.unsqueeze(1)
    survival = 1.0 - total_event_probability
    before_support = expanded_time < 0.0
    cif = torch.where(before_support.unsqueeze(1), torch.zeros_like(cif), cif)
    survival = torch.where(before_support, torch.ones_like(survival), survival)
    return cif, survival


def sample_event_times(
    x: Tensor,
    uniform: Tensor,
    params: dict[str, Tensor],
) -> Tensor:
    """Sample event times with the analytic generalized inverse CDF."""
    if uniform.shape != (x.shape[0],):
        raise ValueError("uniform must contain one value per subject")
    if bool(((uniform <= 0.0) | (uniform >= 1.0)).any().item()):
        raise ValueError("uniform values must lie in (0, 1)")
    log_total_weight = torch.logsumexp(compute_static_logits(x, params), dim=1)
    initial_probability = torch.sigmoid(log_total_weight)
    positive_time = (torch.logit(uniform) - log_total_weight) / params["alpha"]
    return torch.where(
        uniform <= initial_probability,
        torch.zeros_like(positive_time),
        positive_time,
    )


def _as_finite_tensor(
    values: Sequence[float],
    expected_length: int,
    name: str,
) -> Tensor:
    if len(values) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} values")
    tensor = torch.tensor(tuple(values), dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def generate_parameters(
    K: int = K,
    p: int = P,
    seed: int = 42,
    beta_values: Sequence[float] = DEFAULT_BETA_VALUES,
    intercepts: Sequence[float] = DEFAULT_INTERCEPTS,
    alpha: float = 1.15,
) -> dict[str, Tensor]:
    """Return the frozen Case I v3 affine parameters.

    ``beta_values`` may contain one first-coordinate coefficient per cause or
    all ``K * p`` matrix entries in row-major order.  ``seed`` is accepted for
    protocol compatibility; no coefficient is selected by seed.
    """
    if (K, p) != (3, 5):
        raise ValueError("Case I v3 requires K=3 and p=5")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    beta_tensor = _as_finite_tensor(beta_values, len(beta_values), "beta_values")
    if len(beta_tensor) == K:
        beta = torch.zeros(K, p)
        beta[:, 0] = beta_tensor
    elif len(beta_tensor) == K * p:
        beta = beta_tensor.reshape(K, p)
    else:
        raise ValueError(f"beta_values must contain {K} or {K * p} values")
    intercept_tensor = _as_finite_tensor(intercepts, K, "intercepts")

    del seed
    return {
        "beta": beta,
        "intercept": intercept_tensor,
        "alpha": torch.tensor(alpha),
    }


def _validate_parameters(
    params: dict[str, Tensor],
    K: int,
    p: int,
) -> None:
    beta = params["beta"]
    intercept = params["intercept"]
    alpha = params["alpha"]
    if beta.shape != (K, p):
        raise ValueError(f"expected beta shape {(K, p)}, got {tuple(beta.shape)}")
    if intercept.shape != (K,):
        raise ValueError(
            f"expected intercept shape {(K,)}, got {tuple(intercept.shape)}"
        )
    if alpha.numel() != 1:
        raise ValueError("alpha must be a scalar")
    if not bool(torch.isfinite(beta).all().item()):
        raise ValueError("beta must contain only finite values")
    if not bool(torch.isfinite(intercept).all().item()):
        raise ValueError("intercept must contain only finite values")
    if not bool(torch.isfinite(alpha).all().item()) or float(alpha.item()) <= 0.0:
        raise ValueError("alpha must be finite and positive")


def _apply_censoring(
    t_event: Tensor,
    event_cause: Tensor,
    censor_rate: float,
) -> tuple[Tensor, Tensor]:
    median_time = float(t_event.median().item())
    censor_scale = median_time / (-math.log1p(-censor_rate))
    censor_time = torch.distributions.Exponential(1.0 / censor_scale).sample(
        (len(t_event),)
    )
    observed_time = torch.minimum(t_event, censor_time)
    observed_event = torch.where(
        t_event <= censor_time,
        event_cause,
        torch.zeros_like(event_cause),
    )
    return observed_time, observed_event


def generate_data(
    n: int = 5000,
    K: int = K,
    p: int = P,
    censor_rate: float = 0.5,
    seed: int = 42,
    params: dict[str, Tensor] | None = None,
) -> dict[str, Tensor | dict[str, Tensor]]:
    """Generate one Case I v3 sample, including explicit atom diagnostics."""
    if n <= 0:
        raise ValueError("n must be positive")
    if (K, p) != (3, 5):
        raise ValueError("Case I v3 requires K=3 and p=5")
    if not math.isfinite(censor_rate) or not 0.0 < censor_rate < 1.0:
        raise ValueError("censor_rate must lie in (0, 1)")
    torch.manual_seed(seed)
    if params is None:
        params = generate_parameters(K=K, p=p, seed=seed)
    _validate_parameters(params, K, p)

    x = torch.randn(n, p)
    assert_monotone_cif(
        functools.partial(compute_cif, params=params),
        x[:32],
        torch.linspace(0.0, T_MAX, 161),
    )
    epsilon = torch.finfo(x.dtype).eps
    uniform = torch.rand(n).clamp(min=epsilon, max=1.0 - epsilon)
    t_event = sample_event_times(x, uniform, params)
    cause_probabilities = compute_cause_probabilities(x, params)
    event_cause = torch.multinomial(cause_probabilities, num_samples=1).squeeze(1) + 1
    y, delta = _apply_censoring(t_event, event_cause, censor_rate)
    initial_event_probability = compute_initial_event_probability(x, params)
    return {
        "X": x,
        "Y": y,
        "Delta": delta,
        "T_true": t_event,
        "epsilon_true": event_cause,
        "cause_probabilities": cause_probabilities,
        "linear_predictors": compute_static_logits(x, params),
        "initial_event_probability": initial_event_probability,
        "is_time_zero_atom": t_event == 0.0,
        "params": params,
    }
