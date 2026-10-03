#!/usr/bin/env python3
"""Case III v4 data generation with nonlinear time-covariate interaction.

The static cause score is linear:

    eta_k(x) = intercept_k + beta_k^T x / sqrt(3).

All strong nonlinearity is isolated in a folded single-index time rate:

    z(x) = (x_1 - x_2 + x_3) / sqrt(3),
    q(x) = min(|z(x)|, 2),
    g(x) = 0.12 + 0.14 q(x),
    G(t, x) = exp(t g(x)) - 1,

and defines

    F_k(t | x) = exp(eta_k(x)) G(t, x)
                 / (1 + G(t, x) sum_j exp(eta_j(x))),
    S(t | x) = 1 / (1 + G(t, x) sum_j exp(eta_j(x))).

The rate satisfies ``0.12 <= g(x) <= 0.40``. If ``r_k = exp(eta_k)`` and
``R = sum_k r_k``, then for ``t >= 0``

    d F_k(t | x) / dt
        = r_k g(x) exp(t g(x)) / (1 + R G(t, x))^2 > 0.

Thus every true CIF is monotone, ``F_k(0 | x) = 0``, ``S(0 | x) = 1``, and
the probabilities sum to one. The folded index is even in ``z`` and therefore
linearly orthogonal to the raw centered Gaussian covariates, while remaining a
piecewise-linear function that a ReLU network can represent efficiently.
"""

from __future__ import annotations

import functools
import math

import torch
from torch import Tensor

from .utils import assert_monotone_cif, assign_causes_and_censor, solve_inverse_cdf

T_MAX = 120.0


def _log_expm1(value: Tensor) -> Tensor:
    """Return ``log(expm1(value))`` without overflowing for large values."""
    direct = torch.log(torch.expm1(value.clamp(max=20.0)))
    return torch.where(value > 20.0, value, direct)


def compute_static_logits(x: Tensor, params: dict[str, Tensor]) -> Tensor:
    """Compute the linear, time-independent cause scores."""
    return params["intercept"].unsqueeze(0) + x @ params["beta"].T / math.sqrt(3.0)


def compute_time_rate(x: Tensor, params: dict[str, Tensor]) -> Tensor:
    """Compute the bounded folded single-index time rate ``g(x)``."""
    projection = x @ params["time_direction"]
    folded_projection = projection.abs().clamp(max=params["projection_cap"])
    return params["rate_floor"] + params["rate_weight"] * folded_projection


def compute_cif(
    x: Tensor,
    t: Tensor,
    params: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """Compute the Case III v4 CIF and survival probabilities."""
    if t.dim() == 0:
        t = t.expand(x.shape[0])
    static_logits = compute_static_logits(x, params)
    cause_probability = torch.softmax(static_logits, dim=1)
    time_rate = compute_time_rate(x, params)
    cumulative_odds_argument = t.clamp(min=0.0) * time_rate
    log_total_odds = torch.logsumexp(static_logits, dim=1) + _log_expm1(
        cumulative_odds_argument
    )
    total_event_probability = torch.sigmoid(log_total_odds)
    cif = cause_probability * total_event_probability.unsqueeze(1)
    return cif, 1.0 - total_event_probability


def generate_parameters(
    K: int = 3,
    p: int = 3,
    seed: int = 42,
    beta_min: float = 0.05,
    beta_max: float = 0.15,
    rate_floor: float = 0.12,
    rate_weight: float = 0.14,
    projection_cap: float = 2.0,
) -> dict[str, Tensor]:
    """Generate fixed parameters for the Case III v4 truth."""
    if (K, p) != (3, 3):
        raise ValueError("Case III v4 requires K=p=3")
    if not 0 < beta_min < beta_max:
        raise ValueError("beta bounds must satisfy 0 < beta_min < beta_max")
    if rate_floor <= 0:
        raise ValueError("rate_floor must be positive")
    if rate_weight <= 0:
        raise ValueError("rate_weight must be positive")
    if projection_cap <= 0:
        raise ValueError("projection_cap must be positive")

    generator = torch.Generator().manual_seed(seed)
    beta = beta_min + (beta_max - beta_min) * torch.rand(K, p, generator=generator)
    return {
        "beta": beta,
        "intercept": torch.linspace(-4.1, -3.9, K),
        "time_direction": torch.tensor([1.0, -1.0, 1.0]) / math.sqrt(3.0),
        "projection_cap": torch.tensor(projection_cap),
        "rate_floor": torch.tensor(rate_floor),
        "rate_weight": torch.tensor(rate_weight),
    }


def generate_data(
    n: int = 5000,
    K: int = 3,
    p: int = 3,
    censor_rate: float = 0.5,
    seed: int = 42,
    params: dict[str, Tensor] | None = None,
) -> dict[str, Tensor | dict[str, Tensor]]:
    """Generate one Case III v4 subject sample."""
    torch.manual_seed(seed)
    if params is None:
        params = generate_parameters(K=K, p=p, seed=seed)
    beta = params["beta"]
    intercept = params["intercept"]
    if beta.shape != (K, p):
        raise ValueError(f"expected beta shape {(K, p)}, got {tuple(beta.shape)}")
    if intercept.shape != (K,):
        raise ValueError(
            f"expected intercept shape {(K,)}, got {tuple(intercept.shape)}"
        )

    x = torch.randn(n, p)
    cif_fn = functools.partial(compute_cif, params=params)
    assert_monotone_cif(cif_fn, x[:32], torch.linspace(0.0, T_MAX, 121))
    t_event = solve_inverse_cdf(x, torch.rand(n), cif_fn, t_max=T_MAX)
    y, delta, epsilon = assign_causes_and_censor(x, t_event, cif_fn, censor_rate)
    return {
        "X": x,
        "Y": y,
        "Delta": delta,
        "T_true": t_event,
        "epsilon_true": epsilon,
        "time_rate": compute_time_rate(x, params),
        "params": params,
    }
