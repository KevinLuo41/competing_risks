#!/usr/bin/env python3
"""Case II v3 data generation with strong nonlinear covariate effects.

For ``r_k(x) = exp(a_k(x))``, where

    a_k(x) = intercept_k + beta_k^T x / sqrt(p)
             + quadratic_weight tanh(x_k^2 - 1)
             + pairwise_weight tanh(x_k x_{k+1}),

and

    G(t) = exp((t / time_scale)^3) - 1,

the true CIF is

    F_k(t | x) = r_k(x) G(t) / (1 + G(t) sum_j r_j(x)).

The covariate indices are taken cyclically. The bounded quadratic and pairwise
terms provide strong nonlinear covariate effects without introducing any
covariate-by-time interaction. Under independent centered Gaussian covariates,
both nonlinear terms are linearly orthogonal to the original covariates. The
nonlinear cumulative-odds time function satisfies G(0)=0 and is strictly
increasing for t>0. Consequently,

    d F_k(t | x) / dt = r_k(x) G'(t) / (1 + G(t) sum_j r_j(x))^2 >= 0.
"""

from __future__ import annotations

import functools
import math

import torch
from torch import Tensor

from .utils import assert_monotone_cif, assign_causes_and_censor, solve_inverse_cdf

T_MAX = 60.0


def _log_cumulative_odds(t: Tensor, time_scale: Tensor) -> Tensor:
    scaled_cubic = (t.clamp(min=0.0) / time_scale).pow(3)
    log_expm1 = torch.log(torch.expm1(scaled_cubic.clamp(max=20.0)))
    return torch.where(scaled_cubic > 20.0, scaled_cubic, log_expm1)


def _nonlinear_score(x: Tensor, K: int, params: dict[str, Tensor]) -> Tensor:
    cause_index = torch.arange(K, device=x.device) % x.shape[1]
    next_index = (cause_index + 1) % x.shape[1]
    primary = x[:, cause_index]
    adjacent = x[:, next_index]
    return params["quadratic_weight"] * torch.tanh(primary.square() - 1.0) + params[
        "pairwise_weight"
    ] * torch.tanh(primary * adjacent)


def compute_static_logits(x: Tensor, params: dict[str, Tensor]) -> Tensor:
    """Compute the time-independent cause scores for Case II."""
    K = params["intercept"].shape[0]
    return (
        params["intercept"].unsqueeze(0)
        + x @ params["beta"].T / math.sqrt(x.shape[1])
        + _nonlinear_score(x, K, params)
    )


def compute_cif(
    x: Tensor,
    t: Tensor,
    params: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """Compute the Case II v3 CIF and survival probabilities."""
    if t.dim() == 0:
        t = t.expand(x.shape[0])
    static_logits = compute_static_logits(x, params)
    cause_probability = torch.softmax(static_logits, dim=1)
    log_total_odds = torch.logsumexp(static_logits, dim=1) + _log_cumulative_odds(
        t, params["time_scale"]
    )
    total_event_probability = torch.sigmoid(log_total_odds)
    cif = cause_probability * total_event_probability.unsqueeze(1)
    return cif, 1.0 - total_event_probability


def generate_parameters(
    K: int = 8,
    p: int = 8,
    seed: int = 42,
    beta_min: float = 0.1,
    beta_max: float = 0.3,
    quadratic_weight: float = 1.5,
    pairwise_weight: float = 1.0,
    time_scale: float = 12.0,
) -> dict[str, Tensor]:
    """Generate fixed parameters dominated by nonlinear covariate effects."""
    if not 0 < beta_min < beta_max:
        raise ValueError("beta bounds must satisfy 0 < beta_min < beta_max")
    if quadratic_weight <= 0:
        raise ValueError("quadratic_weight must be positive")
    if pairwise_weight <= 0:
        raise ValueError("pairwise_weight must be positive")
    if time_scale <= 0:
        raise ValueError("time_scale must be positive")
    generator = torch.Generator().manual_seed(seed)
    beta = beta_min + (beta_max - beta_min) * torch.rand(K, p, generator=generator)
    return {
        "beta": beta,
        "intercept": torch.linspace(-5.1, -4.9, K),
        "quadratic_weight": torch.tensor(quadratic_weight),
        "pairwise_weight": torch.tensor(pairwise_weight),
        "time_scale": torch.tensor(time_scale),
    }


def generate_data(
    n: int = 5000,
    K: int = 8,
    p: int = 8,
    censor_rate: float = 0.5,
    seed: int = 42,
    params: dict[str, Tensor] | None = None,
) -> dict[str, Tensor | dict[str, Tensor]]:
    """Generate one Case II v3 subject sample."""
    torch.manual_seed(seed)
    if params is None:
        params = generate_parameters(K=K, p=p, seed=seed)
    beta = params["beta"]
    intercept = params["intercept"]
    time_scale = params["time_scale"]
    if beta.shape != (K, p):
        raise ValueError(f"expected beta shape {(K, p)}, got {tuple(beta.shape)}")
    if intercept.shape != (K,):
        raise ValueError(
            f"expected intercept shape {(K,)}, got {tuple(intercept.shape)}"
        )
    if time_scale.numel() != 1 or float(time_scale.item()) <= 0:
        raise ValueError("time_scale must be a positive scalar")

    x = torch.randn(n, p)
    cif_fn = functools.partial(compute_cif, params=params)
    assert_monotone_cif(cif_fn, x[:50], torch.linspace(0.0, T_MAX, 300))
    t_event = solve_inverse_cdf(x, torch.rand(n), cif_fn, t_max=T_MAX)
    y, delta, epsilon = assign_causes_and_censor(x, t_event, cif_fn, censor_rate)
    return {
        "X": x,
        "Y": y,
        "Delta": delta,
        "T_true": t_event,
        "epsilon_true": epsilon,
        "params": params,
    }
