#!/usr/bin/env python3
"""Case III data generation with nonlinear time-covariate interaction.

The event logits are

    mu_k(x, t) = a_k(x) + t * g(x),
    a_k(x) = intercept_k + W_out_k @ tanh(W_shared @ x),
    g(x) = slope_min + slope_range * sigmoid(gamma^T x / sqrt(p)).

The term ``t * g(x)`` is an explicit time-covariate interaction.  The same
strictly positive slope ``g(x)`` is shared by all causes.  If
``q_k = exp(a_k(x) + t * g(x))`` and ``D = 1 + sum_k q_k``, then

    d F_k(t | x) / dt = g(x) * q_k / D^2 > 0.

Thus the interaction changes the subject-specific time scale without
violating monotonicity of any true cause-specific CIF.
"""

from __future__ import annotations

import functools
import math

import torch
from torch import Tensor

from .utils import (
    assert_monotone_cif,
    assign_causes_and_censor,
    mu_to_cif,
    solve_inverse_cdf,
)

T_MAX = 120.0


def compute_time_slope(x: Tensor, params: dict[str, Tensor]) -> Tensor:
    """Return the positive subject-specific time slope ``g(x)``."""
    p = x.shape[1]
    score = x @ params["gamma"] / math.sqrt(p)
    return params["slope_min"] + params["slope_range"] * torch.sigmoid(score)


def compute_cif(
    x: Tensor,
    t: Tensor,
    params: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """Compute the monotone Case III CIF and survival probabilities."""
    if t.dim() == 0:
        t = t.expand(x.shape[0])

    hidden = torch.tanh(x @ params["W_shared"].T)
    static_logits = params["intercept"].unsqueeze(0) + hidden @ params["W_out"].T
    time_slope = compute_time_slope(x, params)
    logits = static_logits + t.unsqueeze(1) * time_slope.unsqueeze(1)
    return mu_to_cif(logits)


def generate_parameters(
    K: int = 3,
    p: int = 5,
    h: int = 12,
    seed: int = 42,
    mean_time_slope: float = 0.4,
    interaction_range: float = 0.6,
    gamma_scale: float = 1.0,
) -> dict[str, Tensor]:
    """Generate fixed nonlinear and time-interaction parameters.

    ``interaction_range`` controls heterogeneity in ``g(x)`` while preserving
    its mean around ``mean_time_slope`` for symmetric covariates.  Requiring
    ``interaction_range < 2 * mean_time_slope`` keeps the slope strictly
    positive for every possible covariate value.
    """
    if mean_time_slope <= 0:
        raise ValueError("mean_time_slope must be positive")
    if not 0 < interaction_range < 2 * mean_time_slope:
        raise ValueError("interaction_range must be in (0, 2 * mean_time_slope)")
    if gamma_scale <= 0:
        raise ValueError("gamma_scale must be positive")

    generator = torch.Generator().manual_seed(seed)
    return {
        "W_shared": torch.randn(h, p, generator=generator) * 0.6,
        "W_out": torch.randn(K, h, generator=generator) * 0.8,
        "gamma": torch.randn(p, generator=generator) * gamma_scale,
        "slope_min": torch.tensor(mean_time_slope - interaction_range / 2),
        "slope_range": torch.tensor(interaction_range),
        "intercept": -4.0 * torch.ones(K) - torch.linspace(0, 1.5, K),
    }


def generate_data(
    n: int = 5000,
    K: int = 3,
    p: int = 5,
    censor_rate: float = 0.5,
    seed: int = 42,
    params: dict[str, Tensor] | None = None,
) -> dict[str, Tensor | dict[str, Tensor]]:
    """Generate one Case III subject sample with fixed DGP parameters."""
    torch.manual_seed(seed)
    if params is None:
        params = generate_parameters(K=K, p=p, seed=seed)

    x = torch.randn(n, p)
    cif_fn = functools.partial(compute_cif, params=params)
    assert_monotone_cif(cif_fn, x[:50], torch.linspace(0.01, 30.0, 200))

    t_event = solve_inverse_cdf(x, torch.rand(n), cif_fn, t_max=T_MAX)
    y, delta, epsilon = assign_causes_and_censor(x, t_event, cif_fn, censor_rate)
    return {
        "X": x,
        "Y": y,
        "Delta": delta,
        "T_true": t_event,
        "epsilon_true": epsilon,
        "time_slope": compute_time_slope(x, params),
        "params": params,
    }
