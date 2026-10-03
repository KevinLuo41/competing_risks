#!/usr/bin/env python3
"""
Simulation data generation for competing risks — Case I (linear).

mu_k(x, t) = intercept_k + beta_k^T x + alpha * t

The time slope `alpha` is shared across all K causes. This guarantees that
every cause-specific CIF F_k(t|x) is monotone non-decreasing in t (see
`our_paper/monotonicity_issue.md` for the proof).
"""

from __future__ import annotations

import functools

import torch
from torch import Tensor

from .utils import (
    assert_monotone_cif,
    assign_causes_and_censor,
    mu_to_cif,
    solve_inverse_cdf,
)


def compute_cif(
    x: Tensor,
    t: Tensor,
    beta: Tensor,
    alpha: Tensor,
    intercept: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute CIF values F_k(t|x) for Case I (linear).

    Args:
        x: covariates (n, p)
        t: time points (n,)
        beta: coefficients (K, p)
        alpha: scalar shared time coefficient ()
        intercept: intercept terms (K,)

    Returns:
        F: CIF values (n, K), S: survival values (n,)
    """
    mu = intercept.unsqueeze(0) + x @ beta.T + t.unsqueeze(1) * alpha
    return mu_to_cif(mu)


def generate_parameters(
    K: int = 2,
    p: int = 5,
    seed: int = 42,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate the fixed Case I data-generating parameters.

    A local generator keeps parameter generation independent from the random
    streams used to draw subjects in individual Monte Carlo replicates.
    """
    generator = torch.Generator().manual_seed(seed)
    beta = torch.randn(K, p, generator=generator) * 0.6
    alpha = torch.tensor(0.4)
    intercept = -4.0 * torch.ones(K) - torch.linspace(0, 1.5, K)
    return beta, alpha, intercept


def generate_data(
    n: int = 5000,
    K: int = 2,
    p: int = 5,
    censor_rate: float = 0.5,
    seed: int = 42,
    beta: Tensor | None = None,
    alpha: Tensor | None = None,
    intercept: Tensor | None = None,
) -> dict[str, Tensor]:
    """Generate simulation data for Case I (linear).

    Args:
        n: number of samples
        K: number of competing event types
        p: number of covariates
        censor_rate: approximate censoring proportion
        seed: random seed
        beta: optional fixed coefficients (K, p)
        alpha: optional fixed shared time coefficient
        intercept: optional fixed intercepts (K,)

    Returns:
        dict with X, Y, Delta, T_true, epsilon_true, beta, alpha, intercept
    """
    torch.manual_seed(seed)

    provided = (beta is not None, alpha is not None, intercept is not None)
    if any(provided) and not all(provided):
        raise ValueError("beta, alpha, and intercept must be provided together")
    if beta is None or alpha is None or intercept is None:
        # Preserve the original random stream when parameters are not supplied.
        beta = torch.randn(K, p) * 0.6
        alpha = torch.tensor(0.4)
        intercept = -4.0 * torch.ones(K) - torch.linspace(0, 1.5, K)
    if beta.shape != (K, p):
        raise ValueError(f"expected beta shape {(K, p)}, got {tuple(beta.shape)}")
    if alpha.numel() != 1:
        raise ValueError(f"expected scalar alpha, got shape {tuple(alpha.shape)}")
    if intercept.shape != (K,):
        raise ValueError(
            f"expected intercept shape {(K,)}, got {tuple(intercept.shape)}"
        )

    # Step 1: Generate covariates
    x = torch.randn(n, p)

    # Step 2: Generate event times via inverse CDF
    cif_fn = functools.partial(compute_cif, beta=beta, alpha=alpha, intercept=intercept)

    # Sanity check: verify monotonicity on a sample of x's
    assert_monotone_cif(cif_fn, x[:50], torch.linspace(0.01, 30.0, 200))

    u = torch.rand(n)
    t_event = solve_inverse_cdf(x, u, cif_fn)

    # Steps 3–4: Cause assignment + censoring
    y, delta, epsilon = assign_causes_and_censor(x, t_event, cif_fn, censor_rate)

    return {
        "X": x,
        "Y": y,
        "Delta": delta,
        "T_true": t_event,
        "epsilon_true": epsilon,
        "beta": beta,
        "alpha": alpha,
        "intercept": intercept,
    }
