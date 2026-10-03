#!/usr/bin/env python3
"""
Simulation data generation for competing risks — Case II (nonlinear).

mu_k(x, t) = intercept_k + alpha * t + W_out_k @ tanh(W_shared @ x)

A shared hidden layer extracts nonlinear features from x; cause-specific
output rows W_out[k] map these into per-cause logits. The time effect
`alpha * t` is shared across all K causes, which guarantees that every
cause-specific CIF F_k(t|x) is monotone non-decreasing in t (see
`our_paper/monotonicity_issue.md` for the proof). The nonlinearity is
preserved in the covariate pathway via tanh.
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
    params: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """Compute CIF values F_k(t|x) for Case II (nonlinear, monotone).

    Args:
        x: covariates (n, p)
        t: time points (n,)
        params: dict with W_shared, W_out, alpha, intercept

    Returns:
        F: CIF values (n, K), S: survival values (n,)
    """
    W_shared = params["W_shared"]  # (h, p)
    W_out = params["W_out"]  # (K, h)
    alpha = params["alpha"]  # scalar
    intercept = params["intercept"]  # (K,)
    K = intercept.shape[0]

    if t.dim() == 0:
        t = t.expand(x.shape[0])

    # Shared nonlinear feature extractor on covariates only (no t in tanh)
    hidden = torch.tanh(x @ W_shared.T)  # (n, h)

    # Cause-specific logits: shared time slope + cause-specific nonlinear x effect
    mu_list = []
    for k in range(K):
        mu_k = intercept[k] + alpha * t + hidden @ W_out[k]  # (n,)
        mu_list.append(mu_k)
    mu = torch.stack(mu_list, dim=1)  # (n, K)

    return mu_to_cif(mu)


def generate_parameters(
    K: int = 3,
    p: int = 5,
    h: int = 12,
    seed: int = 42,
) -> dict[str, Tensor]:
    """Generate the fixed Case II data-generating parameters."""
    generator = torch.Generator().manual_seed(seed)
    return {
        "W_shared": torch.randn(h, p, generator=generator) * 0.6,
        "W_out": torch.randn(K, h, generator=generator) * 0.8,
        "alpha": torch.tensor(0.4),
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
    """Generate simulation data for Case II (nonlinear interactive).

    Args:
        n: number of samples
        K: number of competing event types
        p: number of covariates
        censor_rate: approximate censoring proportion
        seed: random seed
        params: optional fixed Case II parameters

    Returns:
        dict with X, Y, Delta, T_true, epsilon_true, params, intercept
    """
    torch.manual_seed(seed)

    if params is None:
        h = 12
        params = {
            "W_shared": torch.randn(h, p) * 0.6,
            "W_out": torch.randn(K, h) * 0.8,
            "alpha": torch.tensor(0.4),
            "intercept": -4.0 * torch.ones(K) - torch.linspace(0, 1.5, K),
        }
    required = {"W_shared", "W_out", "alpha", "intercept"}
    if set(params) != required:
        raise ValueError(
            f"expected parameter keys {sorted(required)}, got {sorted(params)}"
        )
    h = params["W_shared"].shape[0]
    expected_shapes = {
        "W_shared": (h, p),
        "W_out": (K, h),
        "intercept": (K,),
    }
    for name, expected_shape in expected_shapes.items():
        if params[name].shape != expected_shape:
            raise ValueError(
                f"expected {name} shape {expected_shape}, got {tuple(params[name].shape)}"
            )
    if params["alpha"].numel() != 1:
        raise ValueError(
            f"expected scalar alpha, got shape {tuple(params['alpha'].shape)}"
        )

    # Generate covariates
    x = torch.randn(n, p)

    # Generate event times via inverse CDF
    cif_fn = functools.partial(compute_cif, params=params)

    # Sanity check: verify monotonicity on a sample of x's
    assert_monotone_cif(cif_fn, x[:50], torch.linspace(0.01, 30.0, 200))

    u = torch.rand(n)
    t_event = solve_inverse_cdf(x, u, cif_fn)

    # Cause assignment + censoring
    y, delta, epsilon = assign_causes_and_censor(x, t_event, cif_fn, censor_rate)

    return {
        "X": x,
        "Y": y,
        "Delta": delta,
        "T_true": t_event,
        "epsilon_true": epsilon,
        "params": params,
        "intercept": params["intercept"],
    }
