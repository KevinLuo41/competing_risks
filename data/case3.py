#!/usr/bin/env python3
"""
Simulation data generation for competing risks — Case III (functional covariates).

mu_k(X, t) = intercept_k + sum_j <X_j, beta_{k,j}> + alpha * t

where X_j(s) are functional covariates on [0,1], beta_{k,j}(s) are true
functional effects, and <.,.> denotes the L2 inner product. The time slope
`alpha` is shared across all K causes, which guarantees every cause-specific
CIF F_k(t|X) is monotone non-decreasing in t (see
`our_paper/monotonicity_issue.md` for the proof).

Each functional covariate is represented on a discrete grid of G points.
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

# Number of Fourier basis functions for generating smooth random functions
N_BASIS = 5


def _generate_fourier_basis(n_grid: int, n_basis: int) -> Tensor:
    """Generate Fourier basis functions on [0, 1].

    Returns:
        basis: (2*n_basis, n_grid) — cos and sin basis functions
    """
    s = torch.linspace(0, 1, n_grid)  # (G,)
    basis = []
    for freq in range(1, n_basis + 1):
        basis.append(torch.cos(2 * torch.pi * freq * s))
        basis.append(torch.sin(2 * torch.pi * freq * s))
    return torch.stack(basis, dim=0)  # (2*n_basis, G)


def _generate_functional_covariates(
    n: int, p: int, n_grid: int, n_basis: int
) -> Tensor:
    """Generate p functional covariates for n subjects using Fourier basis.

    X_j(s) = sum_l (a_{jl} cos(2*pi*l*s) + b_{jl} sin(2*pi*l*s))
    with coefficients decaying as 1/l for smoothness.

    Args:
        n: number of subjects
        p: number of functional covariates
        n_grid: number of grid points
        n_basis: number of Fourier basis pairs

    Returns:
        X_func: (n, p, n_grid) — discretized functional covariates
    """
    basis = _generate_fourier_basis(n_grid, n_basis)  # (2*n_basis, G)

    # Coefficients with 1/l decay for smoothness
    decay = []
    for freq in range(1, n_basis + 1):
        decay.extend([1.0 / freq, 1.0 / freq])
    decay = torch.tensor(decay)  # (2*n_basis,)

    # Random coefficients: (n, p, 2*n_basis)
    coeffs = torch.randn(n, p, 2 * n_basis) * decay.unsqueeze(0).unsqueeze(0)

    # X_func[i, j, :] = coeffs[i, j, :] @ basis, shape (n, p, G)
    X_func = torch.einsum("npb,bg->npg", coeffs, basis)

    return X_func


def _generate_true_functional_effects(
    K: int, p: int, n_grid: int, n_basis: int
) -> Tensor:
    """Generate true functional effect beta_{k,j}(s) using Fourier basis.

    Returns:
        beta_func: (K, p, n_grid) — true functional effects on grid
    """
    basis = _generate_fourier_basis(n_grid, n_basis)  # (2*n_basis, G)

    # True effect coefficients with decay
    decay = []
    for freq in range(1, n_basis + 1):
        decay.extend([1.0 / freq, 1.0 / freq])
    decay = torch.tensor(decay)

    coeffs = torch.randn(K, p, 2 * n_basis) * 2.0 * decay.unsqueeze(0).unsqueeze(0)

    beta_func = torch.einsum("kpb,bg->kpg", coeffs, basis)  # (K, p, G)
    return beta_func


def compute_cif(
    x_func: Tensor,
    t: Tensor,
    beta_func: Tensor,
    alpha: Tensor,
    intercept: Tensor,
    n_grid: int,
) -> tuple[Tensor, Tensor]:
    """Compute CIF values F_k(t|X) for Case III (functional).

    Args:
        x_func: functional covariates (n, p, G)
        t: time points (n,)
        beta_func: true functional effects (K, p, G)
        alpha: scalar shared time coefficient ()
        intercept: intercept terms (K,)
        n_grid: number of grid points (for normalization)

    Returns:
        F: CIF values (n, K), S: survival values (n,)
    """
    # Inner products: <X_j, beta_{k,j}> = (1/G) sum_g X_j(s_g) * beta_{k,j}(s_g)
    # x_func: (n, p, G), beta_func: (K, p, G)
    # Result: (n, K) = sum over p and G
    inner = torch.einsum("npg,kpg->nk", x_func, beta_func) / n_grid

    # mu_k = intercept_k + <X, beta_k> + alpha * t   (alpha shared across causes)
    mu = intercept.unsqueeze(0) + inner + t.unsqueeze(1) * alpha

    return mu_to_cif(mu)


def generate_data(
    n: int = 5000,
    K: int = 2,
    p: int = 3,
    n_grid: int = 50,
    censor_rate: float = 0.5,
    seed: int = 42,
) -> dict[str, Tensor | int]:
    """Generate simulation data for Case III (functional covariates).

    Args:
        n: number of samples
        K: number of competing event types
        p: number of functional covariates
        n_grid: number of grid points per functional covariate
        censor_rate: approximate censoring proportion
        seed: random seed

    Returns:
        dict with X_func, Y, Delta, T_true, epsilon_true, and true parameters
    """
    torch.manual_seed(seed)

    # True parameters — alpha shared across causes guarantees CIF monotonicity
    alpha = torch.tensor(0.4)  # scalar
    intercept = -3.0 * torch.ones(K) - torch.linspace(0, 1.5, K)

    # True functional effects
    beta_func = _generate_true_functional_effects(K, p, n_grid, N_BASIS)

    # Step 1: Generate functional covariates
    x_func = _generate_functional_covariates(n, p, n_grid, N_BASIS)  # (n, p, G)

    # Step 2: Generate event times via inverse CDF
    cif_fn = functools.partial(
        compute_cif,
        beta_func=beta_func,
        alpha=alpha,
        intercept=intercept,
        n_grid=n_grid,
    )

    # Sanity check: verify monotonicity on a sample of x's
    assert_monotone_cif(cif_fn, x_func[:50], torch.linspace(0.01, 30.0, 200))

    u = torch.rand(n)
    t_event = solve_inverse_cdf(x_func, u, cif_fn)

    # Steps 3–4: Cause assignment + censoring
    y, delta, epsilon = assign_causes_and_censor(x_func, t_event, cif_fn, censor_rate)

    return {
        "X_func": x_func,
        "Y": y,
        "Delta": delta,
        "T_true": t_event,
        "epsilon_true": epsilon,
        "beta_func": beta_func,
        "alpha": alpha,
        "intercept": intercept,
        "n_grid": n_grid,
    }
