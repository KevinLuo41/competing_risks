#!/usr/bin/env python3
"""
Shared utilities for competing risks data generation.

Contains the numerically stable mu→CIF conversion, inverse CDF solver,
and censoring/cause-assignment logic used by all cases.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def assert_monotone_cif(
    cif_fn: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    x_sample: Tensor,
    t_grid: Tensor,
    atol: float = 1e-6,
) -> None:
    """Verify that F_k(t|x) is non-decreasing in t for every k and every x.

    Evaluates the CIF on a dense time grid for a sample of covariates and
    asserts no decrease larger than `atol` between consecutive grid points.
    Raises AssertionError with diagnostic info if any cause is non-monotone.

    Args:
        cif_fn: callable(x, t) -> (F, S) where F has shape (n, K)
        x_sample: covariates (n, p) or (n, p, G) for functional inputs
        t_grid: 1-D tensor of time points (G_t,), strictly increasing
        atol: tolerance for numerical decreases (default 1e-6)
    """
    n = x_sample.shape[0]
    g_t = t_grid.shape[0]
    F_traj = []
    for t in t_grid:
        F_t, _ = cif_fn(x_sample, t.expand(n))
        F_traj.append(F_t)
    F = torch.stack(F_traj, dim=1)  # (n, G_t, K)
    diffs = F[:, 1:, :] - F[:, :-1, :]  # (n, G_t-1, K)
    min_diff = diffs.min().item()
    if min_diff < -atol:
        K = F.shape[2]
        per_cause = [(diffs[:, :, k] < -atol).sum().item() for k in range(K)]
        raise AssertionError(
            f"CIF non-monotone (atol={atol}): min ΔF = {min_diff:.6e}; "
            f"violating (sample, t)-pairs per cause = {per_cause} "
            f"(out of n={n}, t_grid={g_t})"
        )


def mu_to_cif(mu: Tensor) -> tuple[Tensor, Tensor]:
    """Convert raw logits mu to CIF values F and survival S.

    F_k = exp(mu_k) / (1 + sum exp(mu_j)), numerically stable.

    Args:
        mu: raw logits (n, K)

    Returns:
        F: CIF values (n, K), S: survival values (n,)
    """
    mu_max = mu.max(dim=1, keepdim=True).values.clamp(min=0.0)
    exp_mu = torch.exp(mu - mu_max)
    denom = torch.exp(-mu_max) + exp_mu.sum(dim=1, keepdim=True)
    f = exp_mu / denom  # (n, K)
    s = torch.exp(-mu_max).squeeze(1) / denom.squeeze(1)  # (n,)
    return f, s


def solve_inverse_cdf(
    x: Tensor,
    u: Tensor,
    cif_fn: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    t_max: float = 30.0,
    n_iter: int = 80,
) -> Tensor:
    """Solve F(t|x) = u for t using bisection method.

    Args:
        x: covariates (n, p)
        u: uniform random values (n,)
        cif_fn: callable(x, t) -> (F, S) that computes CIF values
        t_max: upper bound for bisection
        n_iter: number of bisection iterations

    Returns:
        t_event: solved event times (n,)
    """
    n = x.shape[0]
    t_lo = torch.zeros(n)
    t_hi = torch.ones(n) * t_max

    for _ in range(n_iter):
        t_mid = (t_lo + t_hi) / 2.0
        _, s_mid = cif_fn(x, t_mid)
        f_mid = 1.0 - s_mid
        t_lo = torch.where(f_mid < u, t_mid, t_lo)
        t_hi = torch.where(f_mid < u, t_hi, t_mid)

    return (t_lo + t_hi) / 2.0


def assign_causes_and_censor(
    x: Tensor,
    t_event: Tensor,
    cif_fn: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    censor_rate: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Draw cause indicators and simulate censoring.

    Steps:
      1. Compute F_k(T|x), draw cause from multinomial
      2. Simulate censoring from Exponential distribution

    Args:
        x: covariates (n, p)
        t_event: true event times (n,)
        cif_fn: callable(x, t) -> (F, S)
        censor_rate: target censoring proportion; 0 disables censoring

    Returns:
        y: observed times (n,)
        delta: event indicators (n,), 0=censored, 1..K=event
        epsilon: true cause indicators (n,), 1..K
    """
    n = x.shape[0]

    # Draw cause indicators from multinomial
    f_at_t, _ = cif_fn(x, t_event)
    cause_probs = f_at_t / f_at_t.sum(dim=1, keepdim=True).clamp(min=1e-10)
    epsilon = torch.multinomial(cause_probs, num_samples=1).squeeze(1) + 1
    if censor_rate == 0.0:
        return t_event, epsilon, epsilon

    # Simulate censoring from Exponential distribution
    # Goal: P(C < T_median) ≈ censor_rate
    # Exponential CDF: P(C < t) = 1 - exp(-t / scale)
    # => scale = T_median / (-log(1 - censor_rate))
    censor_scale = (
        t_event.median().item() / (-torch.log(torch.tensor(1.0 - censor_rate))).item()
    )
    c = torch.distributions.Exponential(1.0 / censor_scale).sample((n,))

    y = torch.minimum(t_event, c)
    delta = torch.where(t_event <= c, epsilon, torch.zeros_like(epsilon))

    return y, delta, epsilon


def generate_test_observations(
    n_test: int,
    p: int,
    cif_fn: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    censor_rate: float = 0.3,
    seed: int = 999,
    t_max: float = 30.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate synthetic test data (X, Y, Delta) from a known CIF.

    Generates random covariates, draws event times via inverse CDF,
    assigns causes from CIF proportions, and simulates censoring.

    Args:
        n_test: number of test samples
        p: number of covariates (or tuple for functional shape)
        cif_fn: callable(X, t) -> (F, S) returning true CIF values
        censor_rate: target censoring proportion; 0 disables censoring
        seed: random seed for reproducibility
        t_max: upper bound for inverse-CDF event-time solving

    Returns:
        X_test: covariates (n_test, p)
        Y_test: observed times (n_test,)
        Delta_test: event indicators (n_test,), 0=censored, 1..K=event
    """
    torch.manual_seed(seed)
    X = torch.randn(n_test, p)

    u = torch.rand(n_test)
    t_event = solve_inverse_cdf(X, u, cif_fn, t_max=t_max)

    f_at_t, _ = cif_fn(X, t_event)
    cause_probs = f_at_t / f_at_t.sum(dim=1, keepdim=True).clamp(min=1e-10)
    epsilon = torch.multinomial(cause_probs, num_samples=1).squeeze(1) + 1
    if censor_rate == 0.0:
        return X, t_event, epsilon

    censor_scale = (
        t_event.median().item() / (-torch.log(torch.tensor(1.0 - censor_rate))).item()
    )
    c = torch.distributions.Exponential(1.0 / censor_scale).sample((n_test,))

    y = torch.minimum(t_event, c)
    delta = torch.where(t_event <= c, epsilon, torch.zeros_like(epsilon))

    return X, y, delta
