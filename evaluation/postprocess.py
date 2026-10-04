#!/usr/bin/env python3
# pyre-strict
"""Post-hoc monotonicity correction for predicted CIFs.

Uses scikit-learn's IsotonicRegression (Pool-Adjacent-Violators algorithm),
which gives the L2-optimal projection of a sequence onto the cone of non-
decreasing functions (Robertson, Wright, Dykstra 1988). This is the
principled fix for residual non-monotonicity in SoftComp's CIF predictions
outside the training-data-rich time window.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from torch import Tensor


def isotonic_project_cif(cif: Tensor, t_grid: Tensor | None = None) -> Tensor:
    """Apply isotonic regression along the time axis for every (subject, cause).

    Args:
        cif: predicted CIF tensor of shape (n, K, T)
        t_grid: optional 1-D Tensor of length T with the time coordinates.
            Only the strictly-increasing ordering is used; values are not
            otherwise consumed by sklearn's PAV. Defaults to range(T).

    Returns:
        cif_corrected: same shape as cif, non-decreasing in t for every
            (subject, cause) pair, minimizing sum-of-squares distance to cif.
    """
    if cif.dim() != 3:
        raise ValueError(f"expected 3-D tensor (n, K, T), got shape {tuple(cif.shape)}")
    n, K, T = cif.shape
    x = (
        np.arange(T, dtype=np.float64)
        if t_grid is None
        else t_grid.detach().cpu().numpy().astype(np.float64)
    )
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    cif_np = cif.detach().cpu().numpy()
    out = np.empty_like(cif_np)
    for i in range(n):
        for k in range(K):
            out[i, k, :] = iso.fit_transform(x, cif_np[i, k, :])
    return torch.from_numpy(out).to(cif.dtype)


def enforce_cif_simplex(cif: Tensor) -> Tensor:
    """Ensure that the cause-specific CIFs sum to at most one.

    Each subject's complete CIF surface is divided by one constant whenever
    its largest total CIF exceeds one. Applied after PAV, this preserves the
    time monotonicity of every cause while making implied survival
    ``1 - sum_k F_k(t)`` nonnegative.
    """
    if cif.dim() != 3:
        raise ValueError(f"expected 3-D tensor (n, K, T), got shape {tuple(cif.shape)}")
    scale = cif.sum(dim=1).amax(dim=1).clamp(min=1.0).view(-1, 1, 1)
    return cif / scale


def count_monotone_violations(cif: Tensor, atol: float = 1e-6) -> dict[str, float]:
    """Diagnostic: count and characterize non-monotonicity in predicted CIFs.

    Args:
        cif: predicted CIF tensor (n, K, T)
        atol: tolerance for considering a decrease a violation

    Returns:
        dict with violation statistics: fraction of (subject, cause) pairs with
        any violation, max absolute drop, mean violation magnitude
    """
    diffs = cif[:, :, 1:] - cif[:, :, :-1]  # (n, K, T-1)
    has_violation = (diffs < -atol).any(dim=2)  # (n, K)
    n, K = has_violation.shape
    n_pairs = n * K
    n_violating_pairs = int(has_violation.sum().item())
    max_drop = max(0.0, float(-diffs.min().item()))
    neg_diffs = (-diffs).clamp(min=0)
    mean_drop = (
        float(neg_diffs[neg_diffs > 0].mean().item()) if (neg_diffs > 0).any() else 0.0
    )
    return {
        "violating_pair_fraction": n_violating_pairs / n_pairs,
        "max_drop": max_drop,
        "mean_violation_magnitude": mean_drop,
    }
