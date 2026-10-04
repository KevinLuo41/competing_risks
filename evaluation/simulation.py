#!/usr/bin/env python3
"""
MSE and classification accuracy for simulation studies
where the true CIF is known.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def compute_dist(
    cif_pred: Tensor,
    survival_pred: Tensor,
    negative_survival_atol: float = 1e-6,
) -> dict[str, float | int]:
    """Measure probability incoherence and negative implied survival.

    Args:
        cif_pred: predicted CIF tensor with shape (n, K, n_times)
        survival_pred: native overall survival with shape (n, n_times)
        negative_survival_atol: numerical tolerance below zero for counting
            negative implied survival

    Returns:
        Distance from one using native survival, plus exact and
        tolerance-aware counts of negative implied survival
        ``1 - sum_k CIF_k`` over subjects and times.
    """
    if cif_pred.dim() != 3:
        raise ValueError(f"expected CIF shape (n, K, T), got {tuple(cif_pred.shape)}")
    expected_survival_shape = (cif_pred.shape[0], cif_pred.shape[2])
    if tuple(survival_pred.shape) != expected_survival_shape:
        raise ValueError(
            "expected survival shape "
            f"{expected_survival_shape}, got {tuple(survival_pred.shape)}"
        )
    distances = (cif_pred.sum(dim=1) + survival_pred - 1.0).abs()
    implied_survival = 1.0 - cif_pred.sum(dim=1)
    negative = implied_survival < 0.0
    below_tolerance = implied_survival < -negative_survival_atol
    negative_subjects = negative.any(dim=1)
    below_tolerance_subjects = below_tolerance.any(dim=1)
    total_count = implied_survival.numel()
    total_subject_count = implied_survival.shape[0]
    return {
        "Dist": float(distances.mean().item()),
        "Dist_max": float(distances.max().item()),
        "Implied_S_negative_count": int(negative.sum().item()),
        "Implied_S_total_count": total_count,
        "Implied_S_negative_fraction": float(negative.float().mean().item()),
        "Implied_S_negative_subject_count": int(negative_subjects.sum().item()),
        "Implied_S_total_subject_count": total_subject_count,
        "Implied_S_negative_subject_fraction": float(
            negative_subjects.float().mean().item()
        ),
        "Implied_S_below_tolerance_count": int(below_tolerance.sum().item()),
        "Implied_S_below_tolerance_fraction": float(
            below_tolerance.float().mean().item()
        ),
        "Implied_S_below_tolerance_subject_count": int(
            below_tolerance_subjects.sum().item()
        ),
        "Implied_S_below_tolerance_subject_fraction": float(
            below_tolerance_subjects.float().mean().item()
        ),
        "Implied_S_min": float(implied_survival.min().item()),
    }


def compute_mse_accuracy(
    cif_pred: Tensor,
    X_test: Tensor,
    times: Tensor,
    true_cif_fn: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    K: int,
) -> dict[str, float]:
    """Compute MSE and classification accuracy against true CIF.

    For simulation studies where the true CIF is known.

    Args:
        cif_pred: predicted CIF (n, K, n_times) from predict_cif_grid()
        X_test: test covariates (n, ...)
        times: evaluation time grid (n_times,)
        true_cif_fn: callable(X, t) -> (F, S) returning true CIF values
        K: number of causes

    Returns:
        dict with MSE_cause_k, MSE_overall, classification_accuracy
    """
    n = X_test.shape[0]
    n_times = len(times)

    # Generate true CIF grid
    cif_true = torch.zeros(n, K, n_times)
    with torch.no_grad():
        for ti in range(n_times):
            t_batch = times[ti].expand(n)
            f_true, _ = true_cif_fn(X_test, t_batch)
            cif_true[:, :, ti] = f_true

    # MSE per cause
    mse_per_cause = torch.zeros(K)
    for k in range(K):
        mse_per_cause[k] = (cif_pred[:, k, :] - cif_true[:, k, :]).pow(2).mean()

    results: dict[str, float] = {}
    for k in range(K):
        results[f"MSE_cause_{k + 1}"] = mse_per_cause[k].item()
    results["MSE_overall"] = mse_per_cause.mean().item()

    # Classification accuracy: argmax(S, F_1, ..., F_K)
    s_pred = 1.0 - cif_pred.sum(dim=1)  # (n, n_times)
    s_true = 1.0 - cif_true.sum(dim=1)  # (n, n_times)
    correct = 0
    total = 0
    for ti in range(n_times):
        pred_full = torch.cat([s_pred[:, ti : ti + 1], cif_pred[:, :, ti]], dim=1)
        true_full = torch.cat([s_true[:, ti : ti + 1], cif_true[:, :, ti]], dim=1)
        correct += int(
            (pred_full.argmax(dim=1) == true_full.argmax(dim=1)).sum().item()
        )
        total += n
    results["classification_accuracy"] = correct / total if total > 0 else 0.0

    return results
