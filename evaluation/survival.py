#!/usr/bin/env python3
"""
Survival analysis evaluation metrics for competing risks models.

Metrics follow the DKAJ paper (arXiv:2512.08063):
  - C^td: time-dependent concordance index (Antolini formulation)
  - IBS: IPCW integrated Brier score

All public functions accept and return torch Tensors.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def _km_censoring_survival(Y: Tensor, Delta: Tensor) -> Callable:
    """Compute KM estimate of the censoring distribution G(t) = P(C > t).

    Treats censoring (Delta == 0) as the event of interest and actual events
    (Delta > 0) as censored observations.

    Args:
        Y: observed times (n,)
        Delta: event indicators (n,), 0=censored, 1..K=event

    Returns:
        callable: t (Tensor) -> G(t) survival values (Tensor)
    """
    n = Y.shape[0]
    censor_indicator = (Delta == 0).long()

    order = Y.argsort()
    y_sorted = Y[order]
    c_sorted = censor_indicator[order]

    # Build step function arrays
    times_list = [0.0]
    surv_list = [1.0]
    n_at_risk = n
    s = 1.0
    for i in range(n):
        if c_sorted[i] == 1:  # censoring "event"
            s *= 1.0 - 1.0 / n_at_risk
        n_at_risk -= 1
        times_list.append(float(y_sorted[i].item()))
        surv_list.append(s)

    times_arr = torch.tensor(times_list)
    surv_arr = torch.tensor(surv_list)

    def g_fn(t: Tensor) -> Tensor:
        # Step function: G(t) = surv at largest time <= t
        idx = torch.searchsorted(times_arr, t, side="right") - 1
        idx = idx.clamp(0, len(surv_arr) - 1)
        return surv_arr[idx].clamp(min=1e-8)

    return g_fn


def compute_ctd(
    Y_test: Tensor,
    Delta_test: Tensor,
    cif_pred: Tensor,
    times: Tensor,
    event_k: int,
) -> float:
    """Time-dependent concordance index (Antolini formulation).

    For each pair (i, j) where subject i has event_k at T_i and T_i < T_j
    (regardless of j's event status), we check whether
    CIF_k(T_i | x_i) > CIF_k(T_i | x_j).

    Args:
        Y_test: observed times (n,)
        Delta_test: event indicators (n,), 0=censored, 1..K=event
        cif_pred: predicted CIF values (n, n_times) for cause event_k
        times: time grid (n_times,)
        event_k: event type (1-indexed)

    Returns:
        concordance index in [0, 1]
    """
    event_indices = torch.nonzero(Delta_test == event_k, as_tuple=False).squeeze(1)
    if event_indices.numel() == 0:
        return 0.0
    event_times = Y_test[event_indices]
    time_indices = torch.searchsorted(times, event_times, side="right") - 1
    time_indices = time_indices.clamp(0, len(times) - 1)

    event_risk = cif_pred[event_indices, time_indices]
    comparison_risk = cif_pred[:, time_indices]
    comparable = Y_test.unsqueeze(1) > event_times.unsqueeze(0)
    total = int(comparable.sum().item())
    if total == 0:
        return 0.0
    concordant = (
        (event_risk.unsqueeze(0) > comparison_risk) & comparable
    ).sum().float() + 0.5 * (
        (event_risk.unsqueeze(0) == comparison_risk) & comparable
    ).sum().float()
    return float((concordant / total).item())


def compute_ibs(
    Y_test: Tensor,
    Delta_test: Tensor,
    Y_train: Tensor,
    Delta_train: Tensor,
    cif_pred: Tensor,
    times: Tensor,
    event_k: int,
) -> float:
    """IPCW integrated Brier score for cause-specific CIF.

    Uses KM estimate of censoring distribution from training data.
    Integrates Brier score over the time grid with trapezoidal rule.
    Normalizes by the time range.

    Args:
        Y_test: test observed times (n,)
        Delta_test: test event indicators (n,), 0=censored, 1..K=event
        Y_train: training observed times (for censoring KM)
        Delta_train: training event indicators (for censoring KM)
        cif_pred: predicted CIF values (n, n_times) for cause event_k
        times: time grid (n_times,)
        event_k: event type (1-indexed)

    Returns:
        integrated Brier score (lower is better)
    """
    g_fn = _km_censoring_survival(Y_train, Delta_train)
    before_or_at = Y_test.unsqueeze(1) <= times.unsqueeze(0)
    observed_event = Delta_test.unsqueeze(1) != 0
    event_of_interest = Delta_test.unsqueeze(1) == event_k
    after = Y_test.unsqueeze(1) > times.unsqueeze(0)

    target = (before_or_at & event_of_interest).float()
    g_at_observed_time = g_fn(Y_test).unsqueeze(1)
    g_at_evaluation_time = g_fn(times).unsqueeze(0)
    weights = (before_or_at & observed_event).float() / g_at_observed_time
    weights = weights + after.float() / g_at_evaluation_time
    bs_values = ((target - cif_pred).pow(2) * weights).mean(dim=0)

    # Trapezoidal integration, normalized by time range
    dt = times[1:] - times[:-1]
    ibs = float(((bs_values[:-1] + bs_values[1:]) / 2.0 * dt).sum().item())
    ibs /= float((times[-1] - times[0]).item())
    return ibs


def build_evaluation_time_grid(
    Y_test: Tensor,
    Delta_test: Tensor,
    n_grid: int = 100,
    percentile_cap: float = 90.0,
) -> Tensor:
    """Build evaluation time grid from quantiles of test event times.

    Args:
        Y_test: test observed times (n,)
        Delta_test: test event indicators (n,)
        n_grid: number of grid points
        percentile_cap: upper percentile to truncate at

    Returns:
        times: sorted time grid (Tensor)
    """
    event_mask = Delta_test > 0
    event_times = Y_test[event_mask] if event_mask.sum() > 0 else Y_test
    event_times_sorted, _ = event_times.sort()
    n_events = len(event_times_sorted)

    # Quantile indices
    quantile_fracs = torch.linspace(0, percentile_cap / 100.0, n_grid)
    indices = (quantile_fracs * (n_events - 1)).long().clamp(0, n_events - 1)
    times = event_times_sorted[indices]
    times = torch.unique(times)
    # Remove zero or near-zero times
    times = times[times > 1e-6]
    return times


def evaluate_cif_metrics(
    cif_pred: Tensor,
    Y_test: Tensor,
    Delta_test: Tensor,
    Y_train: Tensor,
    Delta_train: Tensor,
    eval_times: Tensor,
    K: int,
) -> dict[str, float]:
    """Evaluate a CIF prediction tensor using C^td and IBS for all causes.

    Convenience wrapper that calls compute_ctd and compute_ibs per cause
    and computes overall averages.

    Args:
        cif_pred: predicted CIF (n, K, n_times)
        Y_test: test observed times (n,)
        Delta_test: test event indicators (n,)
        Y_train: training observed times (for censoring KM)
        Delta_train: training event indicators (for censoring KM)
        eval_times: time grid (n_times,)
        K: number of causes

    Returns:
        dict with Ctd_cause_k, IBS_cause_k, and overall averages
    """
    results: dict[str, float] = {}
    for k in range(1, K + 1):
        results[f"Ctd_cause_{k}"] = compute_ctd(
            Y_test, Delta_test, cif_pred[:, k - 1, :], eval_times, k
        )
        results[f"IBS_cause_{k}"] = compute_ibs(
            Y_test,
            Delta_test,
            Y_train,
            Delta_train,
            cif_pred[:, k - 1, :],
            eval_times,
            k,
        )
    ctd_vals = [results[f"Ctd_cause_{k}"] for k in range(1, K + 1)]
    ibs_vals = [results[f"IBS_cause_{k}"] for k in range(1, K + 1)]
    results["Ctd_overall"] = sum(ctd_vals) / K
    results["IBS_overall"] = sum(ibs_vals) / K
    return results
