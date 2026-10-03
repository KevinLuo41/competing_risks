#!/usr/bin/env python3
"""
CS-Cox: Cause-specific Cox proportional hazards. Fits one CoxPHFitter per
cause (treating other events as censored) and combines them using
probability-conserving exponential-hazard integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import Tensor


class CsCox:
    """Cause-specific Cox proportional hazards for competing risks.

    For each cause k, fits a separate Cox model treating events from other
    causes as censored. At each merged baseline-hazard time, converts the
    total hazard increment into an event probability and allocates it among
    causes in proportion to their hazard increments:

        event_probability_j = 1 - exp(-sum_k delta H_kj)
        delta CIF_kj = S(t_j-|x) * event_probability_j
                        * delta H_kj / sum_l delta H_lj

    This discretization is consistent with ``S(t|x) = exp(-sum_k H_k(t|x))``
    and guarantees that the predicted CIFs and survival sum to one.

    Uses lifelines.CoxPHFitter with L2 regularization.

    Args:
        n_causes: number of competing event types (K)
        penalizer: L2 regularization strength
    """

    def __init__(self, n_causes: int, penalizer: float = 0.01) -> None:
        self.n_causes = n_causes
        self.penalizer = penalizer
        self.models: list = []
        self.baseline_hazards: list = []

    @classmethod
    def load_from_checkpoint(cls, data: dict) -> CsCox:
        """Reconstruct a fitted CsCox model from a checkpoint.

        Args:
            data: checkpoint dict with keys: models, baseline_hazards, n_causes,
                and optional penalizer

        Returns:
            A CsCox instance ready for predict_cif()
        """
        model = cls(
            n_causes=data["n_causes"],
            penalizer=data.get("penalizer", 0.01),
        )
        model.models = data["models"]
        model.baseline_hazards = data["baseline_hazards"]
        return model

    def fit(
        self,
        X_train: Tensor,
        Y_train: Tensor,
        Delta_train: Tensor,
    ) -> None:
        """Fit cause-specific Cox models.

        Args:
            X_train: training covariates (n_train, p), torch Tensor
            Y_train: training observed times (n_train,)
            Delta_train: training event indicators (n_train,), 0=censored, 1..K=event
        """
        from lifelines import CoxPHFitter, utils as lifelines_utils

        original_check_nans_or_infs = lifelines_utils.check_nans_or_infs
        original_isnan = np.isnan

        # Pandas exposes the Cox design as a multidimensional strided array.
        # NumPy 2.2's UBSan build rejects its SIMD isnan path, so run the same
        # elementwise checks on a flat contiguous view; the fitted data is unchanged.
        def _check_nans_or_infs_contiguous(value: object) -> None:
            if isinstance(value, (pd.Series, pd.DataFrame)):
                value = value.to_numpy()
            contiguous_value = np.ascontiguousarray(value).reshape(-1)
            original_check_nans_or_infs(contiguous_value)

        def _isnan_contiguous(value: object) -> np.ndarray:
            array = np.asarray(value)
            result = original_isnan(np.ascontiguousarray(array).reshape(-1))
            return result.reshape(array.shape)

        X_np = np.ascontiguousarray(X_train.numpy(), dtype=np.float64)
        Y_np = np.ascontiguousarray(Y_train.numpy(), dtype=np.float64)
        Delta_np = np.ascontiguousarray(Delta_train.numpy(), dtype=np.int64)

        p = X_np.shape[1]
        col_names = [f"x{i}" for i in range(p)]

        self.models = []
        self.baseline_hazards = []

        for k in range(1, self.n_causes + 1):
            # For cause k: event = (Delta == k), censored = everything else
            event_k = (Delta_np == k).astype(int)

            columns = {
                name: np.ascontiguousarray(X_np[:, index])
                for index, name in enumerate(col_names)
            }
            columns["T"] = Y_np.copy()
            columns["E"] = event_k.copy()
            df = pd.DataFrame(columns, copy=True)

            cph = CoxPHFitter(penalizer=self.penalizer)
            lifelines_utils.check_nans_or_infs = _check_nans_or_infs_contiguous
            np.isnan = _isnan_contiguous
            try:
                cph.fit(df, duration_col="T", event_col="E")
            finally:
                lifelines_utils.check_nans_or_infs = original_check_nans_or_infs
                np.isnan = original_isnan

            self.models.append(cph)
            self.baseline_hazards.append(cph.baseline_hazard_)

            n_events = event_k.sum()
            print(f"  CS-Cox: cause {k} fitted ({n_events} events)")

    def predict_cif(
        self,
        X_test: Tensor,
        times: Tensor,
    ) -> Tensor:
        """Predict CIF for each cause at given time points."""
        cif, _ = self.predict_cif_survival(X_test, times)
        return cif

    def predict_cif_survival(
        self,
        X_test: Tensor,
        times: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict CIF for each cause at given time points.

        Uses probability-conserving cause-specific hazard integration:
        1. Compute cause-specific cumulative hazard H_k(t|x) for each k
        2. Overall survival: S(t|x) = exp(-sum_k H_k(t|x))
        3. At each baseline-hazard time, convert the total hazard increment
           delta H into event probability 1 - exp(-delta H), allocate it
           among causes in proportion to delta H_k, and multiply by S(t-|x)

        Args:
            X_test: test covariates (n, p)
            times: time points to evaluate (n_times,)

        Returns:
            cif: (n, K, n_times) CIF values
            survival: (n, n_times) native overall survival values
        """
        X_np = np.ascontiguousarray(X_test.numpy(), dtype=np.float64)
        times_np = np.ascontiguousarray(times.numpy(), dtype=np.float64)
        n = X_np.shape[0]
        p = X_np.shape[1]
        n_times = len(times_np)
        col_names = [f"x{i}" for i in range(p)]

        df_test = pd.DataFrame(
            {
                name: np.ascontiguousarray(X_np[:, index])
                for index, name in enumerate(col_names)
            },
            copy=True,
        )

        # Collect baseline hazard times and values for each cause
        cause_bh_times: list[np.ndarray] = []
        cause_bh_values: list[np.ndarray] = []
        cause_log_partial_hazard: list[np.ndarray] = []

        for k_idx in range(self.n_causes):
            cph = self.models[k_idx]
            bh = self.baseline_hazards[k_idx]

            # Baseline hazard at each unique event time
            bh_times = bh.index.values.astype(float)
            bh_values = bh.values.flatten().astype(float)

            cause_bh_times.append(bh_times)
            cause_bh_values.append(bh_values)

            # Log partial hazard: beta^T * x for each test sample
            log_ph = cph.predict_log_partial_hazard(df_test).values.flatten()
            cause_log_partial_hazard.append(log_ph)

        # Merge all unique baseline hazard times across causes
        all_bh_times = np.sort(np.unique(np.concatenate(cause_bh_times)))

        # For each cause and each merged time, interpolate baseline hazard
        # (step function: h0(t) at the closest time <= t)
        cause_h0_at_merged = np.zeros((self.n_causes, len(all_bh_times)))
        for k_idx in range(self.n_causes):
            bt = cause_bh_times[k_idx]
            bv = cause_bh_values[k_idx]
            for j, t_j in enumerate(all_bh_times):
                idx = np.searchsorted(bt, t_j, side="right") - 1
                if idx >= 0 and bt[idx] == t_j:
                    cause_h0_at_merged[k_idx, j] = bv[idx]
                # else: 0 (no baseline hazard event at this time for this cause)

        # Compute individual hazard increments: h_k(t_j|x) = h0_k(t_j) * exp(beta_k^T x)
        # Shape: (n_causes, n, n_merged_times)
        cause_h_ind = np.zeros((self.n_causes, n, len(all_bh_times)))
        for k_idx in range(self.n_causes):
            # exp(log_partial_hazard) gives the hazard ratio
            hr = np.exp(cause_log_partial_hazard[k_idx])  # (n,)
            # h_k(t_j|x_i) = h0_k(t_j) * hr_i
            cause_h_ind[k_idx] = np.outer(
                hr, cause_h0_at_merged[k_idx]
            )  # (n, n_merged)

        # Compute overall hazard increments and cumulative hazard.
        total_h_increment = cause_h_ind.sum(axis=0)  # (n, n_merged)
        total_h_cumsum = total_h_increment.cumsum(axis=1)  # (n, n_merged)

        # Overall survival just before each merged time:
        # S(t_j-|x) = exp(-sum_k sum_{t_l < t_j} h_k(t_l|x))
        # For j=0: S(t_0-) = 1 (no events before first time)
        surv_before = np.ones((n, len(all_bh_times)))
        if len(all_bh_times) > 1:
            surv_before[:, 1:] = np.exp(-total_h_cumsum[:, :-1])

        # Convert each cumulative-hazard jump into event probability under the
        # same exponential-survival convention used below. Allocate that event
        # probability among causes in proportion to their hazard increments.
        # This makes sum_k CIF_k(t|x) + S(t|x) = 1 at every time point.
        event_probability = -np.expm1(-total_h_increment)
        cause_fraction = np.divide(
            cause_h_ind,
            total_h_increment[np.newaxis, :, :],
            out=np.zeros_like(cause_h_ind),
            where=total_h_increment[np.newaxis, :, :] > 0,
        )
        cif_increments = (
            cause_fraction * (surv_before * event_probability)[np.newaxis, :, :]
        )

        # Cumulative sum to get CIF at each merged time
        cif_cumsum = cif_increments.cumsum(axis=2)  # (n_causes, n, n_merged)

        # Interpolate to requested evaluation times
        cif = np.zeros((n, self.n_causes, n_times))
        survival = np.ones((n, n_times))
        for ti, t_val in enumerate(times_np):
            idx = np.searchsorted(all_bh_times, t_val, side="right") - 1
            if idx < 0:
                # Before any baseline hazard time: CIF = 0
                continue
            idx = min(idx, len(all_bh_times) - 1)
            survival[:, ti] = np.exp(-total_h_cumsum[:, idx])
            for k_idx in range(self.n_causes):
                cif[:, k_idx, ti] = cif_cumsum[k_idx, :, idx]

        return (
            torch.tensor(cif, dtype=torch.float32),
            torch.tensor(survival, dtype=torch.float32),
        )
