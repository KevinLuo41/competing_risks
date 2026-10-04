#!/usr/bin/env python3

import later.unittest
import torch

from ..evaluation.survival import compute_ctd, compute_ibs


class SurvivalMetricsTest(later.unittest.TestCase):
    def test_ctd_counts_concordance_and_ties(self) -> None:
        y_test = torch.tensor([1.0, 2.0, 3.0])
        delta_test = torch.tensor([1, 0, 2])
        times = torch.tensor([1.0, 2.0, 3.0])
        cif = torch.tensor(
            [
                [0.6, 0.7, 0.8],
                [0.4, 0.5, 0.6],
                [0.6, 0.7, 0.8],
            ]
        )

        result = compute_ctd(y_test, delta_test, cif, times, event_k=1)

        self.assertAlmostEqual(0.75, result)

    def test_ibs_matches_hand_calculation_without_censoring(self) -> None:
        y_test = torch.tensor([1.0, 3.0])
        delta_test = torch.tensor([1, 2])
        y_train = torch.tensor([1.0, 2.0, 3.0])
        delta_train = torch.tensor([1, 2, 1])
        times = torch.tensor([1.0, 2.0])
        cif = torch.tensor([[0.5, 0.6], [0.2, 0.3]])

        result = compute_ibs(
            y_test,
            delta_test,
            y_train,
            delta_train,
            cif,
            times,
            event_k=1,
        )

        self.assertAlmostEqual(0.135, result, places=6)
