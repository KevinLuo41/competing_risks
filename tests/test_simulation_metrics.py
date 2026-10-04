#!/usr/bin/env python3

import later.unittest
import torch

from ..evaluation.simulation import compute_dist


class SimulationMetricsTest(later.unittest.TestCase):
    def test_reports_negative_implied_survival(self) -> None:
        cif = torch.tensor(
            [
                [[0.2, 0.7], [0.3, 0.5]],
                [[0.1, 0.4], [0.2, 0.4]],
            ]
        )
        native_survival = torch.tensor([[0.5, 0.0], [0.7, 0.2]])

        metrics = compute_dist(cif, native_survival)

        self.assertAlmostEqual(0.25, metrics["Implied_S_negative_fraction"])
        self.assertEqual(1, metrics["Implied_S_negative_count"])
        self.assertEqual(4, metrics["Implied_S_total_count"])
        self.assertEqual(1, metrics["Implied_S_negative_subject_count"])
        self.assertEqual(2, metrics["Implied_S_total_subject_count"])
        self.assertAlmostEqual(0.5, metrics["Implied_S_negative_subject_fraction"])
        self.assertEqual(1, metrics["Implied_S_below_tolerance_count"])
        self.assertEqual(1, metrics["Implied_S_below_tolerance_subject_count"])
        self.assertAlmostEqual(-0.2, metrics["Implied_S_min"], places=6)
        self.assertAlmostEqual(0.05, metrics["Dist"], places=6)

    def test_separates_roundoff_from_substantive_negative_survival(self) -> None:
        cif = torch.tensor([[[0.5], [0.50000012]]])
        native_survival = torch.zeros(1, 1)

        metrics = compute_dist(cif, native_survival)

        self.assertEqual(1, metrics["Implied_S_negative_count"])
        self.assertEqual(1, metrics["Implied_S_negative_subject_count"])
        self.assertEqual(0, metrics["Implied_S_below_tolerance_count"])
        self.assertEqual(0, metrics["Implied_S_below_tolerance_subject_count"])
