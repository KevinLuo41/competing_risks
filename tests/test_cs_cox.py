#!/usr/bin/env python3

import math

import unittest
import numpy as np
import pandas as pd
import torch

from ..baseline_models.cs_cox import CsCox


class _ConstantPartialHazardModel:
    def predict_log_partial_hazard(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(np.zeros(len(data), dtype=np.float64))


class CsCoxTest(unittest.TestCase):
    def test_predictions_are_probability_coherent(self) -> None:
        model = CsCox(n_causes=2)
        model.models = [_ConstantPartialHazardModel(), _ConstantPartialHazardModel()]
        model.baseline_hazards = [
            pd.DataFrame({"hazard": [0.2, 0.3]}, index=[1.0, 3.0]),
            pd.DataFrame({"hazard": [0.1, 0.4]}, index=[1.0, 2.0]),
        ]
        times = torch.tensor([0.5, 1.0, 2.0, 3.0])

        cif, survival = model.predict_cif_survival(torch.zeros(2, 1), times)

        self.assertTrue(torch.equal(cif[:, :, 0], torch.zeros(2, 2)))
        self.assertTrue(torch.equal(survival[:, 0], torch.ones(2)))
        self.assertTrue(
            torch.allclose(
                cif.sum(dim=1) + survival,
                torch.ones_like(survival),
                atol=1e-6,
            )
        )
        self.assertTrue(bool((cif[:, :, 1:] >= cif[:, :, :-1]).all().item()))
        self.assertTrue(bool((survival[:, 1:] <= survival[:, :-1]).all().item()))

        first_event_probability = -math.expm1(-0.3)
        expected_first_cif = torch.tensor(
            [
                first_event_probability * 2.0 / 3.0,
                first_event_probability / 3.0,
            ]
        )
        self.assertTrue(torch.allclose(cif[0, :, 1], expected_first_cif, atol=1e-6))
        self.assertAlmostEqual(math.exp(-0.3), float(survival[0, 1]), places=6)
