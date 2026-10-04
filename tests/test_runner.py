#!/usr/bin/env python3

import unittest

import torch

from ..crsoft_model import CRSoftEnsemble, CRSoftNet
from ..experiments.runner import compute_probability_diagnostics, ModelSpec


class ProbabilityDiagnosticsTest(unittest.TestCase):
    def test_computes_dist_and_tolerance_aware_aco_from_cached_cif(self) -> None:
        cached_cif = torch.tensor([[[0.6, 0.8], [0.3, 0.3]]])
        native_survival = torch.tensor([[0.1, 0.0]])
        spec = ModelSpec(
            name="model",
            cache_key="model",
            predict_survival=lambda model, X, times: (
                cached_cif.clone(),
                native_survival,
            ),
        )

        metrics = compute_probability_diagnostics(
            spec,
            object(),
            cached_cif,
            torch.zeros(1, 1),
            torch.tensor([1.0, 2.0]),
        )

        self.assertAlmostEqual(0.05, metrics["Dist"])
        self.assertEqual(1.0, metrics["ACO_points"])
        self.assertEqual(2.0, metrics["ACO_total_points"])
        self.assertAlmostEqual(0.5, metrics["ACO_fraction"])

    def test_rejects_checkpoint_and_cached_cif_mismatch_by_default(self) -> None:
        spec = ModelSpec(
            name="model",
            cache_key="model",
            predict_survival=lambda model, X, times: (
                torch.zeros(1, 2, 1),
                torch.ones(1, 1),
            ),
        )

        with self.assertRaises(AssertionError):
            compute_probability_diagnostics(
                spec,
                object(),
                torch.ones(1, 2, 1),
                torch.zeros(1, 1),
                torch.tensor([1.0]),
            )

    def test_can_audit_legacy_cached_cif_with_native_survival(self) -> None:
        cached_cif = torch.tensor([[[0.6, 0.8], [0.3, 0.3]]])
        spec = ModelSpec(
            name="legacy model",
            cache_key="legacy",
            predict_survival=lambda model, X, times: (
                torch.zeros_like(cached_cif),
                torch.tensor([[0.1, 0.0]]),
            ),
            allow_diagnostic_cif_mismatch=True,
        )

        metrics = compute_probability_diagnostics(
            spec,
            object(),
            cached_cif,
            torch.zeros(1, 1),
            torch.tensor([1.0, 2.0]),
        )

        self.assertAlmostEqual(0.05, metrics["Dist"])
        self.assertEqual(1.0, metrics["ACO_points"])


class CRSoftEnsemblePredictionTest(unittest.TestCase):
    def test_grid_prediction_returns_mean_native_survival(self) -> None:
        members = [
            self._constant_model(torch.tensor([0.0, 0.0])),
            self._constant_model(torch.tensor([1.0, -1.0])),
        ]
        ensemble = CRSoftEnsemble(members)
        X = torch.zeros(3, 1)
        times = torch.tensor([1.0, 2.0])

        cif, survival = ensemble.predict_cif_survival_grid(X, times)
        member_predictions = [
            member.predict_cif_survival_grid(X, times) for member in members
        ]
        expected_cif = torch.stack(
            [prediction[0] for prediction in member_predictions]
        ).mean(dim=0)
        expected_survival = torch.stack(
            [prediction[1] for prediction in member_predictions]
        ).mean(dim=0)

        torch.testing.assert_close(expected_cif, cif)
        torch.testing.assert_close(expected_survival, survival)
        torch.testing.assert_close(
            torch.ones_like(survival),
            cif.sum(dim=1) + survival,
        )

    def _constant_model(self, output_bias: torch.Tensor) -> CRSoftNet:
        model = CRSoftNet(
            input_dim=1,
            num_causes=2,
            hidden_dim=2,
            num_blocks=0,
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.output_layer.bias.copy_(output_bias)
        return model
