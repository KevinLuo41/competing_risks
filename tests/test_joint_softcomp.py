#!/usr/bin/env python3

import math
import unittest

import torch

from ..crsoft_model.joint_softcomp import (
    aalen_johansen_from_observed,
    JointSoftComp,
    observed_state_labels,
)

TAU = 20.0
HAZARDS = [0.10, 0.05]
# Censoring hazard -> fraction censored before TAU, P(C < min(T, TAU)).
CENSORING = {0.0: 0.0, 0.0386101: 0.2, 0.1507364: 0.5, 0.6000009: 0.8}


def _constant_hazard_observed_probs(
    times: torch.Tensor, hazards: list[float], censoring_hazard: float
) -> torch.Tensor:
    total = sum(hazards) + censoring_hazard
    at_risk = torch.exp(-total * times)
    columns = [at_risk] + [
        rate / total * (1.0 - at_risk) for rate in hazards + [censoring_hazard]
    ]
    return torch.stack(columns, dim=1).unsqueeze(0).double()


class JointSoftCompTest(unittest.TestCase):
    def test_labels_switch_at_observed_time(self) -> None:
        t = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        y = torch.tensor([2.0, 2.5])
        delta = torch.tensor([1, 0])

        labels = observed_state_labels(t, y, delta, num_causes=2)

        self.assertEqual(labels.tolist(), [[0, 1, 1], [0, 0, 3]])

    def test_aalen_johansen_recovers_true_cif_at_every_censoring_level(self) -> None:
        times = torch.linspace(0.0, TAU, 4001, dtype=torch.float64)
        total = sum(HAZARDS)
        for censoring_hazard, rho in CENSORING.items():
            rate = total + censoring_hazard
            censored = censoring_hazard / rate * (1.0 - math.exp(-rate * TAU))
            self.assertAlmostEqual(censored, rho, places=4)
            probs = _constant_hazard_observed_probs(times, HAZARDS, censoring_hazard)
            cif, survival = aalen_johansen_from_observed(probs, num_causes=2)
            for k, hazard in enumerate(HAZARDS):
                truth = hazard / total * (1.0 - torch.exp(-total * times))
                self.assertLess(float((cif[0, k] - truth).abs().max()), 2e-3)
            self.assertLess(
                float((survival[0] - torch.exp(-total * times)).abs().max()), 2e-3
            )

    def test_predictions_are_monotone_and_coherent(self) -> None:
        torch.manual_seed(0)
        model = JointSoftComp(input_dim=3, num_causes=2, grid_size=200)
        times = torch.tensor([0.0, 1.0, 2.5, 5.0, 10.0])

        cif, survival = model.predict_cif_survival_grid(torch.randn(7, 3), times)

        self.assertEqual(tuple(cif.shape), (7, 2, 5))
        self.assertTrue(bool((cif[:, :, 1:] >= cif[:, :, :-1] - 1e-7).all()))
        self.assertLess(float((cif.sum(dim=1) + survival - 1.0).abs().max()), 1e-5)
        self.assertEqual(float(cif[:, :, 0].abs().max()), 0.0)

    def test_training_times_are_drawn_from_event_times(self) -> None:
        torch.manual_seed(0)
        y = torch.tensor([1.0, 2.0, 3.0, 9.0, 9.0, 9.0])
        delta = torch.tensor([1, 2, 1, 0, 0, 0])
        model = JointSoftComp(input_dim=1, num_causes=2)

        model.fit(torch.zeros(6, 1), y, delta, epochs=1, verbose=False, device="cpu")

        self.assertEqual(sorted(model._time_pool.tolist()), [1.0, 2.0, 3.0])

    def test_fit_produces_finite_losses(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(64, 2)
        y = torch.rand(64) * 5.0
        delta = torch.randint(0, 3, (64,))
        model = JointSoftComp(input_dim=2, num_causes=2, grid_size=50)

        losses = model.fit(
            x, y, delta, epochs=2, batch_size=16, n_times=3, verbose=False, device="cpu"
        )

        self.assertTrue(all(math.isfinite(value) for value in losses))
        self.assertAlmostEqual(float(model.tau), float(y.max()), places=5)
