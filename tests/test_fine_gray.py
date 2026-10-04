#!/usr/bin/env python3

import math

import unittest
import numpy as np
import torch

from ..baseline_models import FineGray
from ..data import case2_v3


class FineGrayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.zeros(5, 1)
        self.times = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        self.events = torch.tensor([1, 2, 0, 1, 0])

    def test_prediction_matches_hand_calculated_weighted_risk_sets(self) -> None:
        model = FineGray(n_causes=2)
        model.fit(self.x, self.times, self.events)

        evaluation_times = torch.tensor([0.5, 1.0, 3.9, 4.0, 6.0])
        cif = model.predict_cif(self.x[:1], evaluation_times)

        expected_hazard = torch.tensor(
            [0.0, 1.0 / 5.0, 1.0 / 5.0, 23.0 / 40.0, 23.0 / 40.0]
        )
        expected_cif = 1.0 - torch.exp(-expected_hazard)
        self.assertTrue(torch.allclose(cif[0, 0], expected_cif, atol=1e-6))
        self.assertEqual((1, 2, 5), tuple(cif.shape))
        self.assertTrue(torch.all(cif[:, :, 1:] >= cif[:, :, :-1]))

    def test_checkpoint_round_trip_preserves_predictions(self) -> None:
        x = torch.tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0], [1.5]])
        times = torch.tensor([1.0, 2.0, 2.5, 3.0, 4.0, 5.0])
        events = torch.tensor([1, 2, 0, 1, 2, 0])
        model = FineGray(n_causes=2)
        model.fit(x, times, events)
        evaluation_times = torch.linspace(0.0, 6.0, 13)

        restored = FineGray.load_from_checkpoint(model.to_checkpoint())

        expected = model.predict_cif(x, evaluation_times)
        actual = restored.predict_cif(x, evaluation_times)
        self.assertTrue(torch.equal(expected, actual))

    def test_censoring_ties_follow_cmprsk_risk_set_convention(self) -> None:
        model = FineGray(n_causes=2)
        model.fit(
            torch.zeros(5, 1),
            torch.tensor([1.0, 2.0, 2.0, 3.0, 4.0]),
            torch.tensor([2, 0, 1, 1, 0]),
        )

        cif = model.predict_cif(torch.zeros(1, 1), torch.tensor([3.0]))

        expected_cumulative_hazard = 31.0 / 55.0
        self.assertAlmostEqual(
            1.0 - math.exp(-expected_cumulative_hazard),
            float(cif[0, 0, 0]),
            places=6,
        )

    def test_coefficients_match_comprisk_reference_with_cross_type_ties(
        self,
    ) -> None:
        generator = np.random.default_rng(20260815)
        x = generator.normal(size=(180, 3))
        x[:, 0] = 10.0 + 2.5 * x[:, 0]
        x[:, 1] = -3.0 + 0.7 * x[:, 1]
        x[:, 2] = generator.binomial(1, 0.4, size=180)
        times = generator.integers(1, 10, size=180).astype(np.float64)
        events = generator.choice(
            [0, 1, 2],
            size=180,
            p=[0.32, 0.39, 0.29],
        ).astype(np.int64)
        times[:12] = [2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8]
        events[:12] = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        expected = np.array(
            [
                [0.08286474459156998, 0.021835747705123438, -0.3796183361056758],
                [-0.16184444118208724, 0.3918491560043258, 1.0941909536666155],
            ]
        )
        model = FineGray(n_causes=2)

        model.fit(
            torch.tensor(x),
            torch.tensor(times),
            torch.tensor(events),
        )

        assert model.feature_scale is not None
        actual = np.stack(
            [cause.coefficients / model.feature_scale for cause in model.models]
        )
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0.0)

    def test_survival_is_unclamped_implied_survival(self) -> None:
        model = FineGray(n_causes=2)
        model.fit(self.x, self.times, self.events)

        cif, survival = model.predict_cif_survival(
            self.x[:2], torch.tensor([1.0, 4.0, 6.0])
        )

        self.assertTrue(torch.equal(survival, 1.0 - cif.sum(dim=1)))
        self.assertTrue(torch.isfinite(survival).all())

    def test_missing_cause_is_rejected(self) -> None:
        model = FineGray(n_causes=3)

        with self.assertRaisesRegex(ValueError, "cause 3 has no observed events"):
            model.fit(self.x, self.times, self.events)

    def test_invalid_hyperparameters_are_rejected(self) -> None:
        for constructor in (
            lambda: FineGray(n_causes=0),
            lambda: FineGray(n_causes=2, penalizer=-1.0),
            lambda: FineGray(n_causes=2, max_iter=0),
            lambda: FineGray(n_causes=2, tolerance=math.nan),
        ):
            with self.subTest(constructor=constructor):
                with self.assertRaises(ValueError):
                    constructor()

    def test_nonconverged_fit_is_rejected(self) -> None:
        model = FineGray(n_causes=2, max_iter=1, tolerance=1e-12)
        x = torch.tensor([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]])
        times = torch.tensor([1.0, 2.0, 2.5, 3.0, 4.0, 5.0])
        events = torch.tensor([1, 2, 0, 1, 2, 0])

        with self.assertRaisesRegex(RuntimeError, "failed to converge"):
            model.fit(x, times, events)

    def test_formal_replicate_zero_converges_for_every_cause(self) -> None:
        parameters = case2_v3.generate_parameters(K=8, p=8, seed=42)
        generated = case2_v3.generate_data(
            n=5000,
            K=8,
            p=8,
            censor_rate=0.5,
            seed=50_000,
            params=parameters,
        )
        model = FineGray(n_causes=8)

        model.fit(
            generated["X"][500:],
            generated["Y"][500:],
            generated["Delta"][500:],
        )

        self.assertTrue(all(cause_model.converged for cause_model in model.models))
        self.assertTrue(
            all(cause_model.score_norm < 1e-7 for cause_model in model.models)
        )
