#!/usr/bin/env python3

import unittest
import torch

from ..data.case3_v4 import (
    compute_cif,
    compute_static_logits,
    compute_time_rate,
    generate_data,
    generate_parameters,
    T_MAX,
)


class Case3V4Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_default_parameters_and_generated_data_have_expected_shapes(self) -> None:
        params = generate_parameters()
        data = generate_data(n=64, seed=123, params=params)

        self.assertEqual((3, 3), tuple(params["beta"].shape))
        self.assertEqual((3,), tuple(params["intercept"].shape))
        self.assertEqual((64, 3), tuple(data["X"].shape))
        self.assertEqual((64,), tuple(data["Y"].shape))
        self.assertEqual((64,), tuple(data["Delta"].shape))
        self.assertEqual((64,), tuple(data["T_true"].shape))
        self.assertEqual((64,), tuple(data["epsilon_true"].shape))
        self.assertEqual((3,), tuple(params["time_direction"].shape))
        self.assertEqual((64,), tuple(data["time_rate"].shape))

    def test_static_scores_follow_the_exact_linear_formula(self) -> None:
        params = generate_parameters(seed=42)
        x = torch.randn(128, 3, generator=torch.Generator().manual_seed(123))
        expected = params["intercept"].unsqueeze(0) + x @ params["beta"].T / (3**0.5)

        self.assertTrue(torch.equal(expected, compute_static_logits(x, params)))
        self.assertNotIn("quadratic_weight", params)
        self.assertNotIn("pairwise_weight", params)

    def test_time_rate_matches_folded_single_index_and_is_even(self) -> None:
        params = generate_parameters()
        x = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, -1.0, 1.0],
                [10.0, -10.0, 10.0],
            ]
        )
        projection = x @ params["time_direction"]
        expected = params["rate_floor"] + params[
            "rate_weight"
        ] * projection.abs().clamp(max=params["projection_cap"])
        rate = compute_time_rate(x, params)

        self.assertTrue(torch.equal(expected, rate))
        self.assertTrue(torch.equal(rate, compute_time_rate(-x, params)))
        self.assertEqual(0.12, round(float(rate.min().item()), 2))
        self.assertEqual(0.40, round(float(rate.max().item()), 2))
        self.assertLess(float(rate[0].item()), float(rate[1].item()))
        self.assertLess(float(rate[1].item()), float(rate[2].item()))

    def test_cif_is_finite_monotone_and_probability_coherent(self) -> None:
        params = generate_parameters()
        generator = torch.Generator().manual_seed(123)
        x = torch.randn(32, 3, generator=generator)
        times = torch.linspace(0.0, T_MAX, 241)
        expanded_x = x.unsqueeze(1).expand(-1, times.shape[0], -1).reshape(-1, 3)
        expanded_times = times.unsqueeze(0).expand(x.shape[0], -1).reshape(-1)
        cif, survival = compute_cif(expanded_x, expanded_times, params)
        cif_grid = cif.reshape(x.shape[0], times.shape[0], 3).transpose(1, 2)
        survival_grid = survival.reshape(x.shape[0], times.shape[0])

        self.assertTrue(bool(torch.isfinite(cif_grid).all().item()))
        self.assertTrue(bool(torch.isfinite(survival_grid).all().item()))
        self.assertTrue(bool((cif_grid >= 0.0).all().item()))
        self.assertTrue(bool((survival_grid >= 0.0).all().item()))
        self.assertTrue(
            torch.allclose(
                cif_grid.sum(dim=1) + survival_grid,
                torch.ones_like(survival_grid),
                atol=1e-6,
            )
        )
        self.assertTrue(torch.equal(cif_grid[:, :, 0], torch.zeros(32, 3)))
        self.assertTrue(torch.equal(survival_grid[:, 0], torch.ones(32)))
        self.assertTrue(
            bool((cif_grid[:, :, 1:] >= cif_grid[:, :, :-1] - 1e-6).all().item())
        )
        self.assertTrue(
            bool((survival_grid[:, 1:] <= survival_grid[:, :-1] + 1e-6).all().item())
        )

    def test_cif_matches_the_exact_folded_time_rate_formula(self) -> None:
        params = generate_parameters()
        x = torch.randn(8, 3, generator=torch.Generator().manual_seed(456))
        times = torch.linspace(0.0, 18.0, 8)
        cif, survival = compute_cif(x, times, params)
        static_logits = params["intercept"].unsqueeze(0) + x @ params["beta"].T / (
            3**0.5
        )
        projection = x @ params["time_direction"]
        time_rate = params["rate_floor"] + params[
            "rate_weight"
        ] * projection.abs().clamp(max=params["projection_cap"])
        cumulative_odds = torch.expm1(times * time_rate)
        risk = torch.exp(static_logits)
        denominator = 1.0 + cumulative_odds * risk.sum(dim=1)
        expected_cif = risk * cumulative_odds.unsqueeze(1) / denominator.unsqueeze(1)
        expected_survival = 1.0 / denominator

        self.assertTrue(torch.allclose(cif, expected_cif, atol=1e-6))
        self.assertTrue(torch.allclose(survival, expected_survival, atol=1e-6))

    def test_time_rate_has_low_linear_r_squared_and_wide_spread(self) -> None:
        params = generate_parameters()
        generator = torch.Generator().manual_seed(321)
        x_train = torch.randn(5000, 3, generator=generator)
        x_test = torch.randn(5000, 3, generator=generator)
        train_rate = compute_time_rate(x_train, params)
        test_rate = compute_time_rate(x_test, params)
        train_design = torch.cat([torch.ones(5000, 1), x_train], dim=1)
        test_design = torch.cat([torch.ones(5000, 1), x_test], dim=1)
        coefficients = torch.linalg.lstsq(train_design, train_rate).solution
        residual = test_rate - test_design @ coefficients
        centered = test_rate - test_rate.mean()
        r_squared = 1.0 - residual.square().sum() / centered.square().sum()
        sorted_rate = test_rate.sort().values
        q05 = sorted_rate[round((len(sorted_rate) - 1) * 0.05)]
        q95 = sorted_rate[round((len(sorted_rate) - 1) * 0.95)]

        self.assertLess(float(r_squared.item()), 0.02)
        self.assertGreater(float((q95 - q05).item()), 0.24)

    def test_time_interaction_has_strong_early_to_late_contrast(self) -> None:
        params = generate_parameters()
        x = torch.randn(5000, 3, generator=torch.Generator().manual_seed(987))
        rate = compute_time_rate(x, params)
        early = torch.log(torch.expm1(8.0 * rate))
        late = torch.log(torch.expm1(18.0 * rate))
        contrast = late - early

        self.assertGreater(float(contrast.std(unbiased=False).item()), 0.60)

    def test_fixed_seed_data_have_no_t_max_pileup_and_balanced_events(
        self,
    ) -> None:
        params = generate_parameters(seed=42)
        for seed in (2026, 210_000, 210_001, 220_000, 220_001):
            with self.subTest(seed=seed):
                data = generate_data(n=5000, seed=seed, params=params)
                t_event = data["T_true"]
                delta = data["Delta"]
                epsilon = data["epsilon_true"]

                self.assertTrue(bool((t_event < T_MAX).all().item()))
                self.assertEqual(0, int((t_event == T_MAX).sum().item()))
                self.assertTrue(bool(((delta >= 0) & (delta <= 3)).all().item()))
                self.assertTrue(bool(((epsilon >= 1) & (epsilon <= 3)).all().item()))
                censor_fraction = float((delta == 0).float().mean().item())
                self.assertGreater(censor_fraction, 0.45)
                self.assertLess(censor_fraction, 0.55)
                for cause in range(1, 4):
                    self.assertGreater(int((epsilon == cause).sum().item()), 1000)
                    self.assertGreater(int((delta == cause).sum().item()), 500)
