#!/usr/bin/env python3

import later.unittest
import torch

from ..data.case2_v4 import (
    compute_cif,
    compute_static_logits,
    generate_data,
    generate_parameters,
    T_MAX,
)


class Case2V4Test(later.unittest.TestCase):
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

    def test_cif_is_monotone_and_probability_coherent(self) -> None:
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

    def test_parameters_encode_the_requested_strong_nonlinearity(self) -> None:
        params = generate_parameters()
        beta = params["beta"]

        self.assertGreaterEqual(float(beta.min().item()), 0.05)
        self.assertLessEqual(float(beta.max().item()), 0.15)
        self.assertTrue(
            torch.allclose(params["intercept"], torch.linspace(-4.1, -3.9, 3))
        )
        self.assertAlmostEqual(1.5, float(params["quadratic_weight"].item()))
        self.assertAlmostEqual(1.0, float(params["pairwise_weight"].item()))
        self.assertAlmostEqual(12.0, float(params["time_scale"].item()))

    def test_static_scores_have_low_held_out_linear_r_squared(self) -> None:
        params = generate_parameters()
        generator = torch.Generator().manual_seed(321)
        x_train = torch.randn(5000, 3, generator=generator)
        x_test = torch.randn(2000, 3, generator=generator)
        train_scores = compute_static_logits(x_train, params)
        test_scores = compute_static_logits(x_test, params)
        train_design = torch.cat([torch.ones(5000, 1), x_train], dim=1)
        test_design = torch.cat([torch.ones(2000, 1), x_test], dim=1)
        coefficients = torch.linalg.lstsq(train_design, train_scores).solution
        residual = test_scores - test_design @ coefficients
        centered = test_scores - test_scores.mean(dim=0, keepdim=True)
        r_squared = 1.0 - residual.square().sum(dim=0) / centered.square().sum(dim=0)

        self.assertLess(float(r_squared.mean().item()), 0.05)
        self.assertLess(float(r_squared.max().item()), 0.10)

    def test_generated_data_has_no_t_max_pileup_and_all_causes_are_observed(
        self,
    ) -> None:
        data = generate_data(n=5000, seed=2026, params=generate_parameters(seed=42))
        t_event = data["T_true"]
        delta = data["Delta"]
        epsilon = data["epsilon_true"]

        self.assertTrue(bool((t_event < T_MAX).all().item()))
        self.assertEqual(0, int((t_event == T_MAX).sum().item()))
        self.assertTrue(bool(((delta >= 0) & (delta <= 3)).all().item()))
        self.assertTrue(bool(((epsilon >= 1) & (epsilon <= 3)).all().item()))
        for cause in range(1, 4):
            self.assertGreater(int((epsilon == cause).sum().item()), 1000)
            self.assertGreater(int((delta == cause).sum().item()), 500)
