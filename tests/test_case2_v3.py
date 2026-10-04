#!/usr/bin/env python3

import unittest
import torch

from ..data.case2_v3 import (
    compute_cif,
    compute_static_logits,
    generate_parameters,
    T_MAX,
)


class Case2V3Test(unittest.TestCase):
    def test_cif_is_monotone_and_probability_coherent(self) -> None:
        params = generate_parameters()
        generator = torch.Generator().manual_seed(123)
        x = torch.randn(32, 8, generator=generator)
        trajectories = []
        survival_trajectories = []
        for time in torch.linspace(0.0, T_MAX, 241):
            cif, survival = compute_cif(x, time, params)
            trajectories.append(cif)
            survival_trajectories.append(survival)
            self.assertTrue(bool((cif >= 0.0).all().item()))
            self.assertTrue(bool((survival >= 0.0).all().item()))
            self.assertTrue(
                torch.allclose(
                    cif.sum(dim=1) + survival,
                    torch.ones_like(survival),
                    atol=1e-6,
                )
            )

        cif_grid = torch.stack(trajectories, dim=2)
        survival_grid = torch.stack(survival_trajectories, dim=1)
        self.assertTrue(torch.equal(cif_grid[:, :, 0], torch.zeros(32, 8)))
        self.assertTrue(torch.equal(survival_grid[:, 0], torch.ones(32)))
        self.assertTrue(
            bool((cif_grid[:, :, 1:] >= cif_grid[:, :, :-1] - 1e-6).all().item())
        )
        self.assertTrue(
            bool((survival_grid[:, 1:] <= survival_grid[:, :-1] + 1e-6).all().item())
        )

    def test_nonlinear_effects_dominate_the_linear_effect(self) -> None:
        params = generate_parameters()
        beta = params["beta"]

        self.assertGreaterEqual(float(beta.min().item()), 0.1)
        self.assertLessEqual(float(beta.max().item()), 0.3)
        self.assertAlmostEqual(1.5, float(params["quadratic_weight"].item()))
        self.assertAlmostEqual(1.0, float(params["pairwise_weight"].item()))

    def test_static_scores_are_not_well_approximated_by_linear_effects(self) -> None:
        params = generate_parameters()
        generator = torch.Generator().manual_seed(321)
        x_train = torch.randn(5000, 8, generator=generator)
        x_test = torch.randn(1000, 8, generator=generator)
        train_scores = compute_static_logits(x_train, params)
        test_scores = compute_static_logits(x_test, params)
        train_design = torch.cat([torch.ones(5000, 1), x_train], dim=1)
        test_design = torch.cat([torch.ones(1000, 1), x_test], dim=1)
        coefficients = torch.linalg.lstsq(train_design, train_scores).solution
        residual = test_scores - test_design @ coefficients
        centered = test_scores - test_scores.mean(dim=0, keepdim=True)
        r_squared = 1.0 - residual.square().sum(dim=0) / centered.square().sum(dim=0)

        self.assertLess(float(r_squared.mean().item()), 0.1)
        self.assertLess(float(r_squared.max().item()), 0.15)
