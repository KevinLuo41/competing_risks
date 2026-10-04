#!/usr/bin/env python3

import unittest
import torch

from ..data.case3_v5 import (
    compute_acceleration,
    compute_cause_probabilities,
    compute_cif,
    compute_cumulative_hazard,
    compute_cumulative_hazard_crossing_time,
    compute_cumulative_hazard_derivative,
    compute_folded_index,
    compute_folded_projection,
    compute_hazard_crossing_time,
    compute_static_logits,
    compute_time_coefficients,
    CUMULATIVE_HAZARD_CROSSING_TIME,
    generate_data,
    generate_parameters,
    INTERACTION_EARLY_TIME,
    INTERACTION_LATE_TIME,
    INTERACTION_TRANSITION_TIME,
    T_MAX,
)


class Case3V5Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_default_parameters_and_generated_data_have_expected_shapes(self) -> None:
        params = generate_parameters()
        data = generate_data(n=64, seed=123, params=params)

        self.assertEqual((3,), tuple(params["router_intercept"].shape))
        self.assertEqual((3, 2), tuple(params["router_directions"].shape))
        self.assertEqual((64, 4), tuple(data["X"].shape))
        self.assertEqual((64,), tuple(data["Y"].shape))
        self.assertEqual((64,), tuple(data["Delta"].shape))
        self.assertEqual((64,), tuple(data["T_true"].shape))
        self.assertEqual((64,), tuple(data["epsilon_true"].shape))
        self.assertEqual((64, 3), tuple(data["cause_probabilities"].shape))
        for key in (
            "acceleration",
            "folded_projection",
            "folded_index",
            "hazard_rate_early",
            "hazard_rate_transition",
            "hazard_rate_late",
        ):
            self.assertEqual((64,), tuple(data[key].shape))

    def test_default_parameters_have_positive_time_coefficients(self) -> None:
        params = generate_parameters(seed=42)
        x = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]])
        linear, quadratic = compute_time_coefficients(x, params)

        self.assertTrue(torch.allclose(linear, torch.tensor([0.02, 1.58])))
        self.assertTrue(torch.allclose(quadratic, torch.tensor([2.40, 0.04])))
        self.assertEqual(16.0, float(params["time_scale"].item()))
        self.assertEqual(0.20, round(float(params["acceleration_weight"].item()), 2))
        self.assertEqual(0.0, float(params["folded_acceleration_weight"].item()))
        self.assertEqual(0.75, float(params["folded_acceleration_center"].item()))
        self.assertEqual(2.0, float(params["router_weight"].item()))
        self.assertTrue(torch.equal(params["router_intercept"], torch.zeros(3)))

    def test_cause_router_matches_folded_angular_softmax_formula(self) -> None:
        params = generate_parameters()
        x = torch.randn(128, 4, generator=torch.Generator().manual_seed(123))
        expected_logits = (
            params["router_intercept"].unsqueeze(0)
            + params["router_weight"]
            * (x[:, 1:3] @ params["router_directions"].T).abs()
        )
        logits = compute_static_logits(x, params)
        probabilities = compute_cause_probabilities(x, params)
        projections = x[:, 1:3] @ params["router_directions"].T

        self.assertTrue(torch.equal(expected_logits, logits))
        self.assertTrue(
            torch.equal(
                logits,
                params["router_weight"]
                * (torch.relu(projections) + torch.relu(-projections)),
            )
        )
        self.assertTrue(torch.allclose(probabilities, torch.softmax(logits, dim=1)))
        self.assertTrue(
            torch.allclose(
                probabilities.sum(dim=1),
                torch.ones(x.shape[0]),
                atol=1e-7,
            )
        )
        self.assertTrue(bool((probabilities > 0.0).all().item()))
        reflected = x.clone()
        reflected[:, 1:3] *= -1.0
        self.assertTrue(torch.equal(compute_static_logits(reflected, params), logits))

    def test_router_is_linearly_orthogonal_and_has_balanced_probability_spread(
        self,
    ) -> None:
        params = generate_parameters()
        generator = torch.Generator().manual_seed(321)
        x_train = torch.randn(5000, 4, generator=generator)
        x_test = torch.randn(5000, 4, generator=generator)
        train_logits = compute_static_logits(x_train, params)
        test_logits = compute_static_logits(x_test, params)
        train_design = torch.cat([torch.ones(len(x_train), 1), x_train], dim=1)
        test_design = torch.cat([torch.ones(len(x_test), 1), x_test], dim=1)
        coefficients = torch.linalg.lstsq(train_design, train_logits).solution
        residual = test_logits - test_design @ coefficients
        centered = test_logits - test_logits.mean(dim=0, keepdim=True)
        r_squared = 1.0 - residual.square().sum(dim=0) / centered.square().sum(dim=0)
        probabilities = compute_cause_probabilities(x_test, params)
        marginal = probabilities.mean(dim=0)

        self.assertLess(float(r_squared.max().item()), 0.02)
        self.assertGreater(float(probabilities.max().item()), 0.60)
        self.assertLess(float(probabilities.min().item()), 0.15)
        self.assertTrue(bool(((marginal > 0.30) & (marginal < 0.37)).all().item()))

    def test_folded_index_is_even_bounded_and_linearly_orthogonal(self) -> None:
        params = generate_parameters()
        generator = torch.Generator().manual_seed(456)
        x_train = torch.randn(5000, 4, generator=generator)
        x_test = torch.randn(5000, 4, generator=generator)
        train_index = compute_folded_index(x_train, params)
        test_index = compute_folded_index(x_test, params)
        train_design = torch.cat([torch.ones(len(x_train), 1), x_train], dim=1)
        test_design = torch.cat([torch.ones(len(x_test), 1), x_test], dim=1)
        coefficients = torch.linalg.lstsq(train_design, train_index).solution
        residual = test_index - test_design @ coefficients
        centered = test_index - test_index.mean()
        r_squared = 1.0 - residual.square().sum() / centered.square().sum()
        quantiles = torch.quantile(test_index, torch.tensor([0.05, 0.95]))

        self.assertTrue(
            torch.equal(
                compute_folded_projection(x_test, params),
                compute_folded_projection(-x_test, params),
            )
        )
        self.assertTrue(
            torch.equal(
                compute_folded_projection(x_test, params),
                compute_folded_index(x_test, params),
            )
        )
        self.assertGreaterEqual(float(test_index.min().item()), 0.0)
        self.assertLessEqual(float(test_index.max().item()), 2.0)
        self.assertLess(float(r_squared.item()), 0.02)
        self.assertGreater(float((quantiles[1] - quantiles[0]).item()), 1.80)

    def test_cumulative_hazard_matches_formula_and_has_positive_derivative(
        self,
    ) -> None:
        params = generate_parameters(folded_acceleration_weight=0.25)
        x = torch.randn(64, 4, generator=torch.Generator().manual_seed(789))
        times = torch.linspace(0.0, 32.0, len(x))
        folded_index = x[:, 3].abs().clamp(max=2.0)
        linear = 0.02 + 0.78 * folded_index
        quadratic = 0.04 + 1.18 * (2.0 - folded_index)
        scaled_time = times / 16.0
        expected_acceleration = torch.exp(0.20 * x[:, 0] + 0.25 * (folded_index - 0.75))
        expected = expected_acceleration * (
            linear * scaled_time + quadratic * scaled_time.square()
        )
        cumulative_hazard = compute_cumulative_hazard(x, times, params)
        derivative = compute_cumulative_hazard_derivative(x, times, params)

        self.assertTrue(
            torch.allclose(
                compute_acceleration(x, params), expected_acceleration, atol=1e-6
            )
        )
        self.assertTrue(torch.allclose(cumulative_hazard, expected, atol=1e-6))
        self.assertTrue(bool((derivative > 0.0).all().item()))

    def test_persistent_folded_acceleration_survives_at_reference_time(self) -> None:
        gamma = 0.25
        params = generate_parameters(folded_acceleration_weight=gamma)
        x = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]])
        crossing_time = compute_cumulative_hazard_crossing_time(params)
        reference_hazard = compute_cumulative_hazard(x, crossing_time, params)
        expected_ratio = torch.exp(torch.tensor(2.0 * gamma))

        self.assertTrue(
            torch.allclose(
                reference_hazard[1] / reference_hazard[0],
                expected_ratio,
                atol=1e-6,
            )
        )
        self.assertTrue(
            bool(
                (
                    compute_cumulative_hazard_derivative(
                        x, torch.tensor(INTERACTION_LATE_TIME), params
                    )
                    > 0.0
                )
                .all()
                .item()
            )
        )

    def test_folded_hazard_and_cumulative_effects_cross_at_analytic_times(
        self,
    ) -> None:
        params = generate_parameters()
        x = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]])
        early = compute_cumulative_hazard_derivative(
            x, torch.tensor(INTERACTION_EARLY_TIME), params
        )
        transition = compute_cumulative_hazard_derivative(
            x, torch.tensor(INTERACTION_TRANSITION_TIME), params
        )
        late = compute_cumulative_hazard_derivative(
            x, torch.tensor(INTERACTION_LATE_TIME), params
        )
        self.assertGreater(float(torch.log(early[1] / early[0]).item()), 0.25)
        self.assertTrue(torch.allclose(transition[0], transition[1], atol=1e-7))
        self.assertLess(float(torch.log(late[1] / late[0]).item()), -0.55)
        cumulative_crossing = compute_cumulative_hazard_crossing_time(params)
        cumulative_hazard = compute_cumulative_hazard(x, cumulative_crossing, params)
        self.assertTrue(torch.allclose(cumulative_hazard[0], cumulative_hazard[1]))
        self.assertTrue(
            torch.allclose(
                compute_hazard_crossing_time(params),
                torch.tensor(INTERACTION_TRANSITION_TIME),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                cumulative_crossing,
                torch.tensor(CUMULATIVE_HAZARD_CROSSING_TIME),
                atol=1e-6,
            )
        )

    def test_cif_is_finite_monotone_and_probability_coherent(self) -> None:
        params = generate_parameters()
        x = torch.randn(32, 4, generator=torch.Generator().manual_seed(987))
        times = torch.linspace(0.0, T_MAX, 301)
        expanded_x = x.unsqueeze(1).expand(-1, len(times), -1).reshape(-1, 4)
        expanded_times = times.unsqueeze(0).expand(len(x), -1).reshape(-1)
        cif, survival = compute_cif(expanded_x, expanded_times, params)
        cif_grid = cif.reshape(len(x), len(times), 3).transpose(1, 2)
        survival_grid = survival.reshape(len(x), len(times))

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

    def test_fixed_seed_data_have_no_t_max_pileup_and_balanced_events(
        self,
    ) -> None:
        params = generate_parameters(seed=42)
        for seed in (2026, 310_000, 310_001):
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
                latent_counts = []
                observed_counts = []
                for cause in range(1, 4):
                    latent_count = int((epsilon == cause).sum().item())
                    observed_count = int((delta == cause).sum().item())
                    latent_counts.append(latent_count)
                    observed_counts.append(observed_count)
                    self.assertGreater(latent_count, 1200)
                    self.assertGreater(observed_count, 500)
                self.assertLess(max(latent_counts) / min(latent_counts), 1.25)
                self.assertLess(max(observed_counts) / min(observed_counts), 1.25)
