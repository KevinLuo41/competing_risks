#!/usr/bin/env python3

import unittest
import torch

from ..data.case1_v3 import (
    compute_cause_probabilities,
    compute_cif,
    compute_initial_event_probability,
    compute_static_logits,
    DEFAULT_BETA_VALUES,
    DEFAULT_INTERCEPTS,
    generate_data,
    generate_parameters,
    K,
    P,
    sample_event_times,
    T_MAX,
)


class Case1V3Test(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_default_parameters_match_frozen_affine_candidate(self) -> None:
        params = generate_parameters()
        expected_beta = torch.tensor(
            [
                [-0.7, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.7, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        self.assertEqual(3, K)
        self.assertEqual(5, P)
        self.assertEqual((-0.7, 0.0, 0.7), DEFAULT_BETA_VALUES)
        self.assertTrue(torch.equal(params["beta"], expected_beta))
        self.assertTrue(
            torch.equal(params["intercept"], torch.tensor(DEFAULT_INTERCEPTS))
        )
        self.assertEqual((-16.0, -15.78, -16.0), DEFAULT_INTERCEPTS)
        self.assertAlmostEqual(1.15, float(params["alpha"].item()))
        self.assertEqual({"alpha", "beta", "intercept"}, set(params))

    def test_cause_probabilities_are_sign_symmetric_with_cause_two_centered(
        self,
    ) -> None:
        params = generate_parameters()
        x = torch.zeros(3, P)
        x[:, 0] = torch.tensor([-2.0, 0.0, 2.0])
        probabilities = compute_cause_probabilities(x, params)

        self.assertTrue(
            torch.allclose(
                probabilities[0],
                probabilities[2].flip(0),
                atol=1e-7,
            )
        )
        self.assertAlmostEqual(
            float(probabilities[1, 0].item()),
            float(probabilities[1, 2].item()),
            places=7,
        )
        self.assertGreater(
            float(probabilities[1, 1].item()),
            float(probabilities[1, 0].item()),
        )
        self.assertGreater(
            float(probabilities[1, 1].item()),
            float(probabilities[0, 1].item()),
        )

    def test_static_scores_and_mu_are_strictly_affine(self) -> None:
        params = generate_parameters()
        x = 0.25 * torch.randn(128, P, generator=torch.Generator().manual_seed(123))
        times = torch.linspace(0.0, 8.0, len(x))
        expected_static = params["intercept"].unsqueeze(0) + x @ params["beta"].T
        cif, survival = compute_cif(x, times, params)
        expected_mu = expected_static + times.unsqueeze(1) * params["alpha"]
        expected_full_logits = torch.cat([torch.zeros(len(x), 1), expected_mu], dim=1)
        expected_probabilities = torch.softmax(expected_full_logits, dim=1)

        self.assertTrue(torch.equal(compute_static_logits(x, params), expected_static))
        self.assertTrue(torch.allclose(cif, expected_probabilities[:, 1:], atol=1e-6))
        self.assertTrue(
            torch.allclose(survival, expected_probabilities[:, 0], atol=1e-6)
        )

    def test_time_zero_is_a_coherent_positive_atom(self) -> None:
        params = generate_parameters()
        x = torch.zeros(4, P)
        before_cif, before_survival = compute_cif(x, torch.tensor(-1.0), params)
        zero_cif, zero_survival = compute_cif(x, torch.tensor(0.0), params)
        initial_probability = compute_initial_event_probability(x, params)

        self.assertTrue(torch.equal(before_cif, torch.zeros_like(before_cif)))
        self.assertTrue(torch.equal(before_survival, torch.ones_like(before_survival)))
        self.assertTrue(bool((zero_cif > 0.0).all().item()))
        self.assertTrue(bool((zero_survival < 1.0).all().item()))
        self.assertTrue(
            torch.allclose(zero_cif.sum(dim=1), initial_probability, atol=1e-12)
        )
        self.assertTrue(
            torch.allclose(
                zero_cif.sum(dim=1) + zero_survival,
                torch.ones(len(x)),
                atol=1e-7,
            )
        )
        self.assertLess(float(initial_probability.max().item()), 2e-6)

    def test_cif_is_finite_monotone_and_probability_coherent(self) -> None:
        params = generate_parameters()
        x = torch.randn(32, P, generator=torch.Generator().manual_seed(456))
        times = torch.linspace(0.0, T_MAX, 321)
        expanded_x = x.unsqueeze(1).expand(-1, len(times), -1).reshape(-1, P)
        expanded_times = times.unsqueeze(0).expand(len(x), -1).reshape(-1)
        cif, survival = compute_cif(expanded_x, expanded_times, params)
        cif_grid = cif.reshape(len(x), len(times), K).transpose(1, 2)
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
        self.assertTrue(
            bool((cif_grid[:, :, 1:] >= cif_grid[:, :, :-1] - 1e-6).all().item())
        )
        self.assertTrue(
            bool((survival_grid[:, 1:] <= survival_grid[:, :-1] + 1e-6).all().item())
        )

    def test_analytic_inverse_handles_time_zero_atom_and_positive_times(self) -> None:
        params = generate_parameters()
        x = torch.zeros(6, P)
        initial = compute_initial_event_probability(x, params)
        uniform = torch.tensor(
            [
                float(initial[0].item()) * 0.5,
                float(initial[1].item()),
                float(initial[2].item()) * 1.5,
                0.25,
                0.50,
                0.99,
            ]
        )
        event_times = sample_event_times(x, uniform, params)
        positive = event_times > 0.0
        cif, _ = compute_cif(x[positive], event_times[positive], params)

        self.assertTrue(torch.equal(event_times[:2], torch.zeros(2)))
        self.assertTrue(bool((event_times[2:] > 0.0).all().item()))
        self.assertTrue(
            torch.allclose(cif.sum(dim=1), uniform[positive], atol=2e-6, rtol=2e-6)
        )

    def test_cause_probabilities_are_time_invariant(self) -> None:
        params = generate_parameters()
        x = torch.randn(256, P, generator=torch.Generator().manual_seed(789))
        expected = compute_cause_probabilities(x, params)
        for time in (0.0, 3.0, 12.0, 30.0):
            with self.subTest(time=time):
                cif, _ = compute_cif(x, torch.tensor(time), params)
                recovered = cif / cif.sum(dim=1, keepdim=True)
                self.assertTrue(torch.allclose(recovered, expected, atol=1e-6))

    def test_fixed_seed_data_are_reproducible_and_audit_atom_mass(self) -> None:
        params = generate_parameters(seed=42)
        first = generate_data(n=5000, seed=510_000, params=params)
        repeated = generate_data(n=5000, seed=510_000, params=params)
        different = generate_data(n=5000, seed=510_001, params=params)

        for key in (
            "X",
            "Y",
            "Delta",
            "T_true",
            "epsilon_true",
            "initial_event_probability",
            "is_time_zero_atom",
        ):
            self.assertTrue(torch.equal(first[key], repeated[key]))
        self.assertFalse(torch.equal(first["X"], different["X"]))
        self.assertTrue(bool(torch.isfinite(first["T_true"]).all().item()))
        self.assertTrue(bool((first["T_true"] >= 0.0).all().item()))
        self.assertEqual(0, int((first["T_true"] == T_MAX).sum().item()))
        self.assertTrue(torch.equal(first["is_time_zero_atom"], first["T_true"] == 0.0))
        self.assertTrue(
            torch.allclose(
                first["initial_event_probability"],
                compute_initial_event_probability(first["X"], params),
            )
        )
        initial_probability = first["initial_event_probability"]
        self.assertLess(float(initial_probability.mean().item()), 2e-6)
        self.assertLess(float(torch.quantile(initial_probability, 0.99).item()), 5e-6)
        self.assertLess(float(initial_probability.max().item()), 1e-4)

        delta = first["Delta"]
        epsilon = first["epsilon_true"]
        censor_fraction = float((delta == 0).float().mean().item())
        latent_counts = [int((epsilon == cause).sum().item()) for cause in range(1, 4)]
        observed_counts = [int((delta == cause).sum().item()) for cause in range(1, 4)]
        self.assertEqual([1678, 1679, 1643], latent_counts)
        self.assertEqual([858, 839, 824], observed_counts)
        self.assertEqual(0, int(first["is_time_zero_atom"].sum().item()))
        self.assertAlmostEqual(0.4958, censor_fraction, places=6)

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires K=3 and p=5"):
            generate_parameters(K=2)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            generate_parameters(alpha=0.0)
        with self.assertRaisesRegex(ValueError, "must contain 3 or 15"):
            generate_parameters(beta_values=(1.0, 2.0))
        with self.assertRaisesRegex(ValueError, "requires K=3 and p=5"):
            generate_data(n=10, p=4)
        with self.assertRaisesRegex(ValueError, "censor_rate"):
            generate_data(n=10, censor_rate=1.0)
        with self.assertRaisesRegex(ValueError, "one value per subject"):
            sample_event_times(torch.zeros(2, P), torch.zeros(3), generate_parameters())
        with self.assertRaisesRegex(ValueError, "must lie in"):
            sample_event_times(
                torch.zeros(2, P), torch.tensor([0.5, 1.0]), generate_parameters()
            )
