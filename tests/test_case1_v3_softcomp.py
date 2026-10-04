#!/usr/bin/env python3

from dataclasses import asdict

import later.unittest
import torch

from ..experiments.case1_v3.run import AffineExactSoftComp, K, P, SoftCompConfig


class Case1V3SoftCompTest(later.unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_exact_loss_uses_distinct_atom_event_and_censoring_terms(self) -> None:
        model = AffineExactSoftComp(P, K, initial_alpha=0.8)
        x = torch.zeros(3, P)
        y = torch.tensor([0.0, 1.25, 0.75])
        delta = torch.tensor([2, 1, 0])

        losses = model.exact_negative_log_likelihood(
            x,
            y,
            delta,
            reduction="none",
        )
        cif, survival = model.predict_cif(x, y)
        expected = torch.stack(
            [
                -torch.log(cif[0, 1]),
                -torch.log(model.alpha) - torch.log(cif[1, 0]) - torch.log(survival[1]),
                -torch.log(survival[2]),
            ]
        )
        incorrectly_continuous_atom_loss = (
            -torch.log(model.alpha) - torch.log(cif[0, 1]) - torch.log(survival[0])
        )

        self.assertTrue(torch.allclose(losses, expected, atol=1e-7))
        self.assertFalse(
            torch.isclose(losses[0], incorrectly_continuous_atom_loss, atol=1e-4)
        )

    def test_model_has_only_the_preregistered_affine_parameters(self) -> None:
        model = AffineExactSoftComp(P, K)

        self.assertEqual(K * (P + 1) + 1, model.parameter_count)
        self.assertGreater(float(model.alpha.item()), 0.0)
        self.assertEqual((K,), tuple(model.intercept.shape))
        self.assertEqual((K, P), tuple(model.beta.shape))

    def test_checkpoint_round_trip_preserves_standardization_and_predictions(
        self,
    ) -> None:
        generator = torch.Generator().manual_seed(8321)
        x = 1.5 + 2.0 * torch.randn(96, P, generator=generator)
        y = 0.2 + 3.0 * torch.rand(96, generator=generator)
        delta = torch.arange(96) % (K + 1)
        config = SoftCompConfig(max_iter=5)
        model = AffineExactSoftComp(
            P,
            K,
            initial_alpha=config.initial_alpha,
            atom_tolerance=config.atom_tolerance,
        )
        history = model.fit(x, y, delta, config)
        times = torch.tensor([0.0, 0.5, 2.0, 4.0])
        expected_cif, expected_survival = model.predict_cif_survival_grid(x[:7], times)

        restored = AffineExactSoftComp.load_from_checkpoint(model.to_checkpoint())
        actual_cif, actual_survival = restored.predict_cif_survival_grid(x[:7], times)

        self.assertGreater(len(history), 2)
        self.assertTrue(torch.isfinite(torch.tensor(history)).all().item())
        self.assertTrue(torch.equal(model.x_mean, restored.x_mean))
        self.assertTrue(torch.equal(model.x_scale, restored.x_scale))
        self.assertTrue(torch.equal(expected_cif, actual_cif))
        self.assertTrue(torch.equal(expected_survival, actual_survival))

    def test_config_records_exact_likelihood_without_augmentation_controls(
        self,
    ) -> None:
        payload = asdict(SoftCompConfig())

        self.assertEqual("lbfgs", payload["optimizer"])
        self.assertTrue(payload["standardize_x"])
        self.assertEqual(0.0, payload["l2_penalty"])
        self.assertNotIn("n_aug", payload)
        self.assertNotIn("aug_weight", payload)
        self.assertNotIn("brier_lambda", payload)
