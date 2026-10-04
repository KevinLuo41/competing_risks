#!/usr/bin/env python3

import later.unittest
import torch

from ..baseline_models.dsm import DSM, DSMNet


class DSMTest(later.unittest.TestCase):
    def test_forward_all_matches_individual_cause_heads(self) -> None:
        torch.manual_seed(5)
        net = DSMNet(
            input_dim=3,
            n_causes=4,
            k=3,
            layers=[5],
            distribution="Weibull",
        ).double()
        x = torch.randn(7, 3, dtype=torch.double)

        shapes, scales, logits = net.forward_all(x)
        individual = [net(x, str(cause)) for cause in range(1, 5)]

        self.assertTrue(
            torch.allclose(shapes, torch.stack([value[0] for value in individual], 1))
        )
        self.assertTrue(
            torch.allclose(scales, torch.stack([value[1] for value in individual], 1))
        )
        self.assertTrue(
            torch.allclose(logits, torch.stack([value[2] for value in individual], 1))
        )

    def test_predict_cif_survival_matches_mixture_definition(self) -> None:
        torch.manual_seed(7)
        model = DSM(
            n_causes=2,
            n_features=3,
            k=2,
            layers=[4],
            distribution="Weibull",
        )
        model.net = DSMNet(
            input_dim=3,
            n_causes=2,
            k=2,
            layers=[4],
            distribution="Weibull",
        ).double()
        x = torch.tensor(
            [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]],
            dtype=torch.float32,
        )
        times = torch.tensor([0.2, 1.0, 2.0])

        cif, survival = model.predict_cif_survival(x, times)

        expected_cause_survival = []
        with torch.no_grad():
            for cause in ("1", "2"):
                shape, scale, logits = model.net(x.double(), cause)
                weights = torch.softmax(logits, dim=1)
                component_survival = torch.exp(
                    -torch.pow(
                        torch.exp(scale).unsqueeze(2) * times.double().view(1, 1, -1),
                        torch.exp(shape).unsqueeze(2),
                    )
                )
                expected_cause_survival.append(
                    (weights.unsqueeze(2) * component_survival).sum(dim=1)
                )
        expected = torch.stack(expected_cause_survival, dim=1).float()

        self.assertTrue(torch.allclose(cif, 1.0 - expected, atol=1e-6))
        self.assertTrue(torch.allclose(survival, expected.prod(dim=1), atol=1e-6))

    def test_fit_returns_finite_epoch_history(self) -> None:
        generator = torch.Generator().manual_seed(9)
        x = torch.randn(24, 3, generator=generator)
        times = torch.rand(24, generator=generator) + 0.2
        events = torch.tensor([0, 1, 2] * 8)
        model = DSM(
            n_causes=2,
            n_features=3,
            k=2,
            layers=[4],
            distribution="Weibull",
        )

        history = model.fit(
            x[4:],
            times[4:],
            events[4:],
            x[:4],
            times[:4],
            events[:4],
            epochs=2,
            batch_size=8,
            patience=2,
            device="cpu",
        )

        self.assertEqual(2, len(history))
        self.assertTrue(torch.isfinite(torch.tensor(history)).all().item())
