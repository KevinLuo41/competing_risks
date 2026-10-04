#!/usr/bin/env python3

import unittest
import torch

from ..evaluation.postprocess import enforce_cif_simplex


class PostprocessTest(unittest.TestCase):
    def test_simplex_scaling_preserves_monotonicity(self) -> None:
        cif = torch.tensor(
            [
                [
                    [0.10, 0.40, 0.70],
                    [0.20, 0.50, 0.60],
                ],
                [
                    [0.10, 0.20, 0.30],
                    [0.10, 0.30, 0.40],
                ],
            ]
        )

        projected = enforce_cif_simplex(cif)

        self.assertTrue(bool((projected[:, :, 1:] >= projected[:, :, :-1]).all()))
        self.assertTrue(bool((projected.sum(dim=1) <= 1.0 + 1e-6).all()))
        self.assertTrue(torch.equal(cif[1], projected[1]))
        self.assertAlmostEqual(1.0, float(projected[0, :, -1].sum().item()))
