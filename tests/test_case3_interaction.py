#!/usr/bin/env python3

import unittest

import torch

from ..data.case3_interaction import (
    compute_cif,
    compute_time_slope,
    generate_parameters,
)


class Case3InteractionTest(unittest.TestCase):
    def test_time_slope_is_positive_bounded_and_subject_specific(self) -> None:
        params = generate_parameters(interaction_range=0.6)
        gamma = params["gamma"]
        x = torch.stack([-10.0 * gamma, 10.0 * gamma])

        slopes = compute_time_slope(x, params)

        self.assertGreaterEqual(float(slopes[0].item()), 0.1 - 1e-6)
        self.assertLessEqual(float(slopes[1].item()), 0.7 + 1e-6)
        self.assertLess(float(slopes[0].item()), float(slopes[1].item()))

    def test_cif_is_monotone_and_probability_coherent(self) -> None:
        params = generate_parameters(interaction_range=0.6)
        generator = torch.Generator().manual_seed(123)
        x = torch.randn(8, 5, generator=generator)
        trajectories = []
        for time in torch.linspace(0.0, 40.0, 81):
            cif, survival = compute_cif(x, time, params)
            trajectories.append(cif)
            total_probability = cif.sum(dim=1) + survival
            self.assertTrue(
                torch.allclose(total_probability, torch.ones_like(total_probability))
            )

        stacked = torch.stack(trajectories, dim=2)
        increments = stacked[:, :, 1:] - stacked[:, :, :-1]
        self.assertTrue(bool((increments >= -1e-6).all().item()))

    def test_invalid_interaction_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_parameters(interaction_range=0.0)
        with self.assertRaises(ValueError):
            generate_parameters(interaction_range=0.8)
