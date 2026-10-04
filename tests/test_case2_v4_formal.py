#!/usr/bin/env python3

import argparse
import json
import tempfile
from pathlib import Path

import later.unittest
import torch

from ..crsoft_model import CRSoftNet
from ..data.case2_v4 import generate_parameters
from ..experiments.case2_v4.formal import (
    ALL_MODELS,
    build_config,
    compute_configuration_hash,
    compute_execution_hash,
    evaluate_development_gate,
    make_plot_cohort,
    validate_formal_gate,
)
from ..experiments.case2_v4.run import K, P, SoftCompConfig


def _args(phase: str, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "phase": phase,
        "n_replicates": None,
        "start_replicate": 0,
        "end_replicate": None,
        "train_seed_base": None,
        "test_seed_base": None,
        "model_seed": 0,
        "dgp_seed": 42,
        "cpu_threads": 8,
        "models": None,
        "hidden_dim": 32,
        "num_blocks": 1,
        "epochs": 1000,
        "lr": 1e-3,
        "weight_decay": 3e-3,
        "n_aug": 2,
        "aug_weight": 0.5,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _passing_gate_inputs() -> tuple[dict, dict, dict[str, float]]:
    metric_values = {
        "DeepHit": (0.020, 0.50, 0.060),
        "DSM": (0.010, 0.68, 0.050),
        "cs-Cox": (0.021, 0.52, 0.061),
        "NeuralFG": (0.004, 0.70, 0.040),
        "SoftComp": (0.005, 0.75, 0.045),
        "Fine-Gray": (0.019, 0.51, 0.059),
    }
    metrics = {
        replicate: {
            method: {
                "MSE_overall": values[0],
                "Ctd_overall": values[1],
                "IBS_overall": values[2],
            }
            for method, values in metric_values.items()
        }
        for replicate in range(4)
    }
    diagnostics = {
        "train_counts_censor_then_causes": [2500, 850, 825, 825],
        "test_counts_censor_then_causes": [500, 170, 165, 165],
        "train_censor_fraction": 0.5,
        "test_censor_fraction": 0.5,
        "event_time_at_limit_fraction": 0.0,
        "static_score_linear_r2_mean": 0.01,
        "static_score_linear_r2_max": 0.02,
    }
    manifests = {
        replicate: {"data_diagnostics": dict(diagnostics)} for replicate in range(4)
    }
    truth = {
        "minimum_cif": 0.0,
        "minimum_survival": 0.0,
        "minimum_cif_increment": 0.0,
        "maximum_conservation_error": 1e-7,
        "maximum_cif_at_zero": 0.0,
        "maximum_survival_error_at_zero": 0.0,
    }
    return metrics, manifests, truth


class Case2V4FormalTest(later.unittest.TestCase):
    def test_frozen_schedules_use_all_six_models_and_share_a_hash(self) -> None:
        development = build_config(_args("development"))
        formal = build_config(_args("formal"))

        self.assertEqual(4, development.n_replicates)
        self.assertEqual(50, formal.n_replicates)
        self.assertEqual(ALL_MODELS, development.models)
        self.assertEqual(ALL_MODELS, formal.models)
        self.assertEqual(
            compute_configuration_hash(development),
            compute_configuration_hash(formal),
        )
        self.assertNotEqual(
            compute_execution_hash(development),
            compute_execution_hash(formal),
        )

    def test_protocol_hash_changes_when_softcomp_setting_changes(self) -> None:
        original = build_config(_args("smoke"))
        changed = build_config(_args("smoke", hidden_dim=48))

        self.assertNotEqual(
            compute_configuration_hash(original),
            compute_configuration_hash(changed),
        )

    def test_softcomp_model_is_reduced_for_three_causes(self) -> None:
        config = SoftCompConfig()
        model = CRSoftNet(
            input_dim=P,
            num_causes=K,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
        )

        self.assertEqual(3, K)
        self.assertEqual(3, P)
        self.assertEqual(
            2371, sum(parameter.numel() for parameter in model.parameters())
        )

    def test_plot_cohort_is_deterministic_and_probability_coherent(self) -> None:
        parameters = generate_parameters()
        first = make_plot_cohort(parameters)
        second = make_plot_cohort(parameters)

        self.assertTrue(torch.equal(first.X, second.X))
        self.assertTrue(torch.equal(first.true_cif, second.true_cif))
        self.assertTrue(
            torch.allclose(
                first.true_cif.sum(dim=1) + first.true_survival,
                torch.ones_like(first.true_survival),
                atol=1e-6,
            )
        )

    def test_development_gate_accepts_only_the_frozen_success_criteria(self) -> None:
        metrics, manifests, truth = _passing_gate_inputs()
        passed = evaluate_development_gate(metrics, manifests, "hash", truth)
        self.assertTrue(passed["passed"])
        self.assertEqual(
            {"MSE_overall": 2, "Ctd_overall": 1, "IBS_overall": 2},
            passed["softcomp"]["ranks"],
        )

        metrics[0]["cs-Cox"]["Ctd_overall"] = 0.70
        failed = evaluate_development_gate(metrics, manifests, "hash", truth)
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["cscox"]["passed"])

    def test_formal_gate_requires_complete_matching_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            gate_path = Path(temp_dir) / "gate.json"
            with gate_path.open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "complete": True,
                        "passed": True,
                        "configuration_hash": "expected",
                    },
                    file,
                )
            validate_formal_gate(gate_path, "expected")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                validate_formal_gate(gate_path, "different")
