#!/usr/bin/env python3

import argparse
import json
import tempfile
from pathlib import Path

import unittest
import torch

from ..experiments.case1_v3.formal import (
    _fine_gray_convergence_payload,
    _frozen_protocol_payload,
    ALL_MODELS,
    build_config,
    compute_configuration_hash,
    compute_execution_hash,
    DEVELOPMENT_TEST_SEED_BASE,
    DEVELOPMENT_TRAIN_SEED_BASE,
    evaluate_development_gate,
    FORMAL_TEST_SEED_BASE,
    FORMAL_TRAIN_SEED_BASE,
    generate_frozen_parameters,
    make_plot_cohort,
    PLOT_COHORT_SEED,
    PLOT_TIME_MAX,
    truth_diagnostics,
    validate_formal_gate,
)
from ..experiments.case1_v3.run import K, P

EXPECTED_CONFIGURATION_HASH = (
    "e672a48c96f02398ae5cf0c2e768a10765c0f04d94dbc10416af917863231b3b"
)


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
        "softcomp_optimizer": "lbfgs",
        "softcomp_standardize_x": True,
        "softcomp_initial_alpha": 1.0,
        "softcomp_max_iter": 100,
        "softcomp_lr": 1.0,
        "softcomp_l2_penalty": 0.0,
        "softcomp_tolerance_grad": 1e-7,
        "softcomp_tolerance_change": 1e-9,
        "softcomp_history_size": 50,
        "softcomp_atom_tolerance": 1e-8,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _passing_gate_inputs() -> tuple[dict, dict, dict[str, float]]:
    metric_values = {
        "DeepHit": (0.020, 0.65, 0.060),
        "DSM": (0.010, 0.66, 0.050),
        "cs-Cox": (0.021, 0.60, 0.061),
        "NeuralFG": (0.004, 0.70, 0.040),
        "SoftComp": (0.005, 0.75, 0.045),
        "Fine-Gray": (0.019, 0.55, 0.059),
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
    for replicate in range(4):
        metrics[replicate]["Fine-Gray"].update(
            {
                "Implied_S_below_tolerance_count": 500,
                "Implied_S_below_tolerance_subject_count": 150,
                "Implied_S_min": -0.05,
            }
        )
    diagnostics = {
        "train_counts_censor_then_causes": [2500, 800, 850, 850],
        "test_counts_censor_then_causes": [500, 160, 170, 170],
        "train_censor_fraction": 0.5,
        "test_censor_fraction": 0.5,
        "event_time_at_limit_fraction": 0.0,
        "evaluation_time_max": 24.0,
        "evaluation_grid_critical_time_max_abs_error": 0.0,
        "affine_mu_r2_min": 1.0,
        "affine_mu_r2_min_nonconstant": 1.0,
        "affine_mu_constant_cause_count": 1,
        "affine_mu_nonconstant_cause_count": 2,
        "affine_mu_constant_cause_max_abs_error": 0.0,
        "static_affine_max_abs_error": 0.0,
        "affine_mu_reconstruction_max_abs_error": 1e-12,
        "initial_event_probability_mean": 5e-7,
        "initial_event_probability_p99": 1e-6,
        "initial_event_probability_max": 1e-5,
        "time_zero_atom_count_discrepancy": 0.2,
        "time_zero_atom_count_allowed_discrepancy": 3.0,
    }
    manifests = {
        replicate: {
            "data_diagnostics": dict(diagnostics),
            "fine_gray_convergence": {
                "cause_fit_count": 3,
                "converged_count": 3,
                "all_converged": True,
            },
        }
        for replicate in range(4)
    }
    truth = {
        "minimum_cif": 0.0,
        "minimum_survival": 0.0,
        "minimum_cif_increment": 0.0,
        "maximum_conservation_error": 1e-7,
        "maximum_time_zero_mass_error": 1e-12,
        "static_affine_max_abs_error": 0.0,
        "affine_mu_r2_min": 1.0,
        "affine_mu_r2_min_nonconstant": 1.0,
        "affine_mu_constant_cause_count": 1.0,
        "affine_mu_nonconstant_cause_count": 2.0,
        "affine_mu_constant_cause_max_abs_error": 0.0,
        "affine_mu_reconstruction_max_abs_error": 1e-12,
        "initial_event_probability_mean": 5e-7,
        "initial_event_probability_p99": 1e-6,
        "initial_event_probability_max": 1e-5,
    }
    return metrics, manifests, truth


def _force_cscox_to_ctd_rank_one(
    metrics: dict,
    manifests: dict,
    truth: dict[str, float],
) -> None:
    del manifests, truth
    competing_ctd = {
        "SoftComp": 0.59,
        "NeuralFG": 0.58,
        "DSM": 0.57,
        "DeepHit": 0.56,
        "Fine-Gray": 0.55,
    }
    for replicate in range(4):
        for method, value in competing_ctd.items():
            metrics[replicate][method]["Ctd_overall"] = value


class Case1V3FormalTest(unittest.TestCase):
    def test_frozen_schedules_use_all_six_models_and_share_a_hash(self) -> None:
        development = build_config(_args("development"))
        formal = build_config(_args("formal"))

        self.assertEqual(4, development.n_replicates)
        self.assertEqual(50, formal.n_replicates)
        self.assertEqual(ALL_MODELS, development.models)
        self.assertEqual(ALL_MODELS, formal.models)
        self.assertEqual(511_000, DEVELOPMENT_TRAIN_SEED_BASE)
        self.assertEqual(521_000, DEVELOPMENT_TEST_SEED_BASE)
        self.assertEqual(530_000, FORMAL_TRAIN_SEED_BASE)
        self.assertEqual(540_000, FORMAL_TEST_SEED_BASE)
        self.assertEqual(550_000, PLOT_COHORT_SEED)
        self.assertEqual(
            EXPECTED_CONFIGURATION_HASH,
            compute_configuration_hash(development),
        )
        self.assertEqual(
            compute_configuration_hash(development),
            compute_configuration_hash(formal),
        )
        self.assertNotEqual(
            compute_execution_hash(development),
            compute_execution_hash(formal),
        )

    def test_protocol_hash_changes_with_softcomp_setting(self) -> None:
        original = build_config(_args("smoke"))
        changed = build_config(_args("smoke", softcomp_max_iter=101))

        self.assertNotEqual(
            compute_configuration_hash(original),
            compute_configuration_hash(changed),
        )

    def test_protocol_records_affine_truth_atom_and_gate(self) -> None:
        protocol = _frozen_protocol_payload(build_config(_args("development")))
        dgp = protocol["dgp"]
        gate = protocol["development_gate"]
        amendment = protocol["development_gate_amendment"]
        parameters = generate_frozen_parameters()

        self.assertEqual([-16.0, -15.78, -16.0], dgp["intercepts"])
        self.assertEqual(
            [
                [-0.7, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.7, 0.0, 0.0, 0.0, 0.0],
            ],
            dgp["beta"],
        )
        self.assertNotIn("common_weight", dgp)
        self.assertNotIn("contrast_scales", dgp)
        self.assertNotIn("shared_direction", dgp)
        self.assertEqual(1.15, dgp["time_slope"])
        self.assertEqual([1.5, 8.0, 16.0, 24.0], gate["evaluation_grid_must_cover"])
        self.assertEqual(0.63, gate["maximum_cscox_mean_ctd"])
        self.assertEqual(0.65, gate["maximum_cscox_replicate_ctd"])
        self.assertEqual(2, gate["minimum_cscox_mean_ctd_rank"])
        self.assertEqual(510_000, amendment["superseded_schedule"]["train_seed_base"])
        self.assertEqual(520_000, amendment["superseded_schedule"]["test_seed_base"])
        self.assertEqual(0.625, amendment["superseded_cscox_gate"]["maximum_mean_ctd"])
        self.assertEqual(3, amendment["superseded_cscox_gate"]["minimum_mean_ctd_rank"])
        self.assertFalse(amendment["formal_schedule_changed"])
        self.assertEqual(500, gate["minimum_fine_gray_aco_points_per_replicate"])
        self.assertEqual(
            150,
            gate["minimum_fine_gray_aco_subjects_per_replicate"],
        )
        self.assertEqual(
            -0.05,
            gate["maximum_fine_gray_minimum_implied_survival"],
        )
        self.assertIn("atom_count_discrepancy_floor", gate)
        self.assertNotIn("atom_fraction_discrepancy_floor", gate)
        self.assertEqual((3, 5), tuple(parameters["beta"].shape))
        self.assertTrue(torch.equal(parameters["beta"], torch.tensor(dgp["beta"])))
        self.assertTrue(
            torch.equal(parameters["intercept"], torch.tensor(dgp["intercepts"]))
        )
        self.assertAlmostEqual(1.15, float(parameters["alpha"].item()))

    def test_plot_cohort_is_deterministic_and_probability_coherent(self) -> None:
        parameters = generate_frozen_parameters()
        first = make_plot_cohort(parameters)
        second = make_plot_cohort(parameters)

        self.assertTrue(torch.equal(first.X, second.X))
        self.assertTrue(torch.equal(first.true_cif, second.true_cif))
        self.assertEqual((5, P), tuple(first.X.shape))
        self.assertEqual(PLOT_TIME_MAX, float(first.times[-1].item()))
        self.assertTrue(bool((first.true_cif[:, :, 0].sum(dim=1) > 0.0).all()))
        self.assertTrue(
            torch.allclose(
                first.true_cif.sum(dim=1) + first.true_survival,
                torch.ones_like(first.true_survival),
                atol=1e-6,
            )
        )

    def test_truth_diagnostics_verify_legality_affinity_and_atom(self) -> None:
        diagnostics = truth_diagnostics(generate_frozen_parameters())

        self.assertEqual(3, K)
        self.assertEqual(5, P)
        self.assertGreaterEqual(diagnostics["minimum_cif"], 0.0)
        self.assertGreaterEqual(diagnostics["minimum_survival"], 0.0)
        self.assertGreaterEqual(diagnostics["minimum_cif_increment"], -1e-6)
        self.assertLessEqual(diagnostics["maximum_conservation_error"], 1e-6)
        self.assertLessEqual(diagnostics["maximum_time_zero_mass_error"], 1e-6)
        self.assertLessEqual(
            diagnostics["affine_mu_reconstruction_max_abs_error"],
            1e-5,
        )
        self.assertGreaterEqual(diagnostics["affine_mu_r2_min"], 1.0 - 1e-6)
        self.assertGreaterEqual(
            diagnostics["affine_mu_r2_min_nonconstant"],
            1.0 - 1e-6,
        )
        self.assertEqual(1.0, diagnostics["affine_mu_constant_cause_count"])
        self.assertEqual(2.0, diagnostics["affine_mu_nonconstant_cause_count"])
        self.assertLessEqual(
            diagnostics["affine_mu_constant_cause_max_abs_error"],
            1e-7,
        )
        self.assertGreater(diagnostics["initial_event_probability_mean"], 0.0)

    def test_fine_gray_convergence_is_read_from_checkpoint(self) -> None:
        cause_models = [
            {"converged": True, "iterations": index + 2, "score_norm": 1e-8}
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            method_dir = Path(temp_dir) / "methods" / "fine_gray"
            method_dir.mkdir(parents=True)
            torch.save({"cause_models": cause_models}, method_dir / "checkpoint.pt")
            convergence = _fine_gray_convergence_payload(Path(temp_dir))

        self.assertEqual(3, convergence["cause_fit_count"])
        self.assertEqual(3, convergence["converged_count"])
        self.assertTrue(convergence["all_converged"])

    def test_development_gate_enforces_every_preregistered_condition(self) -> None:
        metrics, manifests, truth = _passing_gate_inputs()
        passed = evaluate_development_gate(metrics, manifests, "hash", truth)

        self.assertTrue(passed["passed"])
        self.assertEqual(
            {"MSE_overall": 2, "Ctd_overall": 1, "IBS_overall": 2},
            passed["softcomp"]["ranks"],
        )
        self.assertGreaterEqual(passed["cscox"]["mean_ctd_rank"], 2)

        failure_mutations = (
            lambda m, a, t: m[0]["cs-Cox"].update({"Ctd_overall": 0.70}),
            _force_cscox_to_ctd_rank_one,
            lambda m, a, t: a[0]["data_diagnostics"].update(
                {"event_time_at_limit_fraction": 0.01}
            ),
            lambda m, a, t: a[0]["data_diagnostics"].update(
                {"affine_mu_reconstruction_max_abs_error": 0.01}
            ),
            lambda m, a, t: a[0]["data_diagnostics"].update(
                {"affine_mu_constant_cause_max_abs_error": 0.01}
            ),
            lambda m, a, t: a[0]["data_diagnostics"].update(
                {"initial_event_probability_p99": 0.10}
            ),
            lambda m, a, t: a[0]["data_diagnostics"].update(
                {"time_zero_atom_count_discrepancy": 4.0}
            ),
            lambda m, a, t: a[0]["fine_gray_convergence"].update(
                {"converged_count": 2, "all_converged": False}
            ),
            lambda m, a, t: m[0]["Fine-Gray"].update(
                {
                    "Implied_S_below_tolerance_count": 0,
                    "Implied_S_below_tolerance_subject_count": 0,
                    "Implied_S_min": 0.01,
                }
            ),
            lambda m, a, t: t.update({"affine_mu_reconstruction_max_abs_error": 0.01}),
        )
        for mutation in failure_mutations:
            metrics, manifests, truth = _passing_gate_inputs()
            mutation(metrics, manifests, truth)
            failed = evaluate_development_gate(metrics, manifests, "hash", truth)
            self.assertFalse(failed["passed"])

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
