#!/usr/bin/env python3

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import later.unittest
import torch

from ..data import case2_v3
from ..experiments.case2_v3.backfill_fine_gray import (
    _all_replicates_complete,
    _validate_legacy_replicate,
    merge_fine_gray_result,
)
from ..experiments.case2_v3.formal import (
    _formal_schedule_metadata,
    _method_is_complete,
    _replicate_is_complete,
    ALL_MODELS,
    compute_configuration_hash,
    development_gate_report,
    DEVELOPMENT_MODELS,
    expected_method_replicate_counts,
    expected_methods_for_replicate,
    ExperimentConfig,
    fine_gray_amendment_hash,
    FORMAL_REPLICATES_BY_METHOD,
    import_development_results,
    make_plot_cohort,
    METHOD_CACHE_KEYS,
    negative_survival_summary,
    ORIGINAL_FORMAL_REPLICATES_BY_METHOD,
    REQUIRED_METHOD_ARTIFACTS,
    SoftCompConfig,
    validate_formal_gate,
)


def _config(phase: str) -> ExperimentConfig:
    return ExperimentConfig(
        phase=phase,
        n_replicates=4 if phase == "development" else 10,
        start_replicate=0,
        end_replicate=4 if phase == "development" else 10,
        train_seed_base=30_000 if phase == "development" else 50_000,
        test_seed_base=40_000 if phase == "development" else 60_000,
        model_seed=0,
        dgp_seed=42,
        cpu_threads=8,
        models=DEVELOPMENT_MODELS if phase == "development" else ALL_MODELS,
        softcomp=SoftCompConfig(),
    )


def _passing_payloads() -> dict[int, dict[str, dict[str, float | int]]]:
    values = {
        "SoftComp": (0.10, 0.90, 0.10),
        "NeuralFG": (0.20, 0.80, 0.20),
        "DeepHit": (0.30, 0.70, 0.30),
        "cs-Cox": (0.40, 0.50, 0.40),
        "DSM": (0.50, 0.60, 0.50),
    }
    return {
        replicate: {
            method: {
                "MSE_overall": metrics[0] + replicate * 0.001,
                "Ctd_overall": metrics[1],
                "IBS_overall": metrics[2] + replicate * 0.001,
            }
            for method, metrics in values.items()
        }
        for replicate in range(4)
    }


def _write_method_artifacts(
    replicate_dir: Path,
    methods: tuple[str, ...],
    configuration_hash: str,
) -> None:
    for method in methods:
        method_dir = replicate_dir / "methods" / METHOD_CACHE_KEYS[method]
        method_dir.mkdir(parents=True)
        for filename in REQUIRED_METHOD_ARTIFACTS:
            if filename != "complete.json":
                (method_dir / filename).write_bytes(b"sentinel")
        complete: dict[str, object] = {
            "complete": True,
            "configuration_hash": configuration_hash,
        }
        if method == "Fine-Gray":
            complete["method_configuration_hash"] = fine_gray_amendment_hash()
        with (method_dir / "complete.json").open("w", encoding="utf-8") as file:
            json.dump(complete, file)


class Case2V3FormalTest(later.unittest.TestCase):
    def test_formal_schedule_runs_all_methods_for_ten_replicates(self) -> None:
        config = _config("formal")

        for replicate in range(config.n_replicates):
            self.assertEqual(
                ALL_MODELS,
                expected_methods_for_replicate(config, replicate),
            )
        self.assertEqual(
            FORMAL_REPLICATES_BY_METHOD,
            expected_method_replicate_counts(config),
        )
        with self.assertRaisesRegex(ValueError, "replicate must be"):
            expected_methods_for_replicate(config, 10)

    def test_development_schedule_runs_all_methods_in_every_replicate(self) -> None:
        config = _config("development")

        for replicate in range(config.n_replicates):
            self.assertEqual(
                DEVELOPMENT_MODELS,
                expected_methods_for_replicate(config, replicate),
            )
        self.assertEqual(
            dict.fromkeys(DEVELOPMENT_MODELS, 4),
            expected_method_replicate_counts(config),
        )

    def test_formal_replicate_completion_uses_its_expected_methods(self) -> None:
        config = _config("formal")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replicate_dir = root / "rep_009"
            replicate_dir.mkdir()
            methods_without_dsm = tuple(
                method for method in ALL_MODELS if method != "DSM"
            )
            with (replicate_dir / "metrics.json").open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump({method: {} for method in methods_without_dsm}, file)
            with (replicate_dir / "complete.json").open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    {
                        "complete": True,
                        "configuration_hash": "config",
                        "expected_methods": list(methods_without_dsm),
                        "completed_methods": list(methods_without_dsm),
                    },
                    file,
                )

            self.assertFalse(
                _replicate_is_complete(root / "rep_009", config, 9, "config")
            )
            with (replicate_dir / "metrics.json").open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump({method: {} for method in ALL_MODELS}, file)
            with (replicate_dir / "complete.json").open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    {
                        "complete": True,
                        "configuration_hash": "config",
                        "expected_methods": list(ALL_MODELS),
                        "completed_methods": list(ALL_MODELS),
                    },
                    file,
                )
            _write_method_artifacts(replicate_dir, ALL_MODELS, "config")
            self.assertTrue(
                _replicate_is_complete(root / "rep_009", config, 9, "config")
            )

    def test_fine_gray_completion_requires_its_method_hash(self) -> None:
        method_hash = fine_gray_amendment_hash()
        with tempfile.TemporaryDirectory() as directory:
            method_dir = Path(directory)
            complete_path = method_dir / "complete.json"
            with complete_path.open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "complete": True,
                        "configuration_hash": "base",
                    },
                    file,
                )
            self.assertFalse(_method_is_complete(method_dir, "base", method_hash))
            with complete_path.open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "complete": True,
                        "configuration_hash": "base",
                        "method_configuration_hash": method_hash,
                    },
                    file,
                )
            self.assertTrue(_method_is_complete(method_dir, "base", method_hash))

    def test_configuration_hash_ignores_execution_phase_and_seed_range(self) -> None:
        development_hash = compute_configuration_hash(_config("development"))
        formal_hash = compute_configuration_hash(_config("formal"))

        self.assertEqual(development_hash, formal_hash)
        self.assertEqual(
            "dab3fc72debc8918d6ffbe6e522319d349547014ad206f4f54639c79348cacb0",
            formal_hash,
        )

    def test_formal_schedule_metadata_records_the_amendment(self) -> None:
        metadata = _formal_schedule_metadata(_config("formal"))

        self.assertEqual(
            ORIGINAL_FORMAL_REPLICATES_BY_METHOD,
            metadata["original_formal_replicates_by_method"],
        )
        self.assertEqual(
            FORMAL_REPLICATES_BY_METHOD,
            metadata["effective_formal_replicates_by_method"],
        )
        self.assertEqual(
            list(range(10)),
            metadata["schedule_amendment"]["retained_replicates"],
        )
        self.assertEqual("Fine-Gray", metadata["method_amendment"]["added_method"])
        self.assertEqual(
            list(range(10)),
            metadata["method_amendment"]["required_replicates"],
        )
        self.assertEqual(
            fine_gray_amendment_hash(),
            metadata["method_amendment"]["configuration_hash"],
        )
        self.assertEqual(
            "d69d45f9e3acadf7a77a2a00cb8ab8f527aed925d0fb886e2f0327785a4947c5",
            fine_gray_amendment_hash(),
        )

    def test_fine_gray_backfill_merges_without_changing_existing_results(
        self,
    ) -> None:
        config = _config("formal")
        configuration_hash = compute_configuration_hash(config)
        fine_gray_payloads = {
            "metrics.json": {"MSE_overall": 0.25},
            "timings.json": {"train_time_sec": 1.5},
            "all_subject_probability_diagnostics.json": {"Implied_S_negative_count": 0},
        }
        with tempfile.TemporaryDirectory() as directory:
            replicate_dir = Path(directory)
            original_payloads = {}
            for filename in fine_gray_payloads:
                original_payloads[filename] = {
                    method: {"sentinel": index}
                    for index, method in enumerate(DEVELOPMENT_MODELS)
                }
                with (replicate_dir / filename).open("w", encoding="utf-8") as file:
                    json.dump(original_payloads[filename], file)
            with (replicate_dir / "replicate_manifest.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(
                    {
                        "configuration_hash": configuration_hash,
                        "expected_methods": list(DEVELOPMENT_MODELS),
                    },
                    file,
                )

            merge_fine_gray_result(
                replicate_dir,
                fine_gray_payloads["metrics.json"],
                fine_gray_payloads["timings.json"],
                fine_gray_payloads["all_subject_probability_diagnostics.json"],
                config,
                configuration_hash,
            )
            merge_fine_gray_result(
                replicate_dir,
                fine_gray_payloads["metrics.json"],
                fine_gray_payloads["timings.json"],
                fine_gray_payloads["all_subject_probability_diagnostics.json"],
                config,
                configuration_hash,
            )

            for filename, fine_gray_payload in fine_gray_payloads.items():
                with (replicate_dir / filename).open(encoding="utf-8") as file:
                    payload = json.load(file)
                self.assertEqual(set(ALL_MODELS), set(payload))
                self.assertEqual(fine_gray_payload, payload["Fine-Gray"])
                for method in DEVELOPMENT_MODELS:
                    self.assertEqual(
                        original_payloads[filename][method], payload[method]
                    )
            with (replicate_dir / "complete.json").open(encoding="utf-8") as file:
                complete = json.load(file)
            self.assertEqual(list(ALL_MODELS), complete["expected_methods"])
            self.assertEqual(list(ALL_MODELS), complete["completed_methods"])

    def test_fine_gray_backfill_rejects_hash_mismatch_before_mutation(self) -> None:
        config = _config("formal")
        with tempfile.TemporaryDirectory() as directory:
            replicate_dir = Path(directory)
            original = {method: {} for method in DEVELOPMENT_MODELS}
            for filename in (
                "metrics.json",
                "timings.json",
                "all_subject_probability_diagnostics.json",
            ):
                with (replicate_dir / filename).open("w", encoding="utf-8") as file:
                    json.dump(original, file)
            with (replicate_dir / "replicate_manifest.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump({"configuration_hash": "wrong"}, file)

            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                merge_fine_gray_result(
                    replicate_dir,
                    {},
                    {},
                    {},
                    config,
                    "expected",
                )

            with (replicate_dir / "metrics.json").open(encoding="utf-8") as file:
                self.assertEqual(original, json.load(file))

    def test_fine_gray_backfill_defers_global_update_until_all_replicates_complete(
        self,
    ) -> None:
        config = _config("formal")
        configuration_hash = compute_configuration_hash(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for replicate in range(config.n_replicates):
                replicate_dir = root / "replicates" / f"rep_{replicate:03d}"
                replicate_dir.mkdir(parents=True)
                with (replicate_dir / "metrics.json").open(
                    "w", encoding="utf-8"
                ) as file:
                    json.dump({method: {} for method in ALL_MODELS}, file)
                with (replicate_dir / "complete.json").open(
                    "w", encoding="utf-8"
                ) as file:
                    json.dump(
                        {
                            "complete": True,
                            "configuration_hash": configuration_hash,
                            "expected_methods": list(ALL_MODELS),
                            "completed_methods": list(ALL_MODELS),
                        },
                        file,
                    )
                _write_method_artifacts(
                    replicate_dir,
                    ALL_MODELS,
                    configuration_hash,
                )

            self.assertTrue(_all_replicates_complete(root, config, configuration_hash))
            incomplete_dir = root / "replicates" / "rep_009"
            (
                incomplete_dir
                / "methods"
                / METHOD_CACHE_KEYS["Fine-Gray"]
                / "plot_predictions.pt"
            ).unlink()
            self.assertFalse(_all_replicates_complete(root, config, configuration_hash))

    def test_fine_gray_backfill_rejects_incomplete_legacy_replicate(self) -> None:
        config = _config("formal")
        configuration_hash = compute_configuration_hash(config)
        with tempfile.TemporaryDirectory() as directory:
            replicate_dir = Path(directory)
            with (replicate_dir / "replicate_manifest.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump({"configuration_hash": configuration_hash}, file)
            incomplete_methods = [
                method for method in DEVELOPMENT_MODELS if method != "DSM"
            ]
            with (replicate_dir / "complete.json").open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "complete": True,
                        "configuration_hash": configuration_hash,
                        "expected_methods": incomplete_methods,
                        "completed_methods": incomplete_methods,
                    },
                    file,
                )

            with self.assertRaisesRegex(RuntimeError, "incomplete legacy"):
                _validate_legacy_replicate(replicate_dir, configuration_hash)

    def test_development_gate_requires_cscox_threshold_and_softcomp_top_two(
        self,
    ) -> None:
        report = development_gate_report(_passing_payloads(), "config")

        self.assertTrue(report["passed"])
        self.assertTrue(report["cscox"]["passed"])
        self.assertTrue(report["softcomp"]["passed"])
        self.assertEqual(
            {"MSE_overall": 1, "Ctd_overall": 1, "IBS_overall": 1},
            report["softcomp"]["ranks"],
        )

    def test_development_gate_fails_when_softcomp_is_third(self) -> None:
        payloads = _passing_payloads()
        for replicate in payloads:
            payloads[replicate]["SoftComp"]["MSE_overall"] = 0.35

        report = development_gate_report(payloads, "config")

        self.assertFalse(report["passed"])
        self.assertTrue(report["cscox"]["passed"])
        self.assertFalse(report["softcomp"]["passed"])
        self.assertEqual(3, report["softcomp"]["ranks"]["MSE_overall"])

    def test_negative_survival_summary_uses_all_replicates(self) -> None:
        rows = []
        values = [
            (10, 100, 2, 6000, 5, 1, -0.2),
            (20, 100, 3, 6000, 6, 2, -0.1),
        ]
        names = (
            "Implied_S_negative_count",
            "Implied_S_total_count",
            "Implied_S_negative_subject_count",
            "Implied_S_total_subject_count",
            "Implied_S_below_tolerance_count",
            "Implied_S_below_tolerance_subject_count",
            "Implied_S_min",
        )
        for replicate, replicate_values in enumerate(values):
            for metric, value in zip(names, replicate_values):
                rows.append(
                    {
                        "replicate": replicate,
                        "method": "DSM",
                        "metric": metric,
                        "value": value,
                    }
                )

        summary = negative_survival_summary(rows)[0]

        self.assertEqual(30, summary["negative_point_count"])
        self.assertEqual(200, summary["total_point_count"])
        self.assertEqual(5, summary["negative_subject_count"])
        self.assertEqual(12_000, summary["total_subject_count"])
        self.assertEqual(11, summary["tolerance_negative_point_count"])
        self.assertEqual(3, summary["tolerance_negative_subject_count"])
        self.assertEqual(2, summary["n"])
        self.assertEqual(2, summary["replicates_with_negative_survival"])
        self.assertEqual(
            1.0,
            summary["replicate_fraction_with_negative_survival"],
        )
        self.assertEqual(-0.2, summary["global_minimum_implied_survival"])

    def test_plot_cohort_is_reproducible_and_probability_coherent(self) -> None:
        parameters = case2_v3.generate_parameters(seed=42)
        first = make_plot_cohort(parameters)
        second = make_plot_cohort(parameters)

        self.assertTrue(torch.equal(first.X, second.X))
        self.assertEqual((5, 8, 100), tuple(first.true_cif.shape))
        self.assertTrue(
            torch.allclose(
                first.true_cif.sum(dim=1) + first.true_survival,
                torch.ones_like(first.true_survival),
                atol=1e-6,
            )
        )

    def test_formal_gate_rejects_a_different_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            with path.open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "passed": True,
                        "configuration_hash": "development-config",
                    },
                    file,
                )

            with self.assertRaisesRegex(RuntimeError, "does not match"):
                validate_formal_gate(path, "formal-config")

    def test_import_development_results_merges_verified_single_runs(self) -> None:
        config = _config("development")
        payloads = _passing_payloads()
        with tempfile.TemporaryDirectory() as directory:
            result_specs = []
            for replicate, results in payloads.items():
                path = Path(directory) / f"replicate_{replicate}.json"
                with path.open("w", encoding="utf-8") as file:
                    json.dump(
                        {
                            "protocol": {
                                "n_train": 5000,
                                "n_test": 1000,
                                "train_seed": 30_000 + replicate,
                                "test_seed": 40_000 + replicate,
                                "model_seed": 0,
                                "dgp_seed": 42,
                                "softcomp": asdict(config.softcomp),
                            },
                            "results": {
                                method: {"metrics": metrics}
                                for method, metrics in results.items()
                            },
                        },
                        file,
                    )
                result_specs.append(f"{replicate}={path}")

            report = import_development_results(result_specs, config, "config")

        self.assertTrue(report["complete"])
        self.assertTrue(report["passed"])
        self.assertEqual(4, len(report["sources"]))
