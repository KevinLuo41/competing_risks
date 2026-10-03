#!/usr/bin/env python3
"""Backfill the required Fine-Gray baseline into completed Case II v3 runs."""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import torch
from torch import Tensor

from ...data import case2_v3
from .formal import (
    _aggregate,
    _default_output_dir,
    _formal_schedule_metadata,
    _method_is_complete,
    _read_json,
    _replicate_is_complete,
    _run_method,
    _write_json,
    ALL_MODELS,
    compute_configuration_hash,
    DEVELOPMENT_MODELS,
    ExperimentConfig,
    fine_gray_amendment_hash,
    FORMAL_TEST_SEED_BASE,
    FORMAL_TRAIN_SEED_BASE,
    METHOD_CACHE_KEYS,
    PlotCohort,
    REQUIRED_METHOD_ARTIFACTS,
)
from .run import _fine_gray_spec, PreparedData, SoftCompConfig

METHOD_NAME = "Fine-Gray"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-replicate", type=int, default=0)
    parser.add_argument("--end-replicate", type=int, default=10)
    return parser.parse_args()


def load_backfill_config(
    output_dir: Path,
) -> tuple[ExperimentConfig, str, dict[str, object]]:
    """Load and validate the completed five-method formal configuration."""
    payload = _read_json(output_dir / "config.json")
    execution = payload.get("execution")
    if not isinstance(execution, dict) or execution.get("phase") != "formal":
        raise ValueError("output directory does not contain a formal configuration")
    formal_identity = (
        int(execution["n_replicates"]),
        int(execution["train_seed_base"]),
        int(execution["test_seed_base"]),
    )
    if formal_identity != (10, FORMAL_TRAIN_SEED_BASE, FORMAL_TEST_SEED_BASE):
        raise ValueError(f"unexpected formal schedule: {formal_identity}")
    recorded_models = tuple(execution.get("models", []))
    if recorded_models not in {DEVELOPMENT_MODELS, ALL_MODELS}:
        raise ValueError(f"unexpected recorded model set: {recorded_models}")
    config = ExperimentConfig(
        phase="formal",
        n_replicates=int(execution["n_replicates"]),
        start_replicate=0,
        end_replicate=int(execution["n_replicates"]),
        train_seed_base=int(execution["train_seed_base"]),
        test_seed_base=int(execution["test_seed_base"]),
        model_seed=int(execution["model_seed"]),
        dgp_seed=int(execution["dgp_seed"]),
        cpu_threads=int(execution["cpu_threads"]),
        models=ALL_MODELS,
        softcomp=SoftCompConfig(**execution["softcomp"]),
    )
    configuration_hash = compute_configuration_hash(config)
    if payload.get("configuration_hash") != configuration_hash:
        raise RuntimeError("formal output configuration hash does not match")
    return config, configuration_hash, payload


def _load_plot_cohort(output_dir: Path, configuration_hash: str) -> PlotCohort:
    payload = torch.load(output_dir / "plot_cohort.pt", weights_only=False)
    if payload.get("configuration_hash") != configuration_hash:
        raise RuntimeError("plot cohort configuration hash does not match")
    return PlotCohort(
        X=payload["X"],
        times=payload["times"],
        true_cif=payload["true_cif"],
        true_survival=payload["true_survival"],
    )


def load_prepared_data(replicate_dir: Path) -> PreparedData:
    """Reconstruct the exact saved paired data for one formal replicate."""
    payload = torch.load(replicate_dir / "shared_data.pt", weights_only=False)
    x_train: Tensor = payload["X_train"]
    y_train: Tensor = payload["Y_train"]
    delta_train: Tensor = payload["Delta_train"]
    validation_indices: Tensor = payload["validation_indices"].long()
    fitting_indices: Tensor = payload["fitting_indices"].long()
    parameters: dict[str, Tensor] = payload["parameters"]
    return PreparedData(
        X_train_full=x_train,
        Y_train_full=y_train,
        Delta_train_full=delta_train,
        T_train_true=y_train,
        X_train_fit=x_train[fitting_indices],
        Y_train_fit=y_train[fitting_indices],
        Delta_train_fit=delta_train[fitting_indices],
        X_val=x_train[validation_indices],
        Y_val=y_train[validation_indices],
        Delta_val=delta_train[validation_indices],
        X_test=payload["X_test"],
        Y_test=payload["Y_test"],
        Delta_test=payload["Delta_test"],
        eval_times=payload["eval_times"],
        params=parameters,
        true_cif_fn=functools.partial(case2_v3.compute_cif, params=parameters),
        case_name="case2_v3",
        t_max=case2_v3.T_MAX,
    )


def _validate_existing_methods(payload: dict[str, object], path: Path) -> None:
    methods = set(payload)
    missing = set(DEVELOPMENT_MODELS) - methods
    unexpected = methods - set(ALL_MODELS)
    if missing or unexpected:
        raise RuntimeError(
            f"{path} has missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )


def _validate_legacy_replicate(
    replicate_dir: Path,
    configuration_hash: str,
) -> None:
    manifest_path = replicate_dir / "replicate_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("configuration_hash") != configuration_hash:
        raise RuntimeError(f"configuration hash mismatch in {manifest_path}")

    complete_path = replicate_dir / "complete.json"
    complete = _read_json(complete_path)
    expected_methods = set(complete.get("expected_methods", []))
    completed_methods = set(complete.get("completed_methods", []))
    if (
        complete.get("complete") is not True
        or complete.get("configuration_hash") != configuration_hash
        or not set(DEVELOPMENT_MODELS).issubset(expected_methods)
        or not set(DEVELOPMENT_MODELS).issubset(completed_methods)
        or expected_methods - set(ALL_MODELS)
        or completed_methods - set(ALL_MODELS)
    ):
        raise RuntimeError(f"incomplete legacy replicate: {complete_path}")

    for filename in (
        "metrics.json",
        "timings.json",
        "all_subject_probability_diagnostics.json",
    ):
        path = replicate_dir / filename
        _validate_existing_methods(_read_json(path), path)
    if not (replicate_dir / "shared_data.pt").is_file():
        raise RuntimeError(f"missing shared data in {replicate_dir}")

    for method in DEVELOPMENT_MODELS:
        cache_key = METHOD_CACHE_KEYS[method]
        method_dir = replicate_dir / "methods" / cache_key
        if not _method_is_complete(method_dir, configuration_hash):
            raise RuntimeError(f"incomplete legacy method {method}: {method_dir}")
        missing = [
            filename
            for filename in REQUIRED_METHOD_ARTIFACTS
            if not (method_dir / filename).is_file()
        ]
        if missing:
            raise RuntimeError(
                f"legacy method {method} is missing artifacts: {sorted(missing)}"
            )


def merge_fine_gray_result(
    replicate_dir: Path,
    metrics: dict[str, float | int],
    timings: dict[str, float],
    diagnostics: dict[str, float | int],
    config: ExperimentConfig,
    configuration_hash: str,
) -> None:
    """Merge one completed Fine-Gray result into replicate-level summaries."""
    manifest_path = replicate_dir / "replicate_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("configuration_hash") != configuration_hash:
        raise RuntimeError(f"configuration hash mismatch in {manifest_path}")

    additions: tuple[tuple[str, dict[str, object]], ...] = (
        ("metrics.json", metrics),
        ("timings.json", timings),
        ("all_subject_probability_diagnostics.json", diagnostics),
    )
    merged_payloads: list[tuple[Path, dict[str, object]]] = []
    for filename, method_payload in additions:
        path = replicate_dir / filename
        payload = _read_json(path)
        _validate_existing_methods(payload, path)
        payload[METHOD_NAME] = method_payload
        merged_payloads.append((path, payload))
    for path, payload in merged_payloads:
        _write_json(path, payload)

    amendment = _formal_schedule_metadata(config)["method_amendment"]
    manifest["expected_methods"] = list(ALL_MODELS)
    manifest["method_amendment"] = amendment
    _write_json(manifest_path, manifest)
    _write_json(
        replicate_dir / "complete.json",
        {
            "complete": True,
            "configuration_hash": configuration_hash,
            "expected_methods": list(ALL_MODELS),
            "completed_methods": list(ALL_MODELS),
        },
    )


def _update_config(
    output_dir: Path,
    config: ExperimentConfig,
    payload: dict[str, object],
) -> None:
    execution = payload["execution"]
    if not isinstance(execution, dict):
        raise ValueError("invalid execution configuration")
    execution["models"] = list(ALL_MODELS)
    payload.update(_formal_schedule_metadata(config))
    _write_json(output_dir / "config.json", payload)


def _all_replicates_complete(
    output_dir: Path,
    config: ExperimentConfig,
    configuration_hash: str,
) -> bool:
    return all(
        _replicate_is_complete(
            output_dir / "replicates" / f"rep_{replicate:03d}",
            config,
            replicate,
            configuration_hash,
        )
        for replicate in range(config.n_replicates)
    )


def backfill_replicate(
    output_dir: Path,
    replicate: int,
    config: ExperimentConfig,
    plot_cohort: PlotCohort,
    configuration_hash: str,
) -> None:
    """Fit or resume Fine-Gray for one saved formal replicate."""
    replicate_dir = output_dir / "replicates" / f"rep_{replicate:03d}"
    if not replicate_dir.exists():
        raise FileNotFoundError(f"missing replicate directory: {replicate_dir}")
    _validate_legacy_replicate(replicate_dir, configuration_hash)
    data = load_prepared_data(replicate_dir)
    spec = _fine_gray_spec(data)
    final_method_dir = replicate_dir / "methods" / spec.cache_key
    if final_method_dir.exists():
        if not _method_is_complete(
            final_method_dir,
            configuration_hash,
            fine_gray_amendment_hash(),
        ):
            raise RuntimeError(f"incomplete Fine-Gray directory: {final_method_dir}")
        method_root = replicate_dir
    else:
        method_root = replicate_dir / ".fine_gray_backfill"
    metrics, timings, diagnostics = _run_method(
        method_root, spec, data, plot_cohort, config, configuration_hash
    )
    if method_root != replicate_dir:
        staged_method_dir = method_root / "methods" / spec.cache_key
        final_method_dir.parent.mkdir(parents=True, exist_ok=True)
        staged_method_dir.replace(final_method_dir)
        (method_root / "methods").rmdir()
        method_root.rmdir()
    merge_fine_gray_result(
        replicate_dir,
        metrics,
        timings,
        diagnostics,
        config,
        configuration_hash,
    )


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir or _default_output_dir("formal")
    config, configuration_hash, config_payload = load_backfill_config(output_dir)
    if not 0 <= args.start_replicate < args.end_replicate <= config.n_replicates:
        raise ValueError("replicate range must be within the formal 0--9 schedule")
    torch.set_num_threads(config.cpu_threads)
    plot_cohort = _load_plot_cohort(output_dir, configuration_hash)
    for replicate in range(args.start_replicate, args.end_replicate):
        print(f"\nBackfilling Fine-Gray replicate {replicate:03d}", flush=True)
        backfill_replicate(
            output_dir,
            replicate,
            config,
            plot_cohort,
            configuration_hash,
        )
    if not _all_replicates_complete(output_dir, config, configuration_hash):
        print(
            "Fine-Gray subset completed; global config and aggregate remain "
            "unchanged until all 10 replicates are complete.",
            flush=True,
        )
        return
    _update_config(output_dir, config, config_payload)
    _aggregate(output_dir, config, configuration_hash)
    print(f"Updated six-method artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
