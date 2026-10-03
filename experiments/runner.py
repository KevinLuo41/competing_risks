#!/usr/bin/env python3
# pyre-strict
"""Shared scaffolding for per-case experiment runners.

Each case's `run.py` provides case-specific bits:
  * data preparation (DGP / dataset loading)
  * `ModelSpec` list (which models, how to train each)
  * an `eval_fn(cif) -> dict[str, float]` that knows about K, X_test, Y_test, etc.

Everything else — CLI parsing, two-tier cache logic (model + eval), the
per-model runner, the planning table — lives here.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from ..evaluation import (
    compute_dist,
    get_output_dir,
    load_eval_cache,
    load_model,
    save_eval_cache,
    save_model,
)


# ======================================================================
# Per-model spec — bundles {how-to-train, how-to-(de)serialize, how-to-predict}
# so the unified runner can apply the same load-or-train + load-or-eval logic
# to every entry.
# ======================================================================
@dataclass
class ModelSpec:
    name: str  # human-readable label in the results table
    cache_key: str  # filename stem, e.g. "deephit"
    cli_aliases: list[str] = field(default_factory=list)
    train: Callable[[], object] = lambda: None
    serialize: Callable[[object], dict] = lambda m: {}
    load_ckpt: Callable[[dict], object] = lambda d: None
    predict: Callable[[object, Tensor, Tensor], Tensor] = lambda m, X, t: torch.empty(0)
    predict_survival: (
        Callable[[object, Tensor, Tensor], tuple[Tensor, Tensor]] | None
    ) = None
    allow_diagnostic_cif_mismatch: bool = False
    post_process: Callable[[Tensor], Tensor] = lambda c: c  # default no-op
    is_iteration_target: bool = False  # CRSoft = True → eval cache never read

    def matches(self, token: str) -> bool:
        def norm(s: str) -> str:
            return s.lower().replace("-", "").replace("_", "")

        accepted = {norm(self.name), norm(self.cache_key)} | {
            norm(a) for a in self.cli_aliases
        }
        return norm(token) in accepted


# ======================================================================
# CLI parsing + token resolution
# ======================================================================
def parse_args(prog_desc: str = "Run experiment") -> argparse.Namespace:
    """Standard CLI for case runners.

    --retrain MODEL ...    force retrain (also clears eval cache)
    --reeval  MODEL ...    force re-eval (recomputes metrics)
    --list                show cache state and exit
    """
    p = argparse.ArgumentParser(description=prog_desc)
    p.add_argument(
        "--retrain",
        nargs="+",
        default=[],
        metavar="MODEL",
        help="force retrain (also clears eval cache for that model)",
    )
    p.add_argument(
        "--reeval",
        nargs="+",
        default=[],
        metavar="MODEL",
        help="force re-eval (uses cached model if present, recomputes metrics)",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="list models and current cache state, then exit",
    )
    return p.parse_args()


def resolve_models(tokens: list[str], specs: list[ModelSpec]) -> set[str]:
    """Map CLI tokens to spec.cache_key set, with validation."""
    out: set[str] = set()
    for tok in tokens:
        match = next((s for s in specs if s.matches(tok)), None)
        if match is None:
            valid = ", ".join(sorted({s.cache_key for s in specs}))
            raise SystemExit(f"unknown model name '{tok}'. Valid: {valid}")
        out.add(match.cache_key)
    return out


# ======================================================================
# Cache directory helpers
# ======================================================================
def models_dir(case_name: str) -> Path:
    return get_output_dir(case_name) / "models"


def eval_cache_dir(case_name: str) -> Path:
    return get_output_dir(case_name) / "eval_cache"


def delete_cache(case_name: str, cache_key: str, kinds: tuple[str, ...]) -> None:
    """Delete model.pt and/or eval_cache/.pt for a model. kinds ⊆ {'model', 'eval'}."""
    paths = {
        "model": models_dir(case_name) / f"{cache_key}.pt",
        "eval": eval_cache_dir(case_name) / f"{cache_key}.pt",
    }
    for k in kinds:
        path = paths[k]
        if path.exists():
            path.unlink()
            print(f"  deleted {path.name}")


# ======================================================================
# Reporting
# ======================================================================
def print_cache_state(case_name: str, specs: list[ModelSpec]) -> None:
    print("\nCache state:")
    print(f"  {'model':<12} {'model.pt':<10} {'eval_cache':<10}")
    for s in specs:
        m_ok = (models_dir(case_name) / f"{s.cache_key}.pt").exists()
        e_ok = (eval_cache_dir(case_name) / f"{s.cache_key}.pt").exists()
        print(f"  {s.name:<12} {'✓' if m_ok else 'x':<10} {'✓' if e_ok else 'x':<10}")


def print_plan(case_name: str, specs: list[ModelSpec]) -> None:
    """Print what each model will do, after --retrain/--reeval flags have been applied."""
    print("\nExecution plan:")
    print(f"  {'model':<12} {'model action':<24} {'eval action':<24}")
    print(f"  {'-' * 12:<12} {'-' * 24:<24} {'-' * 24:<24}")
    for s in specs:
        m_present = (models_dir(case_name) / f"{s.cache_key}.pt").exists()
        e_present = (eval_cache_dir(case_name) / f"{s.cache_key}.pt").exists()

        model_action = "load model.pt" if m_present else "TRAIN (no cache)"
        if s.is_iteration_target:
            eval_action = "EVAL (iteration target)"
        elif e_present:
            eval_action = "load eval_cache.pt"
        else:
            eval_action = "EVAL + save cache"

        print(f"  {s.name:<12} {model_action:<24} {eval_action:<24}")
    print()


def print_results(
    case_label: str,
    K: int,
    all_results: list[tuple[str, dict[str, float]]],
) -> None:
    """Print predictive metrics and optional probability diagnostics."""
    print("\n" + "=" * 70)
    print(f"RESULTS -- {case_label}")
    print("=" * 70)
    names = [name for name, _ in all_results]
    results = [r for _, r in all_results]
    header = f"{'Metric':<20}" + "".join(f"{n:>12}" for n in names)
    print(header)
    print("-" * len(header))

    for key, label in [("MSE_overall", "MSE"), ("classification_accuracy", "Accuracy")]:
        if key in results[0]:
            vals = "".join(f"{r[key]:>12.4f}" for r in results)
            print(f"{label:<20}{vals}")
    print()

    for k in range(1, K + 1):
        vals = "".join(f"{r[f'Ctd_cause_{k}']:>12.4f}" for r in results)
        print(f"{'C^td cause ' + str(k):<20}{vals}")
    vals = "".join(f"{r['Ctd_overall']:>12.4f}" for r in results)
    print(f"{'C^td overall':<20}{vals}")
    print()

    for k in range(1, K + 1):
        vals = "".join(f"{r[f'IBS_cause_{k}']:>12.4f}" for r in results)
        print(f"{'IBS cause ' + str(k):<20}{vals}")
    vals = "".join(f"{r['IBS_overall']:>12.4f}" for r in results)
    print(f"{'IBS overall':<20}{vals}")

    if all("Dist" in result for result in results):
        print()
        vals = "".join(f"{result['Dist']:>12.3e}" for result in results)
        print(f"{'Dist':<20}{vals}")
        vals = "".join(f"{result['ACO_points']:>12.0f}" for result in results)
        print(f"{'ACO points':<20}{vals}")
    print("\n" + "=" * 70 + "\nDone!")


def compute_probability_diagnostics(
    spec: ModelSpec,
    model: object,
    cached_cif: Tensor,
    X_test: Tensor,
    eval_times: Tensor,
) -> dict[str, float]:
    """Compute Dist and aggregate-CIF overflow for a fitted model."""
    if spec.predict_survival is None:
        raise ValueError(f"{spec.name} does not expose native survival")

    predicted_cif, native_survival = spec.predict_survival(model, X_test, eval_times)
    predicted_cif = spec.post_process(predicted_cif)
    if not spec.allow_diagnostic_cif_mismatch:
        torch.testing.assert_close(
            predicted_cif,
            cached_cif,
            rtol=2e-4,
            atol=5e-5,
        )

    diagnostics = compute_dist(cached_cif, native_survival)
    return {
        "Dist": float(diagnostics["Dist"]),
        "ACO_points": float(diagnostics["Implied_S_below_tolerance_count"]),
        "ACO_total_points": float(diagnostics["Implied_S_total_count"]),
        "ACO_fraction": float(diagnostics["Implied_S_below_tolerance_fraction"]),
    }


# ======================================================================
# Per-model runner
# ======================================================================
def run_or_load(
    case_name: str,
    spec: ModelSpec,
    X_test: Tensor,
    eval_times: Tensor,
    eval_fn: Callable[[Tensor], dict[str, float]],
    force_reeval: bool,
    include_probability_diagnostics: bool = False,
) -> tuple[object | None, Tensor, dict[str, float]]:
    """Unified per-model runner.

    Returns (model_or_None, cif_pred, metrics).
    `model_or_None` is non-None only when the model was actually used in this run
    (i.e., when no cached eval was available, or when the spec is the iteration target).

    Args:
        case_name: experiment name (e.g. "case1") — used for cache pathing.
        spec: ModelSpec describing the model.
        X_test: test covariates passed to spec.predict(model, X, eval_times).
        eval_times: time grid for prediction.
        eval_fn: takes a CIF tensor (n, K, T), returns a metrics dict.
        force_reeval: if True, ignore eval_cache and re-predict + re-evaluate.
        include_probability_diagnostics: compute Dist and ACO from the model's
            native survival output and final CIF predictions.
    """
    print(f"\n--- {spec.name} ---")

    # Step 1: model — load or train (only when actually needed)
    saved = load_model(case_name, spec.cache_key)
    use_cache = (not spec.is_iteration_target) and (not force_reeval)
    cached_eval = load_eval_cache(case_name, spec.cache_key) if use_cache else None
    diagnostics_cached = cached_eval is not None and all(
        key in cached_eval[1]
        for key in ("Dist", "ACO_points", "ACO_total_points", "ACO_fraction")
    )
    needs_model = (
        spec.is_iteration_target
        or force_reeval
        or cached_eval is None
        or (include_probability_diagnostics and not diagnostics_cached)
    )
    model: object | None = None
    if needs_model:
        if saved is not None:
            model = spec.load_ckpt(saved)
            print("  model loaded from cache")
        else:
            model = spec.train()
            save_model(case_name, spec.cache_key, spec.serialize(model))

    # Step 2: cif + metrics — load eval cache or compute
    if cached_eval is not None:
        cif, metrics = cached_eval
        if include_probability_diagnostics and not diagnostics_cached:
            assert model is not None
            metrics.update(
                compute_probability_diagnostics(
                    spec,
                    model,
                    cif,
                    X_test,
                    eval_times,
                )
            )
            print("  probability diagnostics restored from model checkpoint")
        print("  metrics loaded from eval_cache (skip O(n^2) C^td)")
        return model, cif, metrics

    print("  predicting + evaluating...")
    assert model is not None  # guaranteed by needs_model logic above
    native_survival: Tensor | None = None
    if include_probability_diagnostics:
        if spec.predict_survival is None:
            raise ValueError(f"{spec.name} does not expose native survival")
        cif, native_survival = spec.predict_survival(model, X_test, eval_times)
    else:
        cif = spec.predict(model, X_test, eval_times)
    cif = spec.post_process(cif)
    metrics = eval_fn(cif)
    if native_survival is not None:
        diagnostics = compute_dist(cif, native_survival)
        metrics.update(
            {
                "Dist": float(diagnostics["Dist"]),
                "ACO_points": float(diagnostics["Implied_S_below_tolerance_count"]),
                "ACO_total_points": float(diagnostics["Implied_S_total_count"]),
                "ACO_fraction": float(
                    diagnostics["Implied_S_below_tolerance_fraction"]
                ),
            }
        )
    if not spec.is_iteration_target:
        save_eval_cache(case_name, spec.cache_key, cif, metrics)
    return model, cif, metrics
