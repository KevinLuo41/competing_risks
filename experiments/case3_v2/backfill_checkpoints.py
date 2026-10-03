#!/usr/bin/env python3
"""Recreate and retain Case III baseline checkpoints for all replicates."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch import Tensor

from ...data.case3_interaction import generate_parameters
from ...evaluation import compute_mse_accuracy, get_output_dir
from ..case2_v2.run import (
    _build_replicate_specs,
    _display_name,
    _select_specs,
    _write_torch,
    ExperimentConfig,
    K,
    P,
    PreparedData,
)
from ..runner import ModelSpec
from .run import _prepare_case3_data, CASE_NAME

DEFAULT_MODELS = ("DeepHit", "DSM", "cs-Cox", "NeuralFG")
MSE_ABSOLUTE_TOLERANCE = 1e-6
MSE_RELATIVE_TOLERANCE = 1e-4

logger: logging.Logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=get_output_dir(CASE_NAME))
    parser.add_argument("--start-replicate", type=int, default=0)
    parser.add_argument("--end-replicate", type=int)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _integer_field(payload: dict[object, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise TypeError(f"config.json field {key!r} must be an integer")
    return value


def _load_config(input_dir: Path) -> ExperimentConfig:
    with (input_dir / "config.json").open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError("config.json must contain an object")
    return ExperimentConfig(
        n_replicates=_integer_field(payload, "n_replicates"),
        start_replicate=_integer_field(payload, "start_replicate"),
        end_replicate=_integer_field(payload, "end_replicate"),
        model_seed=_integer_field(payload, "model_seed"),
        dgp_seed=_integer_field(payload, "dgp_seed"),
        train_seed_base=_integer_field(payload, "train_seed_base"),
        test_seed_base=_integer_field(payload, "test_seed_base"),
        cpu_threads=_integer_field(payload, "cpu_threads"),
        models=None,
    )


def _replicate_range(args: argparse.Namespace, config: ExperimentConfig) -> range:
    end_replicate = (
        config.n_replicates if args.end_replicate is None else args.end_replicate
    )
    if not 0 <= args.start_replicate < end_replicate <= config.n_replicates:
        raise ValueError(
            "replicate range must satisfy "
            "0 <= start-replicate < end-replicate <= n-replicates"
        )
    return range(args.start_replicate, end_replicate)


def _standardize_softcomp_checkpoint(replicate_dir: Path) -> None:
    source = replicate_dir / "softcomp_model.pt"
    destination = replicate_dir / "checkpoints" / "crsoft.pt"
    if destination.exists():
        return
    if not source.is_file():
        raise FileNotFoundError(f"missing SoftComp checkpoint: {source}")
    _write_torch(
        destination,
        torch.load(source, map_location="cpu", weights_only=False),
    )


def _load_saved_mse(replicate_dir: Path, spec: ModelSpec) -> float:
    path = replicate_dir / "methods" / f"{spec.cache_key}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"saved method result must be a dictionary: {path}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError(f"saved method metrics must be a dictionary: {path}")
    mse = metrics.get("MSE_overall")
    if not isinstance(mse, float):
        raise TypeError(f"saved MSE_overall must be a float: {path}")
    return mse


def _checkpoint_prediction(
    spec: ModelSpec,
    checkpoint: dict[object, object],
    x_test: Tensor,
    eval_times: Tensor,
) -> Tensor:
    model = spec.load_ckpt(checkpoint)
    cif = spec.predict(model, x_test, eval_times)
    return spec.post_process(cif) if spec.cache_key == "crsoft" else cif


def _verify_mse(
    spec: ModelSpec,
    checkpoint: dict[object, object],
    replicate_dir: Path,
    data: PreparedData,
) -> float:
    cif = _checkpoint_prediction(
        spec,
        checkpoint,
        data.X_test,
        data.eval_times,
    )
    observed = compute_mse_accuracy(
        cif,
        data.X_test,
        data.eval_times,
        data.true_cif_fn,
        K,
    )["MSE_overall"]
    expected = _load_saved_mse(replicate_dir, spec)
    tolerance = max(
        MSE_ABSOLUTE_TOLERANCE,
        abs(expected) * MSE_RELATIVE_TOLERANCE,
    )
    if abs(observed - expected) > tolerance:
        raise RuntimeError(
            f"{_display_name(spec)} checkpoint MSE mismatch: "
            f"observed={observed:.9f}, expected={expected:.9f}, "
            f"tolerance={tolerance:.2g}"
        )
    return observed


def _train_checkpoint(
    spec: ModelSpec,
    replicate_dir: Path,
    data: PreparedData,
    model_seed: int,
) -> None:
    torch.manual_seed(model_seed)
    model = spec.train()
    checkpoint = spec.serialize(model)
    checkpoint["model_seed"] = model_seed
    mse = _verify_mse(spec, checkpoint, replicate_dir, data)
    destination = replicate_dir / "checkpoints" / f"{spec.cache_key}.pt"
    _write_torch(destination, checkpoint)
    logger.info(
        f"  {_display_name(spec)}: saved {destination.name}; "
        f"MSE={mse:.9f} matches original run"
    )


def _run_replicate(
    replicate: int,
    input_dir: Path,
    config: ExperimentConfig,
    parameters: dict[str, Tensor],
    requested_models: list[str],
    overwrite: bool,
) -> None:
    replicate_dir = input_dir / "replicates" / f"rep_{replicate:03d}"
    if not (replicate_dir / "complete.json").is_file():
        raise FileNotFoundError(f"replicate is not complete: {replicate_dir}")
    _standardize_softcomp_checkpoint(replicate_dir)
    data = _prepare_case3_data(replicate, config, parameters)
    specs = _select_specs(_build_replicate_specs(data), tuple(requested_models))
    for spec in specs:
        destination = replicate_dir / "checkpoints" / f"{spec.cache_key}.pt"
        if destination.exists() and not overwrite:
            logger.info(f"  {_display_name(spec)}: checkpoint already exists, skipping")
            continue
        _train_checkpoint(spec, replicate_dir, data, config.model_seed)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    config = _load_config(args.input_dir)
    torch.set_num_threads(config.cpu_threads)
    parameters = generate_parameters(K=K, p=P, seed=config.dgp_seed)
    replicate_range = _replicate_range(args, config)
    for replicate in replicate_range:
        logger.info(f"Replicate {replicate:03d}")
        _run_replicate(
            replicate,
            args.input_dir,
            config,
            parameters,
            args.models,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
