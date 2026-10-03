#!/usr/bin/env python3
"""Repeatable Monte Carlo runner for Case III with time-covariate interaction."""

from __future__ import annotations

import functools
from pathlib import Path

import torch
from torch import Tensor

from ...data.case3_interaction import (
    compute_cif,
    generate_data,
    generate_parameters,
    T_MAX,
)
from ...data.utils import generate_test_observations
from ...evaluation import build_evaluation_time_grid, get_output_dir
from ..case2_v2.run import (
    _aggregate,
    _build_config,
    _data_manifest,
    _execute_models,
    _get_tensor,
    _parse_args,
    _print_replicate_results,
    _protocol_payload,
    _write_json,
    ExperimentConfig,
    K,
    N_TEST,
    N_TRAIN,
    P,
    PreparedData,
)

CASE_NAME = "case3_v2"


def _case3_protocol_payload(config: ExperimentConfig) -> dict[str, object]:
    payload = _protocol_payload(config)
    payload.update(
        {
            "case": "Case III (Nonlinear time-covariate interaction)",
            "dgp": (
                "mu_k(x,t) = intercept_k + W_out_k @ tanh(W_shared @ x) "
                "+ t * [0.1 + 0.6 * sigmoid(gamma^T x / sqrt(p))]"
            ),
            "mean_time_slope": 0.4,
            "interaction_range": 0.6,
            "inverse_cdf_t_max": T_MAX,
        }
    )
    return payload


def _prepare_case3_data(
    replicate: int,
    config: ExperimentConfig,
    parameters: dict[str, Tensor],
) -> PreparedData:
    generated = generate_data(
        n=N_TRAIN,
        K=K,
        p=P,
        censor_rate=0.5,
        seed=config.train_seed_base + replicate,
        params=parameters,
    )
    train_data: dict[str, object] = dict(generated)
    x_train = _get_tensor(train_data, "X")
    y_train = _get_tensor(train_data, "Y")
    delta_train = _get_tensor(train_data, "Delta")
    n_val = int(N_TRAIN * 0.1)
    true_cif_fn = functools.partial(compute_cif, params=parameters)
    x_test, y_test, delta_test = generate_test_observations(
        N_TEST,
        P,
        true_cif_fn,
        censor_rate=0.5,
        seed=config.test_seed_base + replicate,
        t_max=T_MAX,
    )
    eval_times = build_evaluation_time_grid(y_test, delta_test, n_grid=100)
    return PreparedData(
        X_train_full=x_train,
        Y_train_full=y_train,
        Delta_train_full=delta_train,
        X_train_fit=x_train[n_val:],
        Y_train_fit=y_train[n_val:],
        Delta_train_fit=delta_train[n_val:],
        X_val=x_train[:n_val],
        Y_val=y_train[:n_val],
        Delta_val=delta_train[:n_val],
        X_test=x_test,
        Y_test=y_test,
        Delta_test=delta_test,
        eval_times=eval_times,
        parameters=parameters,
        true_cif_fn=true_cif_fn,
    )


def _run_case3_replicate(
    replicate: int,
    output_dir: Path,
    config: ExperimentConfig,
    parameters: dict[str, Tensor],
    resume: bool,
) -> bool:
    replicates_dir = output_dir / "replicates"
    replicates_dir.mkdir(parents=True, exist_ok=True)
    final_dir = replicates_dir / f"rep_{replicate:03d}"
    if final_dir.exists():
        if resume and (final_dir / "complete.json").exists():
            print(f"Replicate {replicate:03d}: already complete, skipping")
            return False
        raise FileExistsError(f"replicate output already exists: {final_dir}")

    temp_dir = replicates_dir / f".rep_{replicate:03d}.in_progress"
    if temp_dir.exists():
        if not resume:
            raise FileExistsError(
                f"partial replicate output already exists; rerun with --resume: {temp_dir}"
            )
        print(f"Replicate {replicate:03d}: resuming partial result")
    else:
        temp_dir.mkdir()

    print(f"\n{'=' * 72}\nReplicate {replicate:03d}\n{'=' * 72}")
    data = _prepare_case3_data(replicate, config, parameters)
    metrics, timings = _execute_models(temp_dir, data, config)
    _write_json(
        temp_dir / "data_manifest.json",
        _data_manifest(replicate, config, data),
    )
    _write_json(temp_dir / "metrics.json", metrics)
    _write_json(temp_dir / "timings.json", timings)
    _write_json(temp_dir / "complete.json", {"complete": True})
    temp_dir.rename(final_dir)
    _print_replicate_results(replicate, metrics, timings)
    return True


def main() -> None:
    args = _parse_args()
    config = _build_config(args)
    output_dir = args.output_dir or get_output_dir(CASE_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.cpu_threads)
    _write_json(output_dir / "config.json", _case3_protocol_payload(config))

    parameters = generate_parameters(K=K, p=P, seed=config.dgp_seed)
    for replicate in range(config.start_replicate, config.end_replicate):
        _run_case3_replicate(
            replicate,
            output_dir,
            config,
            parameters,
            args.resume,
        )
        _aggregate(output_dir, config)
    print(f"\nArtifacts: {output_dir}")


if __name__ == "__main__":
    main()
