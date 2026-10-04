#!/usr/bin/env python3
"""Censored vs. uncensored versions of Cases II and III.

Each invocation trains one method on one formal replicate and writes its test metrics.
--uncensored keeps the same subjects, true event times, causes, and evaluation grid
but removes all censoring from the training and test data. Methods:
  JointSoftComp   unified configuration
  SoftComp        the manuscript's specification (Eq. (5) plus time augmentation, M = 2)
  SoftComp-noaug  the same without time augmentation (M = 0)
  NeuralFG        the manuscript's specification (the best baseline in Cases II and III)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

from ...crsoft_model.joint_softcomp import JointSoftComp
from ...evaluation import build_evaluation_time_grid, compute_mse_accuracy, evaluate_cif_metrics
from ...evaluation.simulation import compute_dist
from ..case2_v4 import run as case2
from ..case3_v5 import run as case3

logger: logging.Logger = logging.getLogger(__name__)

CASES = {2: (case2, (130_000, 140_000)), 3: (case3, (330_000, 340_000))}
METHODS = ("JointSoftComp", "SoftComp", "SoftComp-noaug", "NeuralFG")
MODEL_SEED = 0
Predict = Callable[[object, Tensor, Tensor], tuple[Tensor, Tensor]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=[2, 3], default=3)
    parser.add_argument("--method", choices=METHODS, default="JointSoftComp")
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--uncensored", action="store_true")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def prepare(case: int, replicate: int, uncensored: bool) -> object:
    module, (train_base, test_base) = CASES[case]
    seeds = {
        "n_train": 5000,
        "n_test": 1000,
        "train_seed": train_base + replicate,
        "test_seed": test_base + replicate,
    }
    data = module.prepare_data(**seeds)
    if case == 2:
        # The supplementary README uses 97.5%, while the original Case II
        # formal protocol uses 90%. Keep that formal protocol unchanged.
        # Only Case III defines additional fixed evaluation times in source.
        data = dataclasses.replace(
            data,
            eval_times=build_evaluation_time_grid(
                data.Y_test, data.Delta_test, n_grid=100, percentile_cap=97.5
            ),
        )
    if not uncensored:
        return data
    complete = module.prepare_data(**seeds, censor_rate=0.0)
    return dataclasses.replace(complete, eval_times=data.eval_times)


def build_method(
    name: str, data: object, module: object
) -> tuple[Callable[[], object], Predict, Callable[[Tensor], Tensor]]:
    if name == "JointSoftComp":

        def train() -> JointSoftComp:
            model = JointSoftComp(input_dim=module.P, num_causes=module.K)
            model.fit(
                data.X_train_fit,
                data.Y_train_fit,
                data.Delta_train_fit,
                verbose=False,
                device="cpu",
            )
            return model

        def predict(model: object, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
            return model.predict_cif_survival_grid(x, t)

        return train, predict, lambda cif: cif
    config = module.SoftCompConfig()
    if name == "SoftComp-noaug":
        config = dataclasses.replace(config, n_aug=0)
    specs = {s.name: s for s in module.build_specs(data, config, MODEL_SEED)}
    spec = specs["SoftComp" if name.startswith("SoftComp") else name]
    return spec.train, spec.predict_survival, spec.post_process


def run(args: argparse.Namespace) -> dict:
    module = CASES[args.case][0]
    data = prepare(args.case, args.replicate, args.uncensored)
    train, predict, post_process = build_method(args.method, data, module)
    torch.manual_seed(MODEL_SEED)
    start = time.perf_counter()
    model = train()
    train_time = time.perf_counter() - start
    cif, survival = predict(model, data.X_test, data.eval_times)
    cif = post_process(cif)
    metrics = compute_mse_accuracy(
        cif, data.X_test, data.eval_times, data.true_cif_fn, module.K
    )
    metrics.update(
        evaluate_cif_metrics(
            cif,
            data.Y_test,
            data.Delta_test,
            data.Y_train_full,
            data.Delta_train_full,
            data.eval_times,
            module.K,
        )
    )
    metrics.update(compute_dist(cif, survival))
    return {"metrics": metrics, "train_time_sec": train_time}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    torch.set_num_threads(args.threads)
    payload = {
        "case": args.case,
        "method": args.method,
        "replicate": args.replicate,
        "censoring": "none" if args.uncensored else "manuscript",
        "result": run(args),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"case{args.case}_rep{args.replicate:02d}_{args.method}.json"
    path.write_text(json.dumps(payload))
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
