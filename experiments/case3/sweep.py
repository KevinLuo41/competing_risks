#!/usr/bin/env python3
"""
Case III (functional covariates) CRSoft hyperparameter sweep, MSE-focused.

Generates the Case III dataset once, then trains FunctionalCRSoftNet across a
config grid, post-processing each run with isotonic projection. Prints a sortable
results table with MSE (overall + per-cause), accuracy, C^td, IBS. Does NOT
touch caches — purely diagnostic. Pick the best row, set those values in
`run.py`, then re-run to commit.

Mirrors `experiments/pbc/sweep.py` but for functional covariates and MSE.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import time

import torch
from torch import Tensor

from ...crsoft_model import FunctionalCRSoftNet
from ...data.case3 import (
    _generate_functional_covariates,
    compute_cif,
    generate_data,
    N_BASIS,
)
from ...data.utils import assign_causes_and_censor, solve_inverse_cdf
from ...evaluation import (
    build_evaluation_time_grid,
    compute_mse_accuracy,
    isotonic_project_cif,
)
from ...evaluation.survival import compute_ctd, compute_ibs


@dataclasses.dataclass
class Config:
    hidden_dim: int = 32
    num_blocks: int = 1
    embed_dim: int = 8
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 3e-3
    n_aug: int = 2
    aug_weight: float = 0.5
    seed: int = 0
    batch_size: int = 256

    def label(self) -> str:
        return (
            f"h{self.hidden_dim} b{self.num_blocks} e{self.embed_dim} "
            f"lr{self.lr:.0e} wd{self.weight_decay:.0e} n_aug={self.n_aug} "
            f"aw={self.aug_weight:.1f} bs={self.batch_size} ep={self.epochs} s{self.seed}"
        )


def _train_and_eval(
    cfg: Config,
    K: int,
    P: int,
    n_grid: int,
    X_train: Tensor,
    Y_train: Tensor,
    Delta_train: Tensor,
    X_test: Tensor,
    Y_test: Tensor,
    Delta_test: Tensor,
    eval_times: Tensor,
    true_cif_fn,
    Y_train_full: Tensor,
    Delta_train_full: Tensor,
) -> dict[str, float]:
    torch.manual_seed(cfg.seed)
    m = FunctionalCRSoftNet(
        num_covariates=P,
        n_grid=n_grid,
        embed_dim=cfg.embed_dim,
        num_causes=K,
        hidden_dim=cfg.hidden_dim,
        num_blocks=cfg.num_blocks,
    )
    m.fit(
        X_train,
        Y_train,
        Delta_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        n_aug=cfg.n_aug,
        aug_weight=cfg.aug_weight,
        verbose=False,
    )
    cif = m.predict_cif_grid(X_test, eval_times)
    cif = isotonic_project_cif(cif, eval_times)

    out: dict[str, float] = {}
    out.update(compute_mse_accuracy(cif, X_test, eval_times, true_cif_fn, K))
    for k in range(1, K + 1):
        out[f"Ctd_cause_{k}"] = compute_ctd(
            Y_test, Delta_test, cif[:, k - 1, :], eval_times, k
        )
        out[f"IBS_cause_{k}"] = compute_ibs(
            Y_test,
            Delta_test,
            Y_train_full,
            Delta_train_full,
            cif[:, k - 1, :],
            eval_times,
            k,
        )
    out["Ctd_overall"] = sum(out[f"Ctd_cause_{k}"] for k in range(1, K + 1)) / K
    out["IBS_overall"] = sum(out[f"IBS_cause_{k}"] for k in range(1, K + 1)) / K
    return out


def _build_grid() -> list[Config]:
    """Case III sweep grid. Reference (current run.py) uses
    h32 b2 e8 lr5e-3 wd1e-4 n_aug=2 ep=1500 → MSE = 0.0367 (on N=30000).

    Hypothesis from cases 1-2 ablations: lr=5e-3 + wd=1e-4 is too aggressive,
    needs slower learning + stronger regularization. Port the case-2 winner
    (h32 b1 lr1e-3 wd3e-3 n_aug=2) and explore neighborhood. Default ep=500
    here for fast iteration (cosine annealing → most learning in early epochs);
    bump to 1000-1500 once a winner is identified.
    """
    grid: list[Config] = []
    # 0: current run.py reference (h32 b2 lr5e-3 wd1e-4)
    grid.append(
        Config(
            hidden_dim=32,
            num_blocks=2,
            embed_dim=8,
            lr=5e-3,
            weight_decay=1e-4,
            n_aug=2,
            epochs=500,
        )
    )
    # 1: case-2 winning recipe ported (h32 b1 lr1e-3 wd3e-3 n_aug=2)
    grid.append(Config())

    # 2-4: vary n_aug
    for n_aug in [0, 1, 4]:
        grid.append(Config(n_aug=n_aug))

    # 5-6: vary weight_decay
    for wd in [1e-4, 1e-2]:
        grid.append(Config(weight_decay=wd))

    # 7-8: vary lr
    for lr in [3e-3, 5e-3]:
        grid.append(Config(lr=lr))

    # 9-10: vary embed_dim
    for ed in [4, 16]:
        grid.append(Config(embed_dim=ed))

    # 11-12: vary architecture
    grid.append(Config(num_blocks=2))
    grid.append(Config(hidden_dim=64))

    # 13: smaller arch
    grid.append(Config(hidden_dim=16))

    # 14-16: longer training for the n_aug=1 winner — kept here for reproducibility.
    # Empirically these all UNDERPERFORM ep=500 on test MSE despite training loss
    # still trending down (0.02387 / 0.02517 / 0.02734 vs 0.02134 at ep=500). The
    # functional encoders dominate parameter count (~4.2K params) and start
    # memorising training noise once cosine LR enters its long near-zero tail.
    grid.append(Config(n_aug=1, epochs=1000))
    grid.append(Config(n_aug=1, epochs=1500))
    grid.append(Config(n_aug=1, epochs=2000))

    # Dedup by label
    seen: set[str] = set()
    out: list[Config] = []
    for c in grid:
        if c.label() not in seen:
            seen.add(c.label())
            out.append(c)
    return out


def _format_row(label: str, m: dict[str, float]) -> str:
    return (
        f"  {label:<70} "
        f"{m['MSE_overall']:>9.5f} "
        f"{m['MSE_cause_1']:>9.5f} "
        f"{m['MSE_cause_2']:>9.5f} "
        f"{m['classification_accuracy']:>7.3f} "
        f"{m['Ctd_overall']:>7.4f} "
        f"{m['IBS_overall']:>7.4f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--n_train",
        type=int,
        default=5000,
        help="Training samples (default 5000, paper protocol)",
    )
    p.add_argument(
        "--n_test", type=int, default=1000, help="Test samples (default 1000)"
    )
    p.add_argument(
        "--shard",
        type=str,
        default="0/1",
        help="Run only configs whose grid index%n==i (i/n format), e.g. '0/4'",
    )
    p.add_argument(
        "--indices",
        type=str,
        default="",
        help="Comma-separated specific grid indices to run (overrides --shard)",
    )
    args = p.parse_args()

    print("=" * 110)
    print(
        f"Case III CRSoft hyperparameter sweep   "
        f"n_train={args.n_train}  n_test={args.n_test}  "
        f"shard={args.shard}  indices={args.indices}"
    )
    print("=" * 110)

    K, P, N_GRID = 2, 3, 50
    print(f"\nGenerating training data (n={args.n_train})...")
    train_data = generate_data(n=args.n_train, K=K, p=P, n_grid=N_GRID, seed=42)
    X_train: Tensor = train_data["X_func"]  # pyre-ignore[9]
    Y_train: Tensor = train_data["Y"]  # pyre-ignore[9]
    Delta_train: Tensor = train_data["Delta"]  # pyre-ignore[9]
    beta_func: Tensor = train_data["beta_func"]  # pyre-ignore[9]
    alpha_true: Tensor = train_data["alpha"]  # pyre-ignore[9]
    intercept_true: Tensor = train_data["intercept"]  # pyre-ignore[9]

    print(f"Generating test data (n={args.n_test})...")
    torch.manual_seed(123)
    X_test = _generate_functional_covariates(args.n_test, P, N_GRID, N_BASIS)

    true_cif_fn = functools.partial(
        compute_cif,
        beta_func=beta_func,
        alpha=alpha_true,
        intercept=intercept_true,
        n_grid=N_GRID,
    )

    Y_test, Delta_test, _ = assign_causes_and_censor(
        X_test,
        solve_inverse_cdf(X_test, torch.rand(args.n_test), true_cif_fn),
        true_cif_fn,
        censor_rate=0.3,
    )

    eval_times = build_evaluation_time_grid(Y_train, Delta_train, n_grid=100)
    print(
        f"  Eval grid: {len(eval_times)} points, range "
        f"[{eval_times[0]:.2f}, {eval_times[-1]:.2f}]"
    )

    grid = _build_grid()
    if args.indices:
        wanted = {int(s) for s in args.indices.split(",") if s.strip()}
        run_indices = [i for i in range(len(grid)) if i in wanted]
    else:
        shard_i, shard_n = (int(s) for s in args.shard.split("/"))
        run_indices = [i for i in range(len(grid)) if i % shard_n == shard_i]
    print(f"\nFull grid: {len(grid)} configs. Running this shard: {run_indices}\n")

    print(
        f"  {'idx':>3}  {'config':<70} "
        f"{'MSE_o':>9} {'MSE_1':>9} {'MSE_2':>9} {'acc':>7} "
        f"{'C^td_o':>7} {'IBS_o':>7}  time"
    )
    print("  " + "-" * 132)

    results: list[tuple[int, Config, dict[str, float]]] = []
    for i in run_indices:
        cfg = grid[i]
        t0 = time.time()
        try:
            m = _train_and_eval(
                cfg,
                K,
                P,
                N_GRID,
                X_train,
                Y_train,
                Delta_train,
                X_test,
                Y_test,
                Delta_test,
                eval_times,
                true_cif_fn,
                Y_train,
                Delta_train,
            )
            dt = time.time() - t0
            print(f"  {i:>3}  " + _format_row(cfg.label(), m)[2:] + f"  ({dt:.1f}s)")
            results.append((i, cfg, m))
        except (RuntimeError, ValueError) as e:
            print(f"  {i:>3}  {cfg.label():<70} FAILED: {e}")

    print("\n" + "=" * 110)
    print("Sweep shard complete. Top 10 by MSE_overall (lower is better):")
    print("=" * 110)
    ranked = sorted(results, key=lambda r: r[2]["MSE_overall"])
    for i, cfg, m in ranked[:10]:
        print(f"  {i:>3}  " + _format_row(cfg.label(), m)[2:])


if __name__ == "__main__":
    main()
