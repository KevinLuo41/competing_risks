#!/usr/bin/env python3
"""Synthetic CRSoft sweep: push toward DeepHit/NeuralFG SOTA.

Current cached CRSoft on Synthetic (M=2, lambda=0.3, no Brier):
  Ctd_1=0.7211 Ctd_2=0.7180 Ctd_o=0.7196   IBS_o=0.1781
SOTA targets (DeepHit dominates everything):
  Ctd_1=0.7513 Ctd_2=0.7421 Ctd_o=0.7467   IBS_o=0.1736

Discrimination gap of ~0.027 is the dominant deficit. Paper ablation table 6
(README) showed M=8 lambda=0.5 reaches Ctd=0.743 but IBS=0.217 (under-prediction).
This sweep tests whether Brier-augmented loss can preserve the M=8 Ctd while
restoring IBS. If yes, ensemble averaging is the next lever.

Phases:
  A) Baseline sanity replay of (M=2, lam=0.3) and (M=8, lam=0.5).
  B) Brier-loss sweep over (M, lam) x brier_lambda combos.
  C) If a single config crosses 4/6 SOTA, try ensemble for variance reduction.

Pass --device cuda for GPU; --shard i/n for parallel sweeping (mod-shard over
the (M, lam, blam) cartesian).
"""

from __future__ import annotations

import argparse
import time

import torch
from torch import Tensor

from ...crsoft_model import CRSoftNet
from ...data.synthetic import load_data
from ...evaluation import (
    build_evaluation_time_grid,
    isotonic_project_cif,
    load_eval_cache,
)
from ...evaluation.survival import compute_ibs


# ======================================================================
# Shared eval (mirrors pbc/brier_sweep.py)
# ======================================================================
def _ctd_fast(
    Y_test: Tensor, Delta_test: Tensor, cif: Tensor, times: Tensor, k: int
) -> float:
    ev = torch.where(Delta_test == k)[0]
    if ev.numel() == 0:
        return 0.0
    Ye = Y_test[ev]
    it = (torch.searchsorted(times, Ye, right=True) - 1).clamp(0, len(times) - 1)
    cif_i = cif[ev, it]
    cif_j = cif[:, it].t()
    elig = Y_test.unsqueeze(0) > Ye.unsqueeze(1)
    cif_ib = cif_i.unsqueeze(1)
    score = torch.where(
        cif_ib > cif_j,
        torch.ones_like(cif_j),
        torch.where(
            cif_ib == cif_j, 0.5 * torch.ones_like(cif_j), torch.zeros_like(cif_j)
        ),
    )
    c = float((score * elig.float()).sum().item())
    t = float(elig.float().sum().item())
    return c / t if t > 0 else 0.0


def _evaluate_fast(
    cif: Tensor,
    Y_test: Tensor,
    Delta_test: Tensor,
    Y_train: Tensor,
    Delta_train: Tensor,
    et: Tensor,
    K: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in range(1, K + 1):
        out[f"Ctd_cause_{k}"] = _ctd_fast(Y_test, Delta_test, cif[:, k - 1, :], et, k)
        out[f"IBS_cause_{k}"] = compute_ibs(
            Y_test, Delta_test, Y_train, Delta_train, cif[:, k - 1, :], et, k
        )
    out["Ctd_overall"] = sum(out[f"Ctd_cause_{k}"] for k in range(1, K + 1)) / K
    out["IBS_overall"] = sum(out[f"IBS_cause_{k}"] for k in range(1, K + 1)) / K
    return out


def _print_baselines() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    print("\nBaselines (eval_cache):")
    for n, ck in [
        ("DeepHit", "deephit"),
        ("DSM", "dsm"),
        ("cs-Cox", "cs_cox"),
        ("NeuralFG", "neural_fg"),
        ("CRSoft (cached)", "crsoft"),
    ]:
        c = load_eval_cache("synthetic", ck)
        if c is None:
            continue
        _, m = c
        out[n] = m
        print(
            f"  {n:<18} C^td_o={m['Ctd_overall']:.4f} C^td_1={m['Ctd_cause_1']:.4f} "
            f"C^td_2={m['Ctd_cause_2']:.4f} IBS_o={m['IBS_overall']:.4f} "
            f"IBS_1={m['IBS_cause_1']:.4f} IBS_2={m['IBS_cause_2']:.4f}"
        )
    return out


def _targets(b: dict[str, dict[str, float]]) -> dict[str, float]:
    """SOTA targets across the four trained baselines (excluding our CRSoft)."""
    base = {k: v for k, v in b.items() if not k.startswith("CRSoft")}
    return {
        "Ctd_overall": max(m["Ctd_overall"] for m in base.values()),
        "Ctd_cause_1": max(m["Ctd_cause_1"] for m in base.values()),
        "Ctd_cause_2": max(m["Ctd_cause_2"] for m in base.values()),
        "IBS_overall": min(m["IBS_overall"] for m in base.values()),
        "IBS_cause_1": min(m["IBS_cause_1"] for m in base.values()),
        "IBS_cause_2": min(m["IBS_cause_2"] for m in base.values()),
    }


def _print_row(label: str, m: dict[str, float], t: dict[str, float]) -> int:
    def f(k: str) -> str:
        ok = m[k] >= t[k] if k.startswith("Ctd") else m[k] <= t[k]
        return f"{m[k]:.4f}{'*' if ok else ' '}"

    n = sum(
        [
            m["Ctd_overall"] >= t["Ctd_overall"],
            m["Ctd_cause_1"] >= t["Ctd_cause_1"],
            m["Ctd_cause_2"] >= t["Ctd_cause_2"],
            m["IBS_overall"] <= t["IBS_overall"],
            m["IBS_cause_1"] <= t["IBS_cause_1"],
            m["IBS_cause_2"] <= t["IBS_cause_2"],
        ]
    )
    print(
        f"  {label:<60} C^td_o={f('Ctd_overall')} C^td_1={f('Ctd_cause_1')} "
        f"C^td_2={f('Ctd_cause_2')} IBS_o={f('IBS_overall')} "
        f"IBS_1={f('IBS_cause_1')} IBS_2={f('IBS_cause_2')} | SOTA {n}/6"
    )
    return n


# ======================================================================
# Train/eval one config, return CIF (post-isotonic) and metrics
# ======================================================================
def _train_one(
    *,
    P: int,
    K: int,
    Xt: Tensor,
    Yt: Tensor,
    Dt: Tensor,
    Xs: Tensor,
    Ys: Tensor,
    Ds: Tensor,
    et: Tensor,
    seed: int,
    hidden_dim: int,
    num_blocks: int,
    dropout: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    n_aug: int,
    aug_weight: float,
    brier_lambda: float,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    m = CRSoftNet(
        input_dim=P,
        num_causes=K,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=dropout,
    )
    m.fit(
        Xt,
        Yt,
        Dt,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        n_aug=n_aug,
        aug_weight=aug_weight,
        brier_lambda=brier_lambda,
        verbose=False,
        device=device,
    )
    cif_raw = m.predict_cif_grid(Xs, et)
    cif = isotonic_project_cif(cif_raw, et)
    metrics = _evaluate_fast(cif, Ys, Ds, Yt, Dt, et, K)
    return cif_raw, metrics


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--shard", type=str, default="0/1", help="i/n mod-shard")
    p.add_argument("--epochs", type=int, default=200, help="epochs per train")
    p.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "A", "B", "B2", "C", "D", "E", "F", "G"],

    )
    args = p.parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("=" * 120)
    print("Synthetic CRSoft sweep — Brier loss + ensemble")
    print(
        f"device={device} epochs={args.epochs} shard={shard_i}/{shard_n} phase={args.phase}"
    )
    print("=" * 120)

    data = load_data(test_size=0.3, seed=42)
    K_val: int = data["K"]  # pyre-ignore[8]
    P_val: int = data["P"]  # pyre-ignore[8]
    Xt: Tensor = data["X_train"]  # pyre-ignore[8]
    Yt: Tensor = data["Y_train"]  # pyre-ignore[8]
    Dt: Tensor = data["Delta_train"]  # pyre-ignore[8]
    Xs: Tensor = data["X_test"]  # pyre-ignore[8]
    Ys: Tensor = data["Y_test"]  # pyre-ignore[8]
    Ds: Tensor = data["Delta_test"]  # pyre-ignore[8]
    et = build_evaluation_time_grid(Ys, Ds, n_grid=100)

    base = _print_baselines()
    tgt = _targets(base)
    print(f"\nSOTA targets: {tgt}\n")

    # Common config
    common = {
        "P": P_val,
        "K": K_val,
        "Xt": Xt,
        "Yt": Yt,
        "Dt": Dt,
        "Xs": Xs,
        "Ys": Ys,
        "Ds": Ds,
        "et": et,
        "hidden_dim": 32,
        "num_blocks": 1,
        "dropout": 0.0,
        "epochs": args.epochs,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "batch_size": 512,
        "device": device,
    }

    # ---- Phase A: baseline replays (sanity check) — only on shard 0
    if args.phase in ("all", "A") and shard_i == 0:
        print("[A] Baseline replays (no Brier)...")
        for n_aug, aug_weight in [(2, 0.3), (8, 0.5)]:
            t0 = time.time()
            _, mm = _train_one(
                seed=42,
                n_aug=n_aug,
                aug_weight=aug_weight,
                brier_lambda=0.0,
                **common,  # pyre-ignore[6]
            )
            _print_row(f"M={n_aug} lam={aug_weight} brier=0", mm, tgt)
            print(f"     ({time.time() - t0:.1f}s)", flush=True)

    # ---- Phase B: Brier loss sweep at high-aug regime
    if args.phase in ("all", "B"):
        print("\n[B] Brier-augmented training: vary brier_lambda at chosen (M, lam)...")
        # Six most-promising configs. Hypothesis: M=8 lam=0.5 brier=2 preserves
        # high-aug discrimination (paper M=8 → Ctd=0.743) while Brier rescues
        # IBS from 0.217 toward DeepHit-level 0.174.
        configs = [
            (8, 0.5, 1.0),
            (8, 0.5, 2.0),
            (8, 0.5, 3.0),
            (8, 0.5, 5.0),
            (4, 0.5, 2.0),
            (2, 0.5, 2.0),
        ]
        for idx, (n_aug, aug_weight, blam) in enumerate(configs):
            if idx % shard_n != shard_i:
                continue
            t0 = time.time()
            _, mm = _train_one(
                seed=42,
                n_aug=n_aug,
                aug_weight=aug_weight,
                brier_lambda=blam,
                **common,  # pyre-ignore[6]
            )
            _print_row(f"M={n_aug} lam={aug_weight} brier={blam}", mm, tgt)
            print(f"     ({time.time() - t0:.1f}s)", flush=True)

    # ---- Phase B2: extended Brier sweep — heavier brier_lambda and lower lam
    # Phase B showed (M=8, lam=0.5, brier=5) is best at 2/6 (Ctd_o=0.7471,
    # IBS_o=0.205), but IBS still 0.03 above DeepHit (0.174). Hypothesis:
    # M*lam=80% survival weight is too high; reducing lam to 0.3 (60% survival
    # at M=8) plus heavier brier should rescue IBS without losing Ctd.
    if args.phase in ("all", "B2"):
        print("\n[B2] Heavier Brier + lower lam variations...")
        configs = [
            (8, 0.5, 10.0),
            (8, 0.5, 20.0),
            (8, 0.3, 5.0),
            (8, 0.3, 10.0),
            (8, 0.3, 20.0),
            (12, 0.3, 5.0),
            (12, 0.3, 10.0),
            (4, 0.3, 5.0),
        ]
        for idx, (n_aug, aug_weight, blam) in enumerate(configs):
            if idx % shard_n != shard_i:
                continue
            t0 = time.time()
            _, mm = _train_one(
                seed=42,
                n_aug=n_aug,
                aug_weight=aug_weight,
                brier_lambda=blam,
                **common,  # pyre-ignore[6]
            )
            _print_row(f"M={n_aug} lam={aug_weight} brier={blam}", mm, tgt)
            print(f"     ({time.time() - t0:.1f}s)", flush=True)

    # ---- Phase D: architecture and epoch sweep at the Phase-B winner
    # Hypothesis: 14k-sample Synthetic should benefit from higher capacity
    # (h=64 or 128) and/or longer training (500-1000 epochs).
    if args.phase in ("all", "D"):
        print("\n[D] Architecture / epoch sweep at (M=8, lam=0.5, brier=5)...")
        configs = [
            {"hidden_dim": 64, "num_blocks": 1, "epochs": 200},
            {"hidden_dim": 128, "num_blocks": 1, "epochs": 200},
            {"hidden_dim": 32, "num_blocks": 2, "epochs": 200},
            {"hidden_dim": 32, "num_blocks": 1, "epochs": 500},
            {"hidden_dim": 64, "num_blocks": 1, "epochs": 500},
        ]
        for idx, override in enumerate(configs):
            if idx % shard_n != shard_i:
                continue
            t0 = time.time()
            cfg = {**common, **override}
            _, mm = _train_one(
                seed=42,
                n_aug=8,
                aug_weight=0.5,
                brier_lambda=5.0,
                **cfg,  # pyre-ignore[6]
            )
            label = (
                f"h={override['hidden_dim']} L={override['num_blocks']} "
                f"ep={override['epochs']}"
            )
            _print_row(label, mm, tgt)
            print(f"     ({time.time() - t0:.1f}s)", flush=True)

    # ---- Phase E: keep baseline IBS-good config (M=2, lam=0.3); push Ctd via
    # Brier loss / larger arch / more epochs WITHOUT raising survival weight.
    # The cached baseline (M=2, lam=0.3, brier=0) had IBS=0.178 (matches NeuralFG)
    # but Ctd=0.720 (gap ~0.027). Target: lift Ctd >=0.74 without raising IBS.
    if args.phase in ("all", "E"):
        print("\n[E] Baseline-IBS-anchored Ctd push (low M + Brier)...")
        configs = [
            # (n_aug, aug_weight, brier_lambda, hidden_dim, num_blocks, epochs)
            (2, 0.3, 2.0, 32, 1, 200),
            (2, 0.3, 5.0, 32, 1, 200),
            (2, 0.3, 0.0, 64, 1, 200),
            (2, 0.3, 2.0, 64, 1, 200),
            (2, 0.3, 0.0, 32, 1, 1000),
            (1, 0.5, 0.0, 32, 1, 200),
            (1, 0.5, 2.0, 32, 1, 200),
            (1, 0.3, 2.0, 32, 1, 200),
        ]
        for idx, (n_aug, aug_weight, blam, h, l, ep) in enumerate(configs):
            if idx % shard_n != shard_i:
                continue
            t0 = time.time()
            cfg = {**common, "hidden_dim": h, "num_blocks": l, "epochs": ep}
            _, mm = _train_one(
                seed=42,
                n_aug=n_aug,
                aug_weight=aug_weight,
                brier_lambda=blam,
                **cfg,  # pyre-ignore[6]
            )
            _print_row(
                f"M={n_aug} lam={aug_weight} brier={blam} h={h} L={l} ep={ep}",
                mm,
                tgt,
            )
            print(f"     ({time.time() - t0:.1f}s)", flush=True)

    # ---- Phase F: heterogeneous ensemble (high-Ctd + low-IBS members)
    # Phase B/C show single-config sweeps can't bridge the Ctd/IBS tradeoff.
    # Hypothesis: blending a high-aug member (M=8, lam=0.5, brier=5: high Ctd,
    # high IBS) with a low-aug member (M=2, lam=0.3, brier=0: low Ctd, low IBS)
    # may give a Pareto improvement — the C-index of the blend can exceed the
    # linear interpolation if discriminative info combines positively.
    if args.phase in ("all", "F"):
        if shard_i != 0:
            return  # only one shard runs phase F
        print("\n[F] Heterogeneous ensemble: blend high-Ctd and low-IBS configs...")
        # Train multiple seeds of each config, then evaluate blends.
        n_seeds_per_kind = 4
        cifs_high: list[Tensor] = []
        cifs_low: list[Tensor] = []
        for s in range(n_seeds_per_kind):
            t0 = time.time()
            cif_h, mh = _train_one(
                seed=s,
                n_aug=8,
                aug_weight=0.5,
                brier_lambda=5.0,
                **common,  # pyre-ignore[6]
            )
            cifs_high.append(cif_h)
            print(
                f"  high-Ctd s={s}: Ctd_o={mh['Ctd_overall']:.4f} IBS_o={mh['IBS_overall']:.4f} ({time.time() - t0:.1f}s)",
                flush=True,
            )
            t0 = time.time()
            cif_l, ml = _train_one(
                seed=s,
                n_aug=2,
                aug_weight=0.3,
                brier_lambda=0.0,
                **common,  # pyre-ignore[6]
            )
            cifs_low.append(cif_l)
            print(
                f"   low-IBS s={s}: Ctd_o={ml['Ctd_overall']:.4f} IBS_o={ml['IBS_overall']:.4f} ({time.time() - t0:.1f}s)",
                flush=True,
            )

        cif_high_avg = torch.stack(cifs_high, dim=0).mean(dim=0)
        cif_low_avg = torch.stack(cifs_low, dim=0).mean(dim=0)

        # Endpoints
        for label, cif_raw in [
            (f"{n_seeds_per_kind}-seed high-Ctd ensemble", cif_high_avg),
            (f"{n_seeds_per_kind}-seed low-IBS ensemble", cif_low_avg),
        ]:
            cif_iso = isotonic_project_cif(cif_raw, et)
            mm = _evaluate_fast(cif_iso, Ys, Ds, Yt, Dt, et, K_val)
            _print_row(label, mm, tgt)

        # Blends
        for w in [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]:
            cif_blend = w * cif_high_avg + (1.0 - w) * cif_low_avg
            cif_iso = isotonic_project_cif(cif_blend, et)
            mm = _evaluate_fast(cif_iso, Ys, Ds, Yt, Dt, et, K_val)
            _print_row(f"blend w(high-Ctd)={w:.2f}", mm, tgt)

    # ---- Phase G: single-seed epoch convergence at the Phase-B winner.
    # Per user feedback: ensembling baselines vs single-seed CRSoft is unfair
    # (the comparison baselines DeepHit/DSM/cs-Cox/NeuralFG are single-seed
    # too). So drop the 8-seed bagging from the recipe and instead pay for
    # discrimination via longer training. This phase locks seed=42 (the
    # convention used elsewhere in synthetic/run.py) and sweeps epochs.
    if args.phase in ("all", "G"):
        print(
            "\n[G] Single-seed epoch convergence (seed=42) at (M=8, lam=0.5, brier=5)..."
        )
        # Also include a few brier_lambda variants at the long-epoch regime
        # in case the Pareto-optimum brier shifts with more training.
        configs: list[tuple[int, float, int]] = [
            # (epochs, brier_lambda, weight_decay)
            (200, 5.0, int(1e-3 * 1e6)),
            (500, 5.0, int(1e-3 * 1e6)),
            (1000, 5.0, int(1e-3 * 1e6)),
            (2000, 5.0, int(1e-3 * 1e6)),
            (1000, 3.0, int(1e-3 * 1e6)),
            (1000, 7.0, int(1e-3 * 1e6)),
        ]
        for idx, (ep, blam, wd_micro) in enumerate(configs):
            if idx % shard_n != shard_i:
                continue
            t0 = time.time()
            cfg = {**common, "epochs": ep, "weight_decay": wd_micro / 1e6}
            _, mm = _train_one(
                seed=42,
                n_aug=8,
                aug_weight=0.5,
                brier_lambda=blam,
                **cfg,  # pyre-ignore[6]
            )
            _print_row(f"seed=42 ep={ep} brier={blam} wd={wd_micro / 1e6:.0e}", mm, tgt)
            print(f"     ({time.time() - t0:.1f}s)", flush=True)

    # ---- Phase C: ensemble of the best Phase-B config (multi-seed)
    # Hardcoded to (M=8, lam=0.5, brier=5.0) as the best-guess — adjust if a
    # better config emerges from Phase B/B2, then rerun --phase C.
    if args.phase in ("all", "C"):
        print("\n[C] Ensemble of best Brier config (multi-seed)...")
        best = {"n_aug": 8, "aug_weight": 0.5, "brier_lambda": 5.0}
        cifs: list[Tensor] = []
        for s in range(8):
            t0 = time.time()
            cif_raw, mm = _train_one(
                seed=s,
                **best,  # pyre-ignore[6]
                **common,  # pyre-ignore[6]
            )
            _print_row(f"single-seed s={s}", mm, tgt)
            cifs.append(cif_raw)
            print(f"     ({time.time() - t0:.1f}s)")

        for n_ens in (4, 8):
            cif_avg = torch.stack(cifs[:n_ens], dim=0).mean(dim=0)
            cif_iso = isotonic_project_cif(cif_avg, et)
            mm = _evaluate_fast(cif_iso, Ys, Ds, Yt, Dt, et, K_val)
            _print_row(f"{n_ens}-seed ensemble", mm, tgt)


if __name__ == "__main__":
    main()
