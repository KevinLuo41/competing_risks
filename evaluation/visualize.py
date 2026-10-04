#!/usr/bin/env python3
"""
Visualization and reporting for competing risks model.
Saves plots as PNG files to a configurable output directory.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for devserver
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from ..crsoft_model.crsoft import BaseCRSoftNet

DEFAULT_OUTPUT_DIR = Path("/tmp/competing_risks")

# ── Publication-quality style constants ──
MODEL_STYLES: dict[str, dict] = {
    "True": dict(color="black", linestyle="-", linewidth=2.5, marker=None, zorder=10),
    "SoftComp": dict(
        color="#d62728",
        linestyle="-",
        linewidth=2.0,
        marker="*",
        markersize=8,
        zorder=9,
    ),
    "CRSoft": dict(
        color="#d62728",
        linestyle="-",
        linewidth=2.0,
        marker="*",
        markersize=8,
        zorder=9,
    ),
    "DeepHit": dict(
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        marker="^",
        markersize=5,
        zorder=5,
    ),
    "cs-Cox": dict(
        color="#2ca02c",
        linestyle="-.",
        linewidth=1.5,
        marker="s",
        markersize=4.5,
        zorder=5,
    ),
    "NeuralFG": dict(
        color="#ff7f0e",
        linestyle=":",
        linewidth=1.5,
        marker="D",
        markersize=4.5,
        zorder=5,
    ),
    "DSM": dict(
        color="#9467bd",
        linestyle=":",
        linewidth=1.5,
        marker="x",
        markersize=5,
        zorder=4,
        skip_plot=True,
    ),
}
CAUSE_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]


def _display_name(name: str) -> str:
    return "SoftComp" if name == "CRSoft" else name


def _get_style(name: str) -> dict:
    key = _display_name(name)
    return MODEL_STYLES.get(
        key,
        dict(
            color="gray",
            linestyle="-",
            linewidth=1.2,
            marker=".",
            markersize=4,
            zorder=3,
        ),
    )


def _ensure_output_dir(output_dir: Path | None = None) -> Path:
    out = output_dir or DEFAULT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )


def print_data_summary(
    train_data: dict[str, Tensor],
    K: int,
    P: int,
) -> None:
    """Print a formatted summary of the generated data."""
    delta = train_data["Delta"]
    print("\n─── Data Summary ───")
    print(f"  Covariates:          {P} dimensions")
    print(f"  Event types:         {K}")
    print(f"  Train samples:       {delta.shape[0]}")
    print(f"  Censoring rate:      {(delta == 0).float().mean():.1%}")
    for k in range(1, K + 1):
        print(f"    Event {k} rate:        {(delta == k).float().mean():.1%}")
    print(f"  Median observed Y:   {train_data['Y'].median():.3f}")
    print(f"  Mean true T:         {train_data['T_true'].mean():.3f}")
    if "intercept" in train_data:
        print(f"\n  intercept = {train_data['intercept']}")
    if "beta" in train_data:
        print(f"  beta      = {train_data['beta']}")
    if "alpha" in train_data:
        print(f"  alpha     = {train_data['alpha']}")


def plot_training_loss(
    losses: list[float],
    output_dir: Path | None = None,
) -> str:
    """Plot training loss curve. Returns saved file path."""
    _setup_style()
    out = _ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(range(1, len(losses) + 1), losses, linewidth=1.5, color="#1f77b4")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("SoftComp Training Loss")
    path = out / "training_loss.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return str(path)


def plot_event_time_distribution(
    train_data: dict[str, Tensor],
    K: int,
    output_dir: Path | None = None,
) -> str:
    """Plot Kaplan-Meier survival curve and cumulative incidence functions."""
    _setup_style()
    out = _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    y = train_data["Y"].numpy()
    delta = train_data["Delta"].numpy()
    n = len(y)

    order = y.argsort()
    y_sorted = y[order]
    delta_sorted = delta[order]

    # ── Left: Kaplan-Meier overall survival ──
    ax = axes[0]
    n_at_risk = n
    km_surv = 1.0
    t_plot = [0.0]
    s_plot = [1.0]
    for i in range(n):
        if delta_sorted[i] > 0:
            km_surv *= 1.0 - 1.0 / n_at_risk
        n_at_risk -= 1
        t_plot.append(y_sorted[i])
        s_plot.append(km_surv)

    ax.step(t_plot, s_plot, where="post", color="#1f77b4", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\hat{S}(t)$")
    ax.set_title("Kaplan\u2013Meier Survival Estimate")
    ax.set_ylim(-0.03, 1.03)

    import numpy as np

    cens_times = y_sorted[delta_sorted == 0]
    if len(cens_times) > 0:
        s_at_cens = np.interp(cens_times, t_plot, s_plot)
        ax.plot(
            cens_times,
            s_at_cens,
            "+",
            color="#888888",
            markersize=4,
            alpha=0.3,
            label="Censored",
        )
        ax.legend(framealpha=0.8)

    # ── Right: Cumulative Incidence Functions (Aalen-Johansen) ──
    ax = axes[1]
    cif = [0.0] * K
    t_cif = [0.0]
    cif_plot = [[0.0] for _ in range(K)]

    n_at_risk = n
    km_surv = 1.0
    for i in range(n):
        if delta_sorted[i] > 0:
            cause = int(delta_sorted[i]) - 1
            hazard = 1.0 / n_at_risk
            cif[cause] += km_surv * hazard
            km_surv *= 1.0 - hazard
        n_at_risk -= 1
        t_cif.append(y_sorted[i])
        for k in range(K):
            cif_plot[k].append(cif[k])

    for k in range(K):
        ax.step(
            t_cif,
            cif_plot[k],
            where="post",
            color=CAUSE_COLORS[k % len(CAUSE_COLORS)],
            linewidth=2,
            label=f"Cause {k + 1}",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\hat{F}_k(t)$")
    ax.set_title("Cumulative Incidence (Aalen\u2013Johansen)")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(framealpha=0.8)

    fig.tight_layout()
    path = out / "event_time_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return str(path)


@torch.no_grad()
def plot_cif_curves(
    model: BaseCRSoftNet,
    X_test: Tensor,
    true_cif_fn: Callable[..., tuple[Tensor, Tensor]],
    K: int,
    n_samples: int = 4,
    output_dir: Path | None = None,
) -> str:
    """Plot predicted vs true CIF curves for a few test samples. Returns saved file path."""
    _setup_style()
    out = _ensure_output_dir(output_dir)
    model.eval()

    t_grid = torch.linspace(0.1, 10.0, 100)

    fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4), sharey=True)
    if n_samples == 1:
        axes = [axes]

    for idx in range(n_samples):
        ax = axes[idx]
        x_i = X_test[idx : idx + 1]

        f_pred_list: list[Tensor] = []
        f_true_list: list[Tensor] = []
        for t_val in t_grid:
            t_batch = t_val.unsqueeze(0)
            f_p, _ = model.predict_cif(x_i, t_batch)
            f_t, _ = true_cif_fn(x_i, t_batch)
            f_pred_list.append(f_p.squeeze(0))
            f_true_list.append(f_t.squeeze(0))

        f_pred_all = torch.stack(f_pred_list)
        f_true_all = torch.stack(f_true_list)
        t_np = t_grid.numpy()

        for k in range(K):
            c = CAUSE_COLORS[k % len(CAUSE_COLORS)]
            ax.plot(
                t_np,
                f_true_all[:, k].numpy(),
                color=c,
                linewidth=2,
                label=f"True F_{k + 1}",
            )
            ax.plot(
                t_np,
                f_pred_all[:, k].numpy(),
                color=c,
                linewidth=2,
                linestyle="--",
                label=f"Pred F_{k + 1}",
            )

        ax.set_xlabel("Time $t$")
        if idx == 0:
            ax.set_ylabel(r"$F_k(t|\mathbf{x})$")
        ax.set_title(f"Sample {idx}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, framealpha=0.8)

    fig.suptitle("SoftComp: Predicted vs True CIF", fontsize=13)
    fig.tight_layout()
    path = out / "cif_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return str(path)


def plot_cif_comparison(
    model_cifs: dict[str, Tensor],
    eval_times: Tensor,
    K: int,
    n_samples: int = 6,
    sample_indices: list[int] | None = None,
    event_names: list[str] | None = None,
    output_dir: Path | None = None,
    t_max: float | None = None,
    sample_seed: int = 0,
) -> str:
    """Plot CIF curves from multiple models in a grid: rows=causes, cols=samples.

    Each model gets a unique (color, marker, linestyle) for instant identification.
    SoftComp uses red stars; True (if present) uses thick black solid.
    If t_max is set, truncate the time axis at that value.
    If sample_indices is set, plot those specific test samples.
    Otherwise pick `n_samples` random test indices using `sample_seed` for
    reproducibility.
    """
    _setup_style()
    out = _ensure_output_dir(output_dir)

    has_true = "True" in model_cifs
    model_names = list(model_cifs.keys())

    if sample_indices is not None:
        indices = sample_indices
        n_samples = len(indices)
    else:
        n_test = next(iter(model_cifs.values())).shape[0]
        rng = torch.Generator().manual_seed(sample_seed)
        indices = torch.randperm(n_test, generator=rng)[:n_samples].tolist()

    fig, axes = plt.subplots(
        K,
        n_samples,
        figsize=(4.2 * n_samples, 3.2 * K),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    t_np = eval_times.numpy()

    # Truncate time axis if t_max is specified
    if t_max is not None:
        t_mask = t_np <= t_max
        t_np = t_np[t_mask]
        model_cifs = {name: cif[:, :, t_mask] for name, cif in model_cifs.items()}

    n_t = len(t_np)
    markevery = max(1, n_t // 8)

    # Pre-compute per-cause y-max across plotted models only
    import numpy as np

    row_ymax = []
    for k in range(K):
        ymax_k = 0.0
        for model_name in model_names:
            if _get_style(model_name).get("skip_plot"):
                continue
            cif = model_cifs[model_name]
            for col_idx in indices:
                ymax_k = max(ymax_k, float(cif[col_idx, k, :].max()))
        row_ymax.append(ymax_k)

    for k in range(K):
        for col, sample_idx in enumerate(indices):
            ax = axes[k][col]

            for model_name in model_names:
                style = _get_style(model_name)
                if style.get("skip_plot"):
                    continue
                cif = model_cifs[model_name]
                disp = _display_name(model_name)

                plot_kw = dict(
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    zorder=style.get("zorder", 5),
                    alpha=0.9,
                )
                if style.get("marker"):
                    plot_kw["marker"] = style["marker"]
                    plot_kw["markersize"] = style.get("markersize", 5)
                    plot_kw["markevery"] = markevery

                label = disp if col == 0 and k == 0 else None
                ax.plot(t_np, cif[sample_idx, k, :].numpy(), label=label, **plot_kw)

            cause_label = (
                event_names[k]
                if event_names and k < len(event_names)
                else f"Cause {k + 1}"
            )
            if col == 0:
                ax.set_ylabel(r"$F_{%d}(t|\mathbf{x})$" % (k + 1))
            if k == K - 1:
                ax.set_xlabel("Time $t$")
            if k == 0:
                ax.set_title(f"Sample {sample_idx}")
            # Per-row y-axis: pad 15% above the max value in this row
            ylim_top = min(1.03, row_ymax[k] * 1.15 + 0.02)
            ax.set_ylim(-0.02, ylim_top)

    # Single shared legend at the top
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(handles), 6),
            fontsize=10,
            framealpha=0.9,
            edgecolor="0.7",
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out / "cif_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return str(path)


@torch.no_grad()
def plot_functional_pipeline(
    cif_pred: Tensor,
    X_test: Tensor,
    beta_func: Tensor,
    true_cif_fn: Callable[..., tuple[Tensor, Tensor]],
    eval_times: Tensor,
    K: int,
    sample_indices: list[int] | None = None,
    output_dir: Path | None = None,
) -> str:
    """Three-row "functional pipeline" figure for Case III:

      Row 1 — functional covariates X_j(s) for each subject (p subplots)
      Row 2 — true effect functions β_{k,j}(s)              (K × p grid)
      Row 3 — true vs predicted CIF F_k(t|x)                (K subplots)

    Highlights `len(sample_indices)` test subjects (default 3) in distinct
    colors so the same colors thread through all three rows. Background
    subjects (light gray) in Row 1 give cohort context.

    `cif_pred` must be the post-processed (e.g. isotonic-projected) CIF
    tensor of shape (n_test, K, n_eval_times) — pass in the same tensor
    used by `plot_cif_comparison` so the predictions stay monotone in t.
    """
    _setup_style()
    out = _ensure_output_dir(output_dir)

    n_test, p, n_grid = X_test.shape
    s_grid = torch.linspace(0.0, 1.0, n_grid).numpy()

    if sample_indices is None:
        rng = torch.Generator().manual_seed(0)
        sample_indices = torch.randperm(n_test, generator=rng)[:3].tolist()

    n_high = len(sample_indices)
    high_colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"][:n_high]
    n_bg = min(40, n_test)
    bg_idx = [i for i in range(n_bg) if i not in sample_indices]

    fig = plt.figure(figsize=(4.0 * p, 3.2 * 3 + 0.6))
    gs = fig.add_gridspec(
        3, max(p, K), height_ratios=[1.0, 1.0, 1.0], hspace=0.55, wspace=0.30
    )

    # ── Row 1: functional covariates X_j(s) ──
    for j in range(p):
        ax = fig.add_subplot(gs[0, j])
        for i in bg_idx:
            ax.plot(
                s_grid,
                X_test[i, j].numpy(),
                color="0.75",
                linewidth=0.6,
                alpha=0.45,
                zorder=2,
            )
        for c, idx in zip(high_colors, sample_indices):
            ax.plot(
                s_grid,
                X_test[idx, j].numpy(),
                color=c,
                linewidth=2.0,
                zorder=5,
                label=f"Subject {idx}",
            )
        ax.axhline(0, color="0.5", linewidth=0.5, zorder=1)
        ax.set_title(rf"$X_{{{j + 1}}}(s)$")
        ax.set_xlabel("$s$" if False else "")
        if j == 0:
            ax.set_ylabel("Functional covariate")
        if j == p - 1:
            ax.legend(fontsize=8, framealpha=0.85, loc="best")

    # ── Row 2: true effect functions β_{k,j}(s) ──
    # Layout: a single row of (K*p) cells would be too wide; use a nested gridspec.
    inner = gs[1, :].subgridspec(1, K * p, wspace=0.30)
    for k in range(K):
        for j in range(p):
            ax = fig.add_subplot(inner[0, k * p + j])
            ax.plot(
                s_grid,
                beta_func[k, j].numpy(),
                color=CAUSE_COLORS[k % len(CAUSE_COLORS)],
                linewidth=2.0,
            )
            ax.axhline(0, color="0.5", linewidth=0.5, zorder=1)
            ax.set_title(rf"$\beta_{{{k + 1},{j + 1}}}(s)$", fontsize=10)
            if j == 0 and k == 0:
                ax.set_ylabel("True effect")

    # ── Row 3: true vs predicted CIF for highlighted subjects ──
    # cif_pred is (n_test, K, n_eval_times), already isotonic-projected by the
    # caller (the post_process step on the CRSoft ModelSpec).
    t_np = eval_times.numpy()
    n_t = len(t_np)
    for k in range(K):
        ax = fig.add_subplot(gs[2, k])
        for c, idx in zip(high_colors, sample_indices):
            f_true_t = []
            for t_val in eval_times:
                ft, _ = true_cif_fn(X_test[idx : idx + 1], t_val.unsqueeze(0))
                f_true_t.append(ft.squeeze(0)[k].item())
            ax.plot(t_np, f_true_t, color=c, linewidth=2.0, label=f"Subject {idx}")
            ax.plot(
                t_np,
                cif_pred[idx, k, :].numpy(),
                color=c,
                linewidth=1.6,
                linestyle="--",
                marker="*",
                markersize=6,
                markevery=max(1, n_t // 10),
            )
            ax.set_xlabel("Time $t$")
            ax.set_ylabel(rf"$F_{{{k + 1}}}(t|\mathbf{{X}})$")
            ax.set_title(f"Cause {k + 1}: true (—) vs predicted (-- ★)")
            if k == 0:
                ax.legend(fontsize=8, framealpha=0.85, loc="best")

    fig.suptitle(
        r"Case III pipeline: $\mu_k(\mathbf{X},t) = c_k + \sum_j \langle X_j, \beta_{k,j} \rangle_{L^2} + \alpha\, t$",
        fontsize=12,
        y=1.00,
    )
    path = out / "functional_pipeline.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return str(path)


@torch.no_grad()
def print_example_predictions(
    model: BaseCRSoftNet,
    X_test: Tensor,
    true_cif_fn: Callable[..., tuple[Tensor, Tensor]],
    K: int,
    time_points: list[float] | None = None,
    n_samples: int = 3,
) -> None:
    """Print a table of predicted vs true CIF values."""
    if time_points is None:
        time_points = [1.0, 3.0, 5.0]

    x_sample = X_test[:n_samples]
    print("\n─── Example Predictions ───")
    for t_val in time_points:
        t_batch = torch.full((n_samples,), t_val)
        f_pred, s_pred = model.predict_cif(x_sample, t_batch)
        f_true, s_true = true_cif_fn(x_sample, t_batch)
        print(f"\n  t = {t_val}:")
        for i in range(n_samples):
            pred_str = " ".join(f"{f_pred[i, k]:.4f}" for k in range(K))
            true_str = " ".join(f"{f_true[i, k]:.4f}" for k in range(K))
            print(
                f"    Sample {i}: pred F=({pred_str}), S={s_pred[i]:.4f}"
                f"  |  true F=({true_str}), S={s_true[i]:.4f}"
            )
