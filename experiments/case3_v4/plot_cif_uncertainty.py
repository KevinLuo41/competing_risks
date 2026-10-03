#!/usr/bin/env python3
"""Plot Case III v4 CIF means and standard deviations across formal fits."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from torch import Tensor

from ...evaluation import get_output_dir

METHODS = (
    "DeepHit",
    "DSM",
    "cs-Cox",
    "NeuralFG",
    "SoftComp",
    "Fine-Gray",
)
METHOD_COLORS = {
    "DeepHit": "#1f77b4",
    "DSM": "#9467bd",
    "cs-Cox": "#2ca02c",
    "NeuralFG": "#ff7f0e",
    "SoftComp": "#d62728",
    "Fine-Gray": "#8c564b",
}
METHOD_MARKERS = {
    "DeepHit": "^",
    "DSM": "v",
    "cs-Cox": "s",
    "NeuralFG": "D",
    "SoftComp": "*",
    "Fine-Gray": "P",
}

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CifPlotData:
    """Fixed-cohort truth and formal prediction summaries."""

    times: Tensor
    true_cif: Tensor
    mean_cif: dict[str, Tensor]
    std_cif: dict[str, Tensor]
    n_replicates: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=get_output_dir("case3_v4") / "formal_50",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require_tensor(payload: object, key: str) -> Tensor:
    if not isinstance(payload, dict):
        raise TypeError("saved payload must be a dictionary")
    value = payload.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"saved payload field {key!r} must be a tensor")
    return value


def _load_plot_data(input_dir: Path) -> CifPlotData:
    cohort = torch.load(input_dir / "plot_cohort.pt", weights_only=False)
    aggregate = torch.load(
        input_dir / "aggregate" / "plot_predictions_all.pt",
        weights_only=False,
    )
    if not isinstance(aggregate, dict):
        raise TypeError("plot prediction aggregate must be a dictionary")
    means: dict[str, Tensor] = {}
    standard_deviations: dict[str, Tensor] = {}
    n_replicates: int | None = None
    for method in METHODS:
        method_payload = aggregate.get(method)
        means[method] = _require_tensor(method_payload, "mean")
        standard_deviations[method] = _require_tensor(method_payload, "std")
        final_cif = _require_tensor(method_payload, "final_cif")
        current_n = int(final_cif.shape[0])
        if n_replicates is not None and current_n != n_replicates:
            raise ValueError("methods have inconsistent replicate counts")
        n_replicates = current_n
    if n_replicates is None:
        raise RuntimeError("no method predictions were loaded")
    return CifPlotData(
        times=_require_tensor(cohort, "times"),
        true_cif=_require_tensor(cohort, "true_cif"),
        mean_cif=means,
        std_cif=standard_deviations,
        n_replicates=n_replicates,
    )


def _row_upper_limit(data: CifPlotData, cause_index: int) -> float:
    true_maximum = float(data.true_cif[:, cause_index, :].max().item())
    prediction_maximum = max(
        float(
            (
                data.mean_cif[method][:, cause_index, :]
                + data.std_cif[method][:, cause_index, :]
            )
            .max()
            .item()
        )
        for method in METHODS
    )
    return min(1.02, max(0.1, true_maximum, prediction_maximum) * 1.08)


def _plot_panel(
    axis: Axes,
    times: Tensor,
    true_cif: Tensor,
    mean_cif: dict[str, Tensor],
    std_cif: dict[str, Tensor],
) -> None:
    time_values = times.numpy()
    axis.plot(time_values, true_cif.numpy(), color="#1a1a1a", linewidth=1.8)
    marker_indices = (
        torch.linspace(0, len(times) - 1, steps=8).round().long().unique().numpy()
    )
    for method in METHODS:
        mean_values = mean_cif[method].numpy()
        standard_deviation = std_cif[method].numpy()
        axis.plot(
            time_values,
            mean_values,
            color=METHOD_COLORS[method],
            linewidth=1.1,
            alpha=0.9,
        )
        axis.errorbar(
            time_values[marker_indices],
            mean_values[marker_indices],
            yerr=standard_deviation[marker_indices],
            fmt=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            markersize=5.0 if method == "SoftComp" else 4.0,
            elinewidth=0.65,
            capsize=1.5,
            capthick=0.65,
            alpha=0.82,
        )
    axis.grid(color="#d9d9d9", linewidth=0.55, alpha=0.45)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _populate_axes(data: CifPlotData, axes: list[list[Axes]]) -> None:
    n_samples, n_causes = data.true_cif.shape[:2]
    for cause_index in range(n_causes):
        for sample_index in range(n_samples):
            axis = axes[cause_index][sample_index]
            _plot_panel(
                axis,
                data.times,
                data.true_cif[sample_index, cause_index, :],
                {
                    method: data.mean_cif[method][sample_index, cause_index, :]
                    for method in METHODS
                },
                {
                    method: data.std_cif[method][sample_index, cause_index, :]
                    for method in METHODS
                },
            )
            axis.set_ylim(0, _row_upper_limit(data, cause_index))
            if cause_index == 0:
                axis.set_title(f"Sample {sample_index + 1}", fontsize=10)
            if sample_index == 0:
                axis.set_ylabel(rf"$F_{{{cause_index + 1}}}(t\mid \mathbf{{x}})$")
            if cause_index == n_causes - 1:
                axis.set_xlabel("Time t")


def _legend_handles() -> list[object]:
    handles = [Line2D([], [], color="#1a1a1a", linewidth=1.8, label="True CIF")]
    handles.extend(
        Line2D(
            [],
            [],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=1.1,
            label=method,
        )
        for method in METHODS
    )
    return handles


def _save_figure(data: CifPlotData, output: Path) -> None:
    figure, axes = plt.subplots(
        data.true_cif.shape[1],
        data.true_cif.shape[0],
        figsize=(17.5, 8.0),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    _populate_axes(data, axes.tolist())
    figure.legend(
        handles=_legend_handles(),
        loc="upper center",
        ncol=7,
        frameon=True,
        bbox_to_anchor=(0.5, 0.995),
        title=(f"Points and error bars: mean ± 1 SD across {data.n_replicates} fits"),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.9, h_pad=0.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    data = _load_plot_data(args.input_dir)
    _save_figure(data, args.output)
    logger.info(f"Fixed plot-cohort samples: {data.true_cif.shape[0]}")
    logger.info(f"Aggregated formal fits per method: {data.n_replicates}")
    logger.info(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
