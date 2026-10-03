#!/usr/bin/env python3
"""Plot representative Case II v3 implied-survival curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from torch import Tensor

METHOD_KEYS = ("deephit", "neural_fg", "cs_cox")
METHOD_LABELS = {
    "deephit": "DeepHit",
    "neural_fg": "NeuralFG",
    "cs_cox": "cs-Cox",
}
METHOD_COLORS = {
    "deephit": "#2f6bde",
    "neural_fg": "#e07a26",
    "cs_cox": "#2b8a66",
}
REPRESENTATIVE_QUANTILES = (0.10, 0.35, 0.65, 0.90)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-output", type=Path)
    return parser.parse_args()


def _require_tensor(payload: object, key: str, path: Path) -> Tensor:
    if not isinstance(payload, dict):
        raise TypeError(f"prediction payload must be a dictionary: {path}")
    value = payload.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"prediction field {key!r} must be a tensor: {path}")
    return value


def _load_predictions(
    input_dir: Path,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    times: Tensor | None = None
    true_survival: Tensor | None = None
    predictions = {}
    for method_key in METHOD_KEYS:
        path = input_dir / f"{method_key}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        current_times = _require_tensor(payload, "eval_times", path)
        current_true = _require_tensor(payload, "true_survival", path)
        if times is None:
            times = current_times
            true_survival = current_true
        elif not torch.equal(times, current_times) or not torch.equal(
            true_survival, current_true
        ):
            raise ValueError(f"prediction cohorts do not match: {path}")
        predictions[method_key] = _require_tensor(payload, "implied_survival", path)
    if times is None or true_survival is None:
        raise RuntimeError("no prediction payloads loaded")
    return times, true_survival, predictions


def _representative_indices(true_survival: Tensor) -> list[int]:
    final_survival = true_survival[:, -1]
    order = torch.argsort(final_survival)
    last_position = len(order) - 1
    return [
        int(order[round(quantile * last_position)].item())
        for quantile in REPRESENTATIVE_QUANTILES
    ]


def _plot_subject(
    axis: Axes,
    subject_index: int,
    quantile: float,
    times: Tensor,
    true_survival: Tensor,
    predictions: dict[str, Tensor],
) -> None:
    time_values = times.numpy()
    axis.plot(
        time_values,
        true_survival[subject_index].numpy(),
        color="#161616",
        linewidth=2.2,
        linestyle="--",
        label="True survival",
    )
    for method_key in METHOD_KEYS:
        axis.plot(
            time_values,
            predictions[method_key][subject_index].numpy(),
            color=METHOD_COLORS[method_key],
            linewidth=1.8,
            drawstyle="steps-post" if method_key != "neural_fg" else "default",
            label=METHOD_LABELS[method_key],
        )
    axis.axhline(0.0, color="#a53131", linewidth=1.0, alpha=0.7)
    axis.set_ylim(-0.025, 1.025)
    axis.grid(color="#c8c8c8", linewidth=0.55, alpha=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_title(
        f"Test subject {subject_index} | true final-S quantile {quantile:.0%}",
        fontsize=10,
    )
    axis.set_xlabel("Time t")
    axis.set_ylabel(r"$S_{\mathrm{imp}}(t\mid \mathbf{x})$")


def _save_plot(
    output: Path,
    times: Tensor,
    true_survival: Tensor,
    predictions: dict[str, Tensor],
) -> list[int]:
    indices = _representative_indices(true_survival)
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharex=True, sharey=True)
    for axis, subject_index, quantile in zip(
        axes.flatten(), indices, REPRESENTATIVE_QUANTILES
    ):
        _plot_subject(
            axis,
            subject_index,
            quantile,
            times,
            true_survival,
            predictions,
        )
    handles = [
        Line2D(
            [],
            [],
            color="#161616",
            linewidth=2.2,
            linestyle="--",
            label="True survival",
        )
    ]
    handles.extend(
        Line2D(
            [],
            [],
            color=METHOD_COLORS[method_key],
            linewidth=1.8,
            label=METHOD_LABELS[method_key],
        )
        for method_key in METHOD_KEYS
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Case II v3: representative implied-survival curves",
        fontsize=15,
        y=0.995,
    )
    figure.text(
        0.5,
        0.012,
        "Subjects selected only by true S(t_max) quantiles; red line marks S=0.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.90))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Selected test subjects: {indices}")
    for method_key in METHOD_KEYS:
        selected_minimum = predictions[method_key][indices].min().item()
        print(f"{METHOD_LABELS[method_key]} selected minimum S={selected_minimum:.8f}")
    print(f"Saved {output}")
    return indices


def _write_curve_data(
    output: Path,
    indices: list[int],
    times: Tensor,
    true_survival: Tensor,
    predictions: dict[str, Tensor],
) -> None:
    payload = {
        "times": times.tolist(),
        "subjects": [
            {
                "subject_index": subject_index,
                "true_final_survival_quantile": quantile,
                "true_survival": true_survival[subject_index].tolist(),
                "predictions": {
                    method_key: predictions[method_key][subject_index].tolist()
                    for method_key in METHOD_KEYS
                },
            }
            for subject_index, quantile in zip(indices, REPRESENTATIVE_QUANTILES)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, separators=(",", ":"))
        file.write("\n")


def main() -> None:
    args = _parse_args()
    times, true_survival, predictions = _load_predictions(args.input_dir)
    indices = _save_plot(args.output, times, true_survival, predictions)
    if args.data_output is not None:
        _write_curve_data(
            args.data_output,
            indices,
            times,
            true_survival,
            predictions,
        )


if __name__ == "__main__":
    main()
