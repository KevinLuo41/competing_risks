#!/usr/bin/env python3
"""Plot Case III CIF means and standard deviations across Monte Carlo fits."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from torch import Tensor

from ...baseline_models import CsCox, DeepHit, DSM, NeuralFG
from ...crsoft_model import CRSoftNet
from ...data.case3_interaction import compute_cif, generate_parameters
from ...evaluation import get_output_dir, isotonic_project_cif

CASE_NAME = "case3_v2"
DEFAULT_N_SAMPLES = 5
DEFAULT_REFERENCE_REPLICATE = 0
DEFAULT_SAMPLE_SEED = 0
METHOD_KEYS = ("deephit", "dsm", "cs_cox", "neural_fg", "crsoft")
METHOD_LABELS = {
    "deephit": "DeepHit",
    "dsm": "DSM",
    "cs_cox": "cs-Cox",
    "neural_fg": "NeuralFG",
    "crsoft": "SoftComp",
}
METHOD_COLORS = {
    "deephit": "#1f77b4",
    "dsm": "#9467bd",
    "cs_cox": "#2ca02c",
    "neural_fg": "#ff7f0e",
    "crsoft": "#d62728",
}
METHOD_MARKERS = {
    "deephit": "^",
    "dsm": "v",
    "cs_cox": "s",
    "neural_fg": "D",
    "crsoft": "*",
}

Model = DeepHit | DSM | CsCox | NeuralFG | CRSoftNet

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CifPlotData:
    """Inputs summarized for the Case III CIF figure."""

    times: Tensor
    sample_indices: Tensor
    true_cif: Tensor
    mean_cif: dict[str, Tensor]
    std_cif: dict[str, Tensor]
    n_models: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=get_output_dir(CASE_NAME))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument(
        "--replace-sample-index",
        type=int,
        action="append",
        default=[],
        help="replace this selected index with the next unused index in the random order",
    )
    parser.add_argument(
        "--reference-replicate",
        type=int,
        default=DEFAULT_REFERENCE_REPLICATE,
        help="replicate whose fixed test cohort supplies the plotted subjects and grid",
    )
    return parser.parse_args()


def _load_config(input_dir: Path) -> tuple[int, int, int]:
    with (input_dir / "config.json").open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError("config.json must contain an object")
    return (
        _integer_field(payload, "n_replicates"),
        _integer_field(payload, "dgp_seed"),
        _integer_field(payload, "cpu_threads"),
    )


def _integer_field(payload: dict[object, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise TypeError(f"config.json field {key!r} must be an integer")
    return value


def _require_tensor(payload: object, key: str) -> Tensor:
    if not isinstance(payload, dict):
        raise TypeError("saved prediction payload must be a dictionary")
    value = payload.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"saved prediction field {key!r} must be a tensor")
    return value


def _select_reference_subjects(
    input_dir: Path,
    reference_replicate: int,
    n_samples: int,
    sample_seed: int,
    replace_sample_indices: list[int],
) -> tuple[Tensor, Tensor, Tensor]:
    path = (
        input_dir
        / "replicates"
        / f"rep_{reference_replicate:03d}"
        / "softcomp_predictions.pt"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    x_test = _require_tensor(payload, "X_test")
    times = _require_tensor(payload, "eval_times")
    if not 0 < n_samples <= x_test.shape[0]:
        raise ValueError(f"--n-samples must be between 1 and {x_test.shape[0]}")
    generator = torch.Generator().manual_seed(sample_seed)
    random_order = torch.randperm(x_test.shape[0], generator=generator).tolist()
    sample_indices = random_order[:n_samples]
    replacements = iter(random_order[n_samples:])
    for sample_index in replace_sample_indices:
        if sample_index not in sample_indices:
            raise ValueError(
                f"cannot replace sample {sample_index}: "
                f"initial random selection is {sample_indices}"
            )
        position = sample_indices.index(sample_index)
        replacement = next(
            candidate for candidate in replacements if candidate not in sample_indices
        )
        sample_indices[position] = replacement
    sample_indices_tensor = torch.tensor(sample_indices, dtype=torch.long)
    return x_test[sample_indices_tensor], times, sample_indices_tensor


def _load_model(method_key: str, payload: dict[object, object]) -> Model:
    if method_key == "deephit":
        return DeepHit.load_from_checkpoint(payload)
    if method_key == "dsm":
        return DSM.load_from_checkpoint(payload)
    if method_key == "cs_cox":
        return CsCox.load_from_checkpoint(payload)
    if method_key == "neural_fg":
        return NeuralFG.load_from_checkpoint(payload)
    if method_key == "crsoft":
        return CRSoftNet.load_from_checkpoint(payload)
    raise ValueError(f"unknown method key: {method_key}")


def _predict_cif(model: Model, x: Tensor, times: Tensor) -> Tensor:
    if isinstance(model, CRSoftNet):
        return isotonic_project_cif(model.predict_cif_grid(x, times), times)
    return model.predict_cif(x, times)


def _load_model_prediction(
    path: Path,
    method_key: str,
    x: Tensor,
    times: Tensor,
) -> Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint must be a dictionary: {path}")
    return _predict_cif(_load_model(method_key, payload), x, times)


def _summarize_predictions(
    input_dir: Path,
    n_replicates: int,
    x: Tensor,
    times: Tensor,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    means = {}
    standard_deviations = {}
    for method_key in METHOD_KEYS:
        predictions = []
        for replicate in range(n_replicates):
            path = (
                input_dir
                / "replicates"
                / f"rep_{replicate:03d}"
                / "checkpoints"
                / f"{method_key}.pt"
            )
            if not path.is_file():
                raise FileNotFoundError(f"missing model checkpoint: {path}")
            predictions.append(_load_model_prediction(path, method_key, x, times))
        stacked = torch.stack(predictions)
        means[method_key] = stacked.mean(dim=0)
        standard_deviations[method_key] = stacked.std(dim=0, unbiased=n_replicates > 1)
        logger.info(f"Loaded {n_replicates} {METHOD_LABELS[method_key]} checkpoints")
    return means, standard_deviations


def _compute_true_cif(x: Tensor, times: Tensor, dgp_seed: int) -> Tensor:
    parameters = generate_parameters(seed=dgp_seed)
    true_cif = torch.empty(x.shape[0], 3, times.shape[0])
    with torch.no_grad():
        for time_index, time in enumerate(times):
            values, _ = compute_cif(x, time.expand(x.shape[0]), parameters)
            true_cif[:, :, time_index] = values
    return true_cif


def _prepare_plot_data(
    input_dir: Path,
    n_samples: int,
    sample_seed: int,
    reference_replicate: int,
    replace_sample_indices: list[int],
) -> CifPlotData:
    n_replicates, dgp_seed, cpu_threads = _load_config(input_dir)
    torch.set_num_threads(cpu_threads)
    x, times, sample_indices = _select_reference_subjects(
        input_dir,
        reference_replicate,
        n_samples,
        sample_seed,
        replace_sample_indices,
    )
    mean_cif, std_cif = _summarize_predictions(input_dir, n_replicates, x, times)
    return CifPlotData(
        times=times,
        sample_indices=sample_indices,
        true_cif=_compute_true_cif(x, times, dgp_seed),
        mean_cif=mean_cif,
        std_cif=std_cif,
        n_models=n_replicates,
    )


def _row_upper_limit(data: CifPlotData, cause_index: int) -> float:
    true_max = float(data.true_cif[:, cause_index, :].max().item())
    prediction_max = max(
        float(
            (
                data.mean_cif[method_key][:, cause_index, :]
                + data.std_cif[method_key][:, cause_index, :]
            )
            .max()
            .item()
        )
        for method_key in METHOD_KEYS
    )
    return min(1.02, max(0.1, true_max, prediction_max) * 1.08)


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
    for method_key in METHOD_KEYS:
        mean_values = mean_cif[method_key].numpy()
        std_values = std_cif[method_key].numpy()
        axis.plot(
            time_values,
            mean_values,
            color=METHOD_COLORS[method_key],
            linewidth=1.15,
            alpha=0.9,
        )
        axis.errorbar(
            time_values[marker_indices],
            mean_values[marker_indices],
            yerr=std_values[marker_indices],
            fmt=METHOD_MARKERS[method_key],
            color=METHOD_COLORS[method_key],
            markersize=4.2 if method_key != "crsoft" else 5.2,
            elinewidth=0.65,
            capsize=1.5,
            capthick=0.65,
            alpha=0.85,
        )
    axis.grid(color="#d9d9d9", linewidth=0.55, alpha=0.45)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _populate_axes(data: CifPlotData, axes: list[list[Axes]]) -> None:
    n_causes, n_samples = data.true_cif.shape[1], data.true_cif.shape[0]
    for cause_index in range(n_causes):
        for sample_position in range(n_samples):
            axis = axes[cause_index][sample_position]
            _plot_panel(
                axis,
                data.times,
                data.true_cif[sample_position, cause_index, :],
                {
                    method_key: data.mean_cif[method_key][
                        sample_position, cause_index, :
                    ]
                    for method_key in METHOD_KEYS
                },
                {
                    method_key: data.std_cif[method_key][
                        sample_position, cause_index, :
                    ]
                    for method_key in METHOD_KEYS
                },
            )
            axis.set_ylim(0, _row_upper_limit(data, cause_index))
            if cause_index == 0:
                sample_index = int(data.sample_indices[sample_position].item())
                axis.set_title(f"Sample {sample_index}", fontsize=10)
            if sample_position == 0:
                axis.set_ylabel(rf"$F_{{{cause_index + 1}}}(t\mid \mathbf{{x}})$")
            if cause_index == n_causes - 1:
                axis.set_xlabel("Time t")


def _legend_handles() -> list[object]:
    handles = [Line2D([], [], color="#1a1a1a", linewidth=1.8, label="True CIF")]
    handles.extend(
        Line2D(
            [],
            [],
            color=METHOD_COLORS[method_key],
            marker=METHOD_MARKERS[method_key],
            linewidth=1.15,
            label=METHOD_LABELS[method_key],
        )
        for method_key in METHOD_KEYS
    )
    return handles


def _save_figure(data: CifPlotData, output: Path) -> None:
    figure, axes = plt.subplots(
        data.true_cif.shape[1],
        data.true_cif.shape[0],
        figsize=(16.5, 7.8),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    _populate_axes(data, axes.tolist())
    figure.legend(
        handles=_legend_handles(),
        loc="upper center",
        ncol=6,
        frameon=True,
        bbox_to_anchor=(0.5, 0.995),
        title=f"Points and error bars: mean ± 1 SD across {data.n_models} fits",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94), w_pad=0.9, h_pad=0.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    data = _prepare_plot_data(
        args.input_dir,
        args.n_samples,
        args.sample_seed,
        args.reference_replicate,
        args.replace_sample_index,
    )
    _save_figure(data, args.output)
    sample_indices = ", ".join(str(int(index)) for index in data.sample_indices)
    logger.info(f"Selected test samples: {sample_indices}")
    logger.info(f"Aggregated checkpoints per method: {data.n_models}")
    logger.info(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
