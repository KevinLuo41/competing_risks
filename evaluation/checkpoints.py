
from __future__ import annotations

from pathlib import Path

import torch


def _experiments_dir() -> Path:
    """Locate the local project's `experiments/` directory."""
    return Path(__file__).resolve().parent.parent / "experiments"


def get_output_dir(experiment: str) -> Path:
    """Get output directory for an experiment, e.g. 'case1'.

    Returns: Path to experiments/<experiment>/outputs/
    """
    out = _experiments_dir() / experiment / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_results_txt(
    experiment: str, all_results: list[tuple[str, dict[str, float]]]
) -> None:
    """Save all model metrics to a human-readable text file."""
    path = get_output_dir(experiment) / "results.txt"
    with open(path, "w") as f:
        for name, metrics in all_results:
            f.write(f"=== {name} ===\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.6f}\n")
            f.write("\n")
    print(f"  Results saved to {path}")


def save_model(experiment: str, model_name: str, model_data: dict) -> None:
    """Save a single model's state to disk."""
    models_dir = get_output_dir(experiment) / "models"
    models_dir.mkdir(exist_ok=True)
    path = models_dir / f"{model_name}.pt"
    torch.save(model_data, path)
    print(f"  Model saved to {path}")


def load_model(experiment: str, model_name: str) -> dict | None:
    """Load a single model's state from disk."""
    path = get_output_dir(experiment) / "models" / f"{model_name}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    return None


def all_models_saved(experiment: str, model_names: list[str]) -> bool:
    """Check if all model checkpoints exist."""
    models_dir = get_output_dir(experiment) / "models"
    return all((models_dir / f"{name}.pt").exists() for name in model_names)


def save_eval_cache(
    experiment: str,
    model_name: str,
    cif_pred: torch.Tensor,
    metrics: dict[str, float],
) -> None:
    """Cache the predicted CIF tensor + metrics dict for a (case, model)."""
    cache_dir = get_output_dir(experiment) / "eval_cache"
    cache_dir.mkdir(exist_ok=True)
    torch.save(
        {"cif_pred": cif_pred, "metrics": metrics},
        cache_dir / f"{model_name}.pt",
    )


def load_eval_cache(
    experiment: str, model_name: str
) -> tuple[torch.Tensor, dict[str, float]] | None:
    """Load cached (CIF tensor, metrics dict) if present, else None."""
    path = get_output_dir(experiment) / "eval_cache" / f"{model_name}.pt"
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["cif_pred"], data["metrics"]
