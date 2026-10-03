#!/usr/bin/env python3
"""
Data loader for DeepHit Synthetic competing risks dataset.

2 competing events: Event1 (1), Event2 (2).
12 raw features → 12 after standardization.
Source: synthetic_comprisk.csv (from DeepHit paper, Lee et al. 2018)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


EPS = 1e-8


def load_data(
    test_size: float = 0.3,
    seed: int = 42,
) -> dict[str, Tensor | int | list[str]]:
    """Load and preprocess Synthetic competing risks dataset.

    Following DKAJ: drop true_time/true_label, standardize features,
    add epsilon to duration to avoid zero times.

    Returns:
        dict with X_train, Y_train, Delta_train, X_test, Y_test, Delta_test,
        K, P, events
    """
    from importlib.resources import files

    import pandas as pd

    # Use importlib.resources so the CSV is reachable both from on-disk source
    # and from inside packaged binaries (e.g. buck @mode/opt par).
    with (files(__package__).joinpath("synthetic_comprisk.csv")).open("rb") as f:
        df = pd.read_csv(f)
    df = df.drop(columns=["true_time", "true_label"]).rename(
        columns={"label": "event", "time": "duration"}
    )
    df["duration"] = df["duration"].astype("float32") + EPS

    feature_cols = [c for c in df.columns if c not in ("event", "duration")]
    features = df[feature_cols].values.astype("float32")
    observed_times = df["duration"].values.astype("float32")
    event_indicators = df["event"].values.astype("int64")

    # Train/test split
    rng = np.random.RandomState(seed)
    n_total = len(features)
    n_test = int(n_total * test_size)
    indices = rng.permutation(n_total)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train_raw = features[train_idx]
    X_test_raw = features[test_idx]

    # Standardize: fit on train, apply to test
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0) + 1e-8
    X_train_np = (X_train_raw - mean) / std
    X_test_np = (X_test_raw - mean) / std

    return {
        "X_train": torch.tensor(X_train_np),
        "Y_train": torch.tensor(observed_times[train_idx]),
        "Delta_train": torch.tensor(event_indicators[train_idx]),
        "X_test": torch.tensor(X_test_np),
        "Y_test": torch.tensor(observed_times[test_idx]),
        "Delta_test": torch.tensor(event_indicators[test_idx]),
        "K": 2,
        "P": len(feature_cols),
        "events": ["Event1", "Event2"],
    }
