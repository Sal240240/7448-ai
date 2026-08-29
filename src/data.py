"""Shared loading/splitting utilities for Phase 1 baselines."""
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


class Dataset:
    """Holds the full aligned feature/target/mask/split tables and slices them by split."""

    def __init__(self, min_taxon_prevalence: float = 0.05):
        self.X = pd.read_parquet(PROCESSED_DIR / "features.parquet")
        self.Y = pd.read_parquet(PROCESSED_DIR / "targets.parquet")
        self.mask = pd.read_parquet(PROCESSED_DIR / "targets_mask.parquet")
        manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet")
        splits = pd.read_parquet(PROCESSED_DIR / "splits.parquet")

        self.sample_split = manifest["subject_id"].map(splits["split"])
        assert self.sample_split.isna().sum() == 0, "every sample's subject should have a split assignment"

        # Drop taxa that are essentially never detected -- mostly single-cohort
        # sequencing artifacts, and they only add noise/compute to a linear
        # per-target fit. Threshold computed on train only to avoid leakage.
        train_rows = self.sample_split[self.sample_split == "train"].index
        prevalence = (self.X.loc[train_rows] != 0).mean()
        self.feature_cols = prevalence[prevalence >= min_taxon_prevalence].index.tolist()
        self.X = self.X[self.feature_cols]

    def split_index(self, split: str) -> pd.Index:
        return self.sample_split[self.sample_split == split].index

    def xy(self, split: str):
        idx = self.split_index(split)
        return self.X.loc[idx], self.Y.loc[idx], self.mask.loc[idx]

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    @property
    def n_targets(self) -> int:
        return self.Y.shape[1]


def pearson_per_column(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pearson r per target column, computed only over that column's measured rows."""
    n_targets = y_true.shape[1]
    out = np.full(n_targets, np.nan)
    for j in range(n_targets):
        m = mask[:, j]
        if m.sum() < 3:
            continue
        yt, yp = y_true[m, j], y_pred[m, j]
        if np.std(yt) == 0 or np.std(yp) == 0:
            continue
        out[j] = np.corrcoef(yt, yp)[0, 1]
    return out
