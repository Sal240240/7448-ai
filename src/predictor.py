"""Serving-side inference for the metabolite predictor.

Loads the sparse elastic-net artifact written by train_production_model.py and
answers two questions the webapp needs:

  1. Given a gut taxonomic composition, what metabolite profile does the model
     predict -- and how much should anyone trust each number?
  2. *Why* that number: which microbes pushed it up or down, and by how much.

(2) is exact here, not an approximation. For a linear model the contribution of
feature j to target i is literally coefficient[i, j] * x[j]; the parts sum to
the prediction. Post-hoc attribution methods (SHAP, LIME) exist to estimate
this for models where it can't be read off directly -- using one here would add
error and a dependency to re-derive something the weights already state.

Every prediction carries its held-out test correlation, because a predicted
value with no error bar is a guess wearing a lab coat. Anything the model
predicts poorly should be visibly marked as such rather than rendered
identically to a well-predicted one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse

from taxonomy import friendly_name, match_key, strip_gtdb_suffix

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Confidence bands for held-out Pearson r. Thresholds are judgment calls, but the
# point is that the UI must distinguish them -- r=0.65 and r=0.12 are not the
# same claim, and rendering both as "predicted: 4.2" is how a tool misleads.
CONFIDENCE_BANDS = ((0.5, "good"), (0.3, "moderate"), (0.15, "weak"))


def confidence_band(test_r: float | None) -> str:
    if test_r is None or np.isnan(test_r):
        return "unvalidated"
    for threshold, label in CONFIDENCE_BANDS:
        if test_r >= threshold:
            return label
    return "very_weak"


def arcsine_sqrt(proportion: np.ndarray) -> np.ndarray:
    """Same variance-stabilizing transform the training features were built with."""
    return np.arcsin(np.sqrt(np.clip(proportion, 0.0, 1.0)))


@dataclass
class Contribution:
    """One taxon's exact signed push on one predicted metabolite."""
    feature: str
    genus: str
    contribution: float
    coefficient: float
    abundance: float


@dataclass
class TargetPrediction:
    hmdb_id: str
    value: float
    test_pearson_r: float | None
    confidence: str
    population_percentile: float | None = None
    contributions: list[Contribution] = field(default_factory=list)


class MetabolitePredictor:
    """Sparse linear metabolite predictor plus exact per-feature attribution."""

    def __init__(self, coef: sparse.csr_matrix, intercepts: np.ndarray, feature_cols: list[str],
                 targets: list[dict]):
        self.coef = coef
        self.intercepts = intercepts
        self.feature_cols = feature_cols
        self.targets = targets
        self.feature_index = {name: i for i, name in enumerate(feature_cols)}
        self.target_index = {t["hmdb_id"]: i for i, t in enumerate(targets)}
        # Genus name -> feature column, so the app can address taxa by the name a
        # person recognizes. Several GTDB lineages can share a normalized genus;
        # first wins. Lineages unresolved to genus are addressable only by their
        # full column name.
        self.genus_index: dict[str, int] = {}
        for name, i in self.feature_index.items():
            key = match_key(name)
            if key:
                self.genus_index.setdefault(key, i)

    @classmethod
    def load(cls, models_dir: Path | str = MODELS_DIR) -> "MetabolitePredictor":
        models_dir = Path(models_dir)
        coef = sparse.load_npz(models_dir / "metabolite_predictor.npz").tocsr()
        intercepts = np.load(models_dir / "metabolite_predictor_intercepts.npy")
        meta = json.loads((models_dir / "metabolite_predictor.json").read_text(encoding="utf-8"))
        return cls(coef, intercepts, meta["feature_cols"], meta["targets"])

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    def vector_from_abundances(self, abundances: dict[str, float], baseline: np.ndarray | None = None) -> np.ndarray:
        """Build a model-space feature vector from genus -> relative abundance (0-1).

        `baseline` matters more than it looks: a user adjusting five sliders has
        not told us the other 5,033 taxa are absent. Zeroing them would feed the
        model a composition no gut has ever had and quietly invalidate the
        prediction, so overrides are applied on top of a real population profile.
        """
        x = np.zeros(len(self.feature_cols), dtype=np.float64) if baseline is None else baseline.astype(np.float64).copy()
        for key, proportion in abundances.items():
            idx = self.feature_index.get(key)
            if idx is None:
                # genus_index keys are suffix-stripped. match_key() expects a
                # full lineage and returns "" for a bare genus like
                # "Blautia_A", so strip directly here instead -- otherwise a
                # suffixed genus name silently resolves to nothing.
                idx = self.genus_index.get(strip_gtdb_suffix(key))
            if idx is None:
                continue
            x[idx] = arcsine_sqrt(np.asarray(float(proportion)))
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicted log1p metabolite abundances for one feature vector."""
        return np.asarray(self.coef @ x).ravel() + self.intercepts

    def explain(self, x: np.ndarray, hmdb_id: str, top_n: int = 8) -> list[Contribution]:
        """Exact signed contributions of each taxon to one metabolite prediction."""
        i = self.target_index.get(hmdb_id)
        if i is None:
            return []
        row = self.coef.getrow(i)
        idx, coefs = row.indices, row.data
        contributions = coefs * x[idx]
        order = np.argsort(-np.abs(contributions))[:top_n]
        return [
            Contribution(
                feature=self.feature_cols[idx[j]],
                genus=friendly_name(self.feature_cols[idx[j]]),
                contribution=float(contributions[j]),
                coefficient=float(coefs[j]),
                abundance=float(x[idx[j]]),
            )
            for j in order
        ]

    def predict_with_explanations(
        self,
        x: np.ndarray,
        hmdb_ids: list[str] | None = None,
        top_n: int = 8,
    ) -> list[TargetPrediction]:
        """Predictions for selected metabolites, each with attribution and confidence.

        Population percentiles are attached downstream by simulate.percentile_of,
        which reads a precomputed quantile grid; this returns the raw predictions
        plus their attribution and confidence.
        """
        values = self.predict(x)
        wanted = hmdb_ids or [t["hmdb_id"] for t in self.targets]
        out = []
        for hmdb_id in wanted:
            i = self.target_index.get(hmdb_id)
            if i is None:
                continue
            meta = self.targets[i]
            test_r = meta.get("test_pearson_r")
            out.append(TargetPrediction(
                hmdb_id=hmdb_id,
                value=float(values[i]),
                test_pearson_r=test_r,
                confidence=confidence_band(test_r),
                contributions=self.explain(x, hmdb_id, top_n=top_n),
            ))
        return out
