"""Mapping a gut profile to health-outcome *associations* -- with sources.

This is the layer that answers "so what does my microbiome mean," and it is the
easiest part of this project to get wrong in a way that hurts someone. Two rules
it is built around:

  1. **Association, never diagnosis.** Everything here is of the form "taxa in
     this profile deviate in a direction that published studies have reported in
     condition X, across N studies." That is a literature lookup keyed on the
     user's profile, not a clinical determination, and the output carries the
     citations so a reader can check the claim rather than trust it.

  2. **Show the evidence quality, not just the conclusion.** A signal backed by
     30 consistent studies and one backed by a single 2011 paper render
     differently, because they *are* different. Consistency, study count, recency
     and Disbiome's methodological-quality flags all travel with the signal.

The trained classifiers (train_outcome_model.py) are used only where their
held-out AUROC confidence interval actually excludes chance. Most condition x
cohort pairs don't clear that bar, and a model that can't beat a coin flip on
held-out patients has no business contributing to what a person is told.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from taxonomy import match_key

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

# A taxon has to move meaningfully before its literature associations are worth
# surfacing. Below this, we'd be reading tea leaves off sampling noise.
DEVIATION_THRESHOLD_SD = 0.75
# Disbiome associations thinner than this are kept in the data but not used to
# drive a signal -- one report is a hypothesis, not a finding.
MIN_REPORTS = 2


@dataclass
class EvidenceItem:
    genus: str
    direction: str           # "elevated" / "reduced" -- what the literature reports
    observed_direction: str  # which way THIS profile actually deviates
    n_reports: int
    n_gut_reports: int
    consistency: float
    mean_study_quality: float | None
    latest_year: str
    deviation_sd: float
    pubmed_ids: list[str] = field(default_factory=list)


@dataclass
class OutcomeSignal:
    condition: str
    matching_taxa: int
    total_evidence_weight: float
    evidence: list[EvidenceItem] = field(default_factory=list)
    model_auroc: float | None = None
    model_auroc_ci: tuple[float, float] | None = None
    model_validated: bool = False

    def summary(self) -> str:
        """One plain sentence, deliberately hedged, for non-expert display."""
        taxa = ", ".join(sorted({e.genus for e in self.evidence})[:3])
        return (
            f"{self.matching_taxa} taxa in this profile ({taxa}) deviate in directions "
            f"that published studies have reported in {self.condition}."
        )


@lru_cache(maxsize=1)
def load_associations() -> pd.DataFrame:
    path = REFERENCE_DIR / "taxon_disease_associations.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[(df["n_reports"] >= MIN_REPORTS) & (df["direction"] != "mixed")]


@lru_cache(maxsize=1)
def load_validated_models() -> dict[str, dict]:
    """Condition -> best held-out performance, keeping only models that beat chance.

    'Beat chance' means the bootstrap CI lower bound is above 0.5. On these
    cohort sizes most don't, and that is the honest finding rather than a
    problem to hide.
    """
    path = EXPERIMENTS_DIR / "outcome_model_metrics.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    within = df[df["model"] == "within_cohort"]
    out = {}
    for condition, grp in within.groupby("condition"):
        best = grp.loc[grp["auroc"].idxmax()]
        out[condition] = {
            "auroc": float(best["auroc"]),
            "ci": (float(best["auroc_ci_low"]), float(best["auroc_ci_high"])),
            "cohort": best["cohort"],
            "n_test": int(best["n_test"]),
            "validated": bool(best["auroc_ci_low"] > 0.5),
        }
    return out


# A taxon that is essentially constant across the healthy reference has no
# meaningful scale to measure deviation against. build_app_data.py substitutes a
# 1e-6 epsilon for a zero SD to avoid dividing by zero, which clears a naive
# 1e-9 guard and turns any nonzero abundance into a z-score of 10^3-10^5 --
# enough to dominate the deviation ranking and fabricate a disease signal from a
# taxon that never varies. Require a real amount of variance instead.
MIN_REFERENCE_SD = 1e-4
# Even above that floor, a thin reference can produce implausible z-scores.
# Clipping keeps one artefact from crowding out genuine signal in the ranking.
MAX_ABS_Z = 12.0


def profile_deviations(
    x: np.ndarray,
    feature_cols: list[str],
    reference_mean: np.ndarray,
    reference_sd: np.ndarray,
) -> dict[str, float]:
    """Per-genus deviation from the reference population, in standard deviations.

    Collapsed to genus because that's the resolution the literature layer speaks:
    Disbiome reports "Roseburia", not "d__Bacteria;...;g__Roseburia_A".
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(reference_sd > MIN_REFERENCE_SD, (x - reference_mean) / reference_sd, 0.0)
    z = np.clip(z, -MAX_ABS_Z, MAX_ABS_Z)
    deviations: dict[str, float] = {}
    for i, col in enumerate(feature_cols):
        if not np.isfinite(z[i]) or abs(z[i]) < 1e-9:
            continue
        # Lineages unresolved to genus have no counterpart in a genus-keyed
        # literature database, so they can't drive an association.
        genus = match_key(col)
        if not genus:
            continue
        # Keep the largest-magnitude deviation when GTDB splits one genus across
        # several lineages.
        if genus not in deviations or abs(z[i]) > abs(deviations[genus]):
            deviations[genus] = float(z[i])
    return deviations


def outcome_signals(
    deviations: dict[str, float],
    threshold_sd: float = DEVIATION_THRESHOLD_SD,
    gut_only: bool = True,
    max_signals: int = 8,
) -> list[OutcomeSignal]:
    """Literature-backed condition signals implied by how this profile deviates.

    A taxon contributes to a condition only when the direction it moved matches
    the direction studies report for that condition -- Faecalibacterium being
    *low* matches "reduced in Crohn's"; Faecalibacterium being high does not.
    """
    assoc = load_associations()
    if assoc.empty:
        return []
    if gut_only:
        assoc = assoc[assoc["n_gut_reports"] > 0]

    moved = {g: d for g, d in deviations.items() if abs(d) >= threshold_sd}
    if not moved:
        return []

    relevant = assoc[assoc["genus"].isin(moved.keys())]
    models = load_validated_models()
    condition_lookup = {_normalize_condition(k): v for k, v in models.items()}

    signals = []
    for disease, grp in relevant.groupby("disease"):
        evidence = []
        for _, row in grp.iterrows():
            observed = "elevated" if moved[row["genus"]] > 0 else "reduced"
            if observed != row["direction"]:
                continue
            quality = row.get("mean_study_quality")
            evidence.append(EvidenceItem(
                genus=row["genus"],
                direction=row["direction"],
                observed_direction=observed,
                n_reports=int(row["n_reports"]),
                n_gut_reports=int(row["n_gut_reports"]),
                consistency=float(row["consistency"]),
                mean_study_quality=float(quality) if pd.notna(quality) else None,
                latest_year=str(row.get("latest_year", "")),
                deviation_sd=round(moved[row["genus"]], 2),
                pubmed_ids=[p for p in str(row.get("pubmed_ids", "")).split(";") if p],
            ))
        if not evidence:
            continue

        # Weight by how well-evidenced each matching association is, not just how
        # many matched -- 8 single-report matches shouldn't outrank 2 solid ones.
        #
        # `or 0.5` would be a bug here: a study quality of exactly 0.0 is a real
        # value (the backing studies met none of Disbiome's six methodological
        # flags) and is falsy, so it would be silently promoted to the 0.5
        # "unknown" default -- upgrading the worst-evidenced associations to
        # medium, which is precisely backwards for a layer whose job is to show
        # how well-supported a claim is.
        weight = sum(
            e.consistency
            * np.log1p(e.n_reports)
            * (0.5 if e.mean_study_quality is None else e.mean_study_quality)
            for e in evidence
        )
        model = condition_lookup.get(_normalize_condition(disease))
        signals.append(OutcomeSignal(
            condition=disease,
            matching_taxa=len({e.genus for e in evidence}),
            total_evidence_weight=round(float(weight), 3),
            evidence=sorted(evidence, key=lambda e: -e.n_reports),
            model_auroc=model["auroc"] if model else None,
            model_auroc_ci=model["ci"] if model else None,
            model_validated=bool(model and model["validated"]),
        ))

    signals.sort(key=lambda s: -s.total_evidence_weight)
    return signals[:max_signals]


def _normalize_condition(name: str) -> str:
    """Join Disbiome's 'Crohn's Disease' to our 'crohns_disease'."""
    return re.sub(r"[^a-z]", "", str(name).lower())
