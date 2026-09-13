"""Forward outlook: what actually happened to people who started here.

The honest way to say something about the future of a gut profile, given this
data, is not to fit a forecasting curve. With 331 subjects, irregular sampling,
and six cohorts that each followed a different population for a different
reason, a projected trajectory would be a confident-looking line drawn through
almost nothing.

What the data *can* support is an empirical analogue: find the real people whose
starting profile most resembles this one, and report the distribution of what
was actually measured in them later. No model of change is fitted at all -- the
output is a summary of observed outcomes in similar people, with the sample size
and time span attached so the reader can judge how thin it is.

That framing is also the only one that stays honest when the neighbourhood is
sparse. A forecast model always returns a number; this returns "4 similar people
were followed, here is the spread, treat it accordingly."

Similarity is Aitchison-style: Euclidean distance in the arcsine-sqrt space the
model already works in, restricted to prevalent taxa so the distance isn't
dominated by thousands of near-zero columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

# Taxa present in at least this share of samples are used for the distance.
# Rare taxa are mostly single-cohort sequencing artefacts and add noise, not signal.
PREVALENCE_FLOOR = 0.25
DEFAULT_NEIGHBOURS = 12
# Below this many neighbours with follow-up, the spread is not worth reporting
# as anything but "we don't have the data".
MIN_NEIGHBOURS_FOR_SUMMARY = 4


@dataclass
class MetaboliteOutlook:
    hmdb_id: str
    name: str
    n_subjects: int
    median_change: float
    p25_change: float
    p75_change: float
    share_increasing: float
    median_days: float | None


@dataclass
class OutlookResult:
    n_neighbours: int
    median_follow_up_days: float | None
    cohorts: list[str]
    metabolites: list[MetaboliteOutlook] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_neighbours": self.n_neighbours,
            "median_follow_up_days": self.median_follow_up_days,
            "cohorts": self.cohorts,
            "metabolites": [
                {
                    "hmdb_id": m.hmdb_id,
                    "name": m.name,
                    "n_subjects": m.n_subjects,
                    "median_change": round(m.median_change, 4),
                    "p25_change": round(m.p25_change, 4),
                    "p75_change": round(m.p75_change, 4),
                    "share_increasing": round(m.share_increasing, 3),
                    "median_days": m.median_days,
                }
                for m in self.metabolites
            ],
            "notes": self.notes,
        }


@lru_cache(maxsize=1)
def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]] | None:
    """Features, measured metabolites, and the trajectory index, aligned."""
    traj_path = PROCESSED_DIR / "trajectories.parquet"
    if not traj_path.exists():
        return None
    trajectories = pd.read_parquet(traj_path)
    X = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    Y = pd.read_parquet(PROCESSED_DIR / "targets.parquet")

    shared = trajectories.index.intersection(X.index)
    trajectories = trajectories.loc[shared]
    X = X.loc[shared]
    Y = Y.loc[shared]

    prevalence = (X != 0).mean()
    distance_cols = prevalence[prevalence >= PREVALENCE_FLOOR].index.tolist()
    return trajectories, X, Y, distance_cols


def available() -> bool:
    return _load() is not None


def outlook(
    x: np.ndarray,
    feature_cols: list[str],
    hmdb_ids: list[str],
    metabolite_names: dict[str, str] | None = None,
    k: int = DEFAULT_NEIGHBOURS,
) -> OutlookResult:
    """Observed later change in metabolites among subjects who started similarly.

    `x` is a feature vector in the model's own (arcsine-sqrt) space, ordered by
    `feature_cols`.
    """
    loaded = _load()
    if loaded is None:
        return OutlookResult(0, None, [], notes=["No longitudinal data built -- run build_trajectories.py."])
    trajectories, X, Y, distance_cols = loaded
    metabolite_names = metabolite_names or {}

    # Compare against each subject's FIRST sample: the question is "people who
    # started here", so matching a later sample would leak the outcome.
    first_visits = trajectories[trajectories["visit_order"] == 1]
    if first_visits.empty:
        return OutlookResult(0, None, [], notes=["No first-visit samples available."])

    position = {name: i for i, name in enumerate(feature_cols)}
    usable = [c for c in distance_cols if c in position]
    if not usable:
        return OutlookResult(0, None, [], notes=["Feature spaces do not overlap."])

    query = np.array([x[position[c]] for c in usable], dtype=np.float64)
    baseline_matrix = X.loc[first_visits.index, usable].to_numpy(dtype=np.float64)
    distances = np.linalg.norm(baseline_matrix - query, axis=1)

    nearest = np.argsort(distances)[:k]
    neighbour_rows = first_visits.iloc[nearest]
    # subject_key, not subject_id: raw IDs repeat across cohorts.
    neighbour_subjects = neighbour_rows["subject_key"].tolist()

    follow_ups, per_metabolite = [], {h: [] for h in hmdb_ids}
    for subject in neighbour_subjects:
        subject_samples = trajectories[trajectories["subject_key"] == subject].sort_values("visit_order")
        if len(subject_samples) < 2:
            continue
        first_idx, last_idx = subject_samples.index[0], subject_samples.index[-1]
        span = subject_samples["days_elapsed"].iloc[-1]
        if pd.notna(span) and span > 0:
            follow_ups.append(float(span))
        for hmdb_id in hmdb_ids:
            if hmdb_id not in Y.columns:
                continue
            start, end = Y.loc[first_idx, hmdb_id], Y.loc[last_idx, hmdb_id]
            # Both ends must be genuinely measured; a NaN here means this
            # cohort's panel skipped the compound, not that it was absent.
            if pd.notna(start) and pd.notna(end):
                per_metabolite[hmdb_id].append(float(end - start))

    notes = []
    results = []
    for hmdb_id, changes in per_metabolite.items():
        if len(changes) < MIN_NEIGHBOURS_FOR_SUMMARY:
            continue
        arr = np.array(changes)
        results.append(MetaboliteOutlook(
            hmdb_id=hmdb_id,
            name=metabolite_names.get(hmdb_id, hmdb_id),
            n_subjects=len(arr),
            median_change=float(np.median(arr)),
            p25_change=float(np.percentile(arr, 25)),
            p75_change=float(np.percentile(arr, 75)),
            share_increasing=float((arr > 0).mean()),
            median_days=round(float(np.median(follow_ups)), 1) if follow_ups else None,
        ))

    results.sort(key=lambda m: -abs(m.median_change))

    if not results:
        notes.append(
            "None of the selected metabolites were measured at two timepoints in enough "
            "similar subjects to summarise. That is a limit of the source cohorts' panels, "
            "not a finding about this profile."
        )
    if follow_ups and len(follow_ups) < len(neighbour_subjects):
        notes.append(
            f"{len(neighbour_subjects) - len(follow_ups)} of the {len(neighbour_subjects)} "
            "matched subjects record visit order but not real elapsed time, so they contribute "
            "to the direction of change but not the time span."
        )

    return OutlookResult(
        n_neighbours=len(neighbour_subjects),
        median_follow_up_days=round(float(np.median(follow_ups)), 1) if follow_ups else None,
        cohorts=sorted(set(neighbour_rows.index.get_level_values("cohort"))),
        metabolites=results,
        notes=notes,
    )
