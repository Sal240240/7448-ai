"""The simulation engine: composition in, interpreted prediction out.

Ties together the four pieces the app needs to answer "what would this gut
profile look like, and what is known about profiles like it":

    composition -> predictor.py   -> predicted metabolites (+ exact attribution)
                -> outcomes.py    -> literature-backed condition signals
                -> population PCA -> where this sits among 2,900 real samples
                -> glossary       -> what any of it means in plain language

Design decisions that matter for honesty:

  - **Adjustments are applied on top of a real healthy-population profile**, not
    onto zeros. Someone moving three sliders has said nothing about the other
    5,035 taxa, and a zero vector is a composition no gut has ever had --
    predictions from it would be extrapolation dressed as inference.

  - **Deltas, not absolutes, are the headline for what-if.** The absolute
    predicted value of a metabolite carries the model's full error; the *change*
    under an intervention is far better determined, because the same systematic
    error largely cancels. The API returns both, with the delta framed as the
    interpretable quantity.

  - **Nothing is reported without its confidence.** Every metabolite carries its
    held-out correlation; every condition signal carries study counts,
    consistency and whether a validated classifier exists for it at all.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from outcomes import OutcomeSignal, outcome_signals, profile_deviations
from predictor import MetabolitePredictor, TargetPrediction, arcsine_sqrt
from taxonomy import display_name

ROOT = Path(__file__).resolve().parent.parent
APP_DATA_DIR = ROOT / "webapp" / "data"
REFERENCE_DIR = ROOT / "data" / "reference"

# Metabolites surfaced by default: the ones a non-expert has a chance of
# recognising and that the glossary can explain. Everything else stays
# queryable, just not front-and-centre.
FEATURED_METABOLITES = [
    "HMDB0000039",  # butyrate
    "HMDB0000042",  # acetate
    "HMDB0000237",  # propionate
    "HMDB0000925",  # TMAO
    "HMDB0002302",  # indole-3-propionate
    "HMDB0000626",  # deoxycholate
    "HMDB0000619",  # cholate
    "HMDB0002271",  # imidazole propionate
    "HMDB0000714",  # hippurate
    "HMDB0000254",  # succinate
    "HMDB0000190",  # lactate
    "HMDB0000682",  # indoxyl sulfate
]


@dataclass
class MapPosition:
    x: float
    y: float


@dataclass
class SimulationResult:
    metabolites: list[TargetPrediction]
    signals: list[OutcomeSignal]
    deviations: dict[str, float]
    map_position: MapPosition
    baseline_metabolites: list[TargetPrediction] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = {
            "metabolites": [_target_to_dict(t) for t in self.metabolites],
            "signals": [_signal_to_dict(s) for s in self.signals],
            "deviations": {k: round(v, 3) for k, v in sorted(
                self.deviations.items(), key=lambda kv: -abs(kv[1])
            )[:40]},
            "map_position": asdict(self.map_position),
            "notes": self.notes,
        }
        if self.baseline_metabolites is not None:
            baseline = {t.hmdb_id: t.value for t in self.baseline_metabolites}
            for entry in payload["metabolites"]:
                base = baseline.get(entry["hmdb_id"])
                entry["baseline_value"] = round(base, 4) if base is not None else None
                entry["delta"] = round(entry["value"] - base, 4) if base is not None else None
        return payload


def _target_to_dict(t: TargetPrediction) -> dict:
    glossary = load_glossary().get(t.hmdb_id, {})
    percentile = t.population_percentile
    if percentile is None:
        percentile = percentile_of(t.hmdb_id, t.value)
    return {
        "hmdb_id": t.hmdb_id,
        "name": glossary.get("display_name") or load_metabolite_names().get(t.hmdb_id, t.hmdb_id),
        "family": glossary.get("family"),
        "value": round(t.value, 4),
        "test_pearson_r": t.test_pearson_r,
        "confidence": t.confidence,
        "population_percentile": round(percentile, 1) if percentile is not None else None,
        "explanation": glossary.get("what_it_is"),
        "why_it_matters": glossary.get("why_it_matters"),
        "drivers": [
            {
                "taxon": c.genus,
                "contribution": round(c.contribution, 4),
                "direction": "increases" if c.contribution > 0 else "decreases",
            }
            for c in t.contributions
        ],
    }


def _signal_to_dict(s: OutcomeSignal) -> dict:
    pubs = load_publication_index()
    citations = []
    for item in s.evidence[:6]:
        for pmid in item.pubmed_ids[:2]:
            pub = pubs.get(pmid)
            if pub and not any(c["pmid"] == pmid for c in citations):
                citations.append({
                    "pmid": pmid,
                    "title": pub.get("title", ""),
                    "first_author": pub.get("first_author", ""),
                    "year": pub.get("year"),
                    "journal": pub.get("journal", ""),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
    return {
        "condition": s.condition,
        "matching_taxa": s.matching_taxa,
        "evidence_weight": s.total_evidence_weight,
        "summary": s.summary(),
        "model_auroc": s.model_auroc,
        "model_auroc_ci": list(s.model_auroc_ci) if s.model_auroc_ci else None,
        "model_validated": s.model_validated,
        "evidence": [
            {
                "taxon": e.genus,
                "observed": e.observed_direction,
                "reported": e.direction,
                "deviation_sd": e.deviation_sd,
                "n_reports": e.n_reports,
                "n_gut_reports": e.n_gut_reports,
                "consistency": e.consistency,
                "study_quality": e.mean_study_quality,
                "latest_year": e.latest_year,
            }
            for e in s.evidence[:8]
        ],
        "citations": citations[:6],
    }


@lru_cache(maxsize=1)
def load_predictor() -> MetabolitePredictor:
    return MetabolitePredictor.load()


@lru_cache(maxsize=1)
def load_population_reference() -> dict:
    return json.loads((APP_DATA_DIR / "population_reference.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_population_map() -> dict:
    return json.loads((APP_DATA_DIR / "population_map.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_metabolite_distributions() -> dict:
    path = APP_DATA_DIR / "metabolite_distributions.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def percentile_of(hmdb_id: str, value: float) -> float | None:
    """Where a predicted value sits in the real measured population, 0-100.

    Quantiles are precomputed on a 1% grid, so this is a lookup rather than a
    scan over the full dataset.
    """
    dist = load_metabolite_distributions().get(hmdb_id)
    if not dist:
        return None
    quantiles = dist["quantiles"]
    idx = int(np.searchsorted(quantiles, value, side="right"))
    return float(min(max(idx, 0), 100))


@lru_cache(maxsize=1)
def load_glossary() -> dict:
    path = REFERENCE_DIR / "metabolite_glossary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("metabolites", {})


@lru_cache(maxsize=1)
def load_metabolite_names() -> dict[str, str]:
    import pandas as pd
    path = REFERENCE_DIR / "metabolite_reference.csv"
    if not path.exists():
        return {}
    ref = pd.read_csv(path)
    return dict(zip(ref["hmdb_id"], ref["name"].fillna(ref["hmdb_id"])))


@lru_cache(maxsize=1)
def load_publication_index() -> dict[str, dict]:
    import pandas as pd
    path = REFERENCE_DIR / "publications.csv"
    if not path.exists():
        return {}
    pubs = pd.read_csv(path, dtype={"pmid": str, "year": "Int64"})
    return {
        r["pmid"]: {**r, "year": int(r["year"]) if r.get("year") is not None and r["year"] == r["year"] else None}
        for r in pubs.to_dict("records")
        if isinstance(r.get("pmid"), str)
    }


@lru_cache(maxsize=1)
def load_key_taxa() -> list[dict]:
    path = APP_DATA_DIR / "key_taxa.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


@lru_cache(maxsize=1)
def load_examples() -> dict[str, dict]:
    path = APP_DATA_DIR / "example_profiles.json"
    if not path.exists():
        return {}
    return {e["id"]: e for e in json.loads(path.read_text(encoding="utf-8"))}


def baseline_vector() -> np.ndarray:
    return np.array(load_population_reference()["mean"], dtype=np.float64)


def project_to_map(x: np.ndarray) -> MapPosition:
    """Project a feature vector into the same 2D space as the population scatter."""
    pmap = load_population_map()
    components = np.array(pmap["components"], dtype=np.float64)
    mean = np.array(pmap["mean"], dtype=np.float64)
    coords = components @ (x - mean)
    return MapPosition(x=round(float(coords[0]), 4), y=round(float(coords[1]), 4))


def vector_from_example(example_id: str) -> np.ndarray | None:
    """Reconstruct a stored example's full feature vector.

    Examples ship only their nonzero features; the omitted ones are genuine
    zeros ("not detected"), not unknowns, so they're restored as zeros rather
    than as population means.
    """
    example = load_examples().get(example_id)
    if example is None:
        return None
    predictor = load_predictor()
    x = np.zeros(len(predictor.feature_cols), dtype=np.float64)
    for feature, value in example["composition"].items():
        idx = predictor.feature_index.get(feature)
        if idx is not None:
            x[idx] = value  # already in arcsine-sqrt space
    return x


def simulate(
    adjustments: dict[str, float] | None = None,
    example_id: str | None = None,
    featured_only: bool = True,
    include_baseline: bool = True,
) -> SimulationResult:
    """Run one simulation.

    `adjustments` maps genus name -> relative abundance (0-1). When `example_id`
    is given, adjustments are applied on top of that real sample instead of the
    healthy-population profile.
    """
    predictor = load_predictor()
    reference = load_population_reference()
    notes: list[str] = []

    if example_id:
        base = vector_from_example(example_id)
        if base is None:
            base = baseline_vector()
            notes.append(f"Unknown example '{example_id}'; used the healthy-population profile instead.")
    else:
        base = baseline_vector()

    x = base.copy()
    if adjustments:
        applied, unknown = 0, []
        for name, proportion in adjustments.items():
            # `or` would be wrong here: feature index 0 is falsy, so a valid
            # adjustment on the first column would silently fall through to the
            # genus lookup and be dropped.
            idx = predictor.feature_index.get(name)
            if idx is None:
                idx = predictor.genus_index.get(name)
            if idx is None:
                unknown.append(name)
                continue
            x[idx] = float(arcsine_sqrt(np.asarray(float(proportion))))
            applied += 1
        if unknown:
            notes.append(f"Ignored {len(unknown)} unrecognised taxa: {', '.join(sorted(unknown)[:5])}")

    hmdb_ids = FEATURED_METABOLITES if featured_only else None
    metabolites = predictor.predict_with_explanations(x, hmdb_ids)
    baseline_metabolites = (
        predictor.predict_with_explanations(base, hmdb_ids) if include_baseline else None
    )

    deviations = profile_deviations(
        x,
        predictor.feature_cols,
        np.array(reference["mean"], dtype=np.float64),
        np.array(reference["sd"], dtype=np.float64),
    )
    signals = outcome_signals(deviations)

    if not signals:
        notes.append(
            "No taxa in this profile deviate far enough from the healthy reference "
            "to trigger a literature-backed signal."
        )

    return SimulationResult(
        metabolites=metabolites,
        signals=signals,
        deviations=deviations,
        map_position=project_to_map(x),
        baseline_metabolites=baseline_metabolites,
        notes=notes,
    )


def outlook_for(
    adjustments: dict[str, float] | None = None,
    example_id: str | None = None,
    k: int = 12,
) -> dict:
    """Empirical forward outlook for a profile -- see trajectory.py for the framing.

    Deliberately a separate call from simulate(): the neighbour search touches
    the full feature table, so folding it into every slider drag would make the
    simulator sluggish for a panel most visitors open once.
    """
    import trajectory

    predictor = load_predictor()
    base = vector_from_example(example_id) if example_id else baseline_vector()
    if base is None:
        base = baseline_vector()

    x = base.copy()
    for name, proportion in (adjustments or {}).items():
        idx = predictor.feature_index.get(name)
        if idx is None:
            idx = predictor.genus_index.get(name)
        if idx is not None:
            x[idx] = float(arcsine_sqrt(np.asarray(float(proportion))))

    result = trajectory.outlook(
        x, predictor.feature_cols, FEATURED_METABOLITES, load_metabolite_names(), k=k
    )
    return result.to_dict()


def taxon_options() -> list[dict]:
    """Slider set for the UI, with healthy-reference abundance as the default."""
    return [
        {
            "genus": t["genus"],
            "display_name": t.get("display_name", t["genus"]),
            "feature": t["feature"],
            "healthy_abundance": t["healthy_mean_abundance"],
            "prevalence": t["prevalence"],
            "n_disease_associations": t["n_disease_associations"],
        }
        for t in load_key_taxa()
    ]
