"""
Phase 2 step 8: precompute everything the webapp serves.

The app has to answer, in milliseconds, for a visitor with no sequencing data
of their own: what does a normal gut look like, what happens if I change it,
where do I sit relative to real people, and how much of this should I believe.
That means shipping a handful of derived artifacts rather than making the API
re-read 36MB of parquet per request.

What gets built, and why each one exists:

  - **population reference** -- per-feature mean/SD over healthy-control samples.
    The simulator applies a user's slider changes *on top of* this, because a
    person adjusting five taxa hasn't told us the other 5,033 are absent;
    zeroing them would feed the model a composition no human gut has.

  - **example profiles** -- real samples from the public cohorts, so a first-time
    visitor has something to look at immediately. These are already-published,
    de-identified research data; no new identifiability is introduced by
    including a composition vector and a condition label.

  - **key taxa** -- the slider set. Chosen as the intersection of "common enough
    to be present in most people", "the model actually uses it", and "the
    literature has something to say about it". An arbitrary top-20-by-abundance
    list would include taxa nobody can interpret.

  - **population map** -- a 2D PCA of all samples, plus the components
    themselves, so the backend can project a live simulated profile into the
    same space rather than showing a static picture.

  - **model summary** -- performance numbers read from the actual experiment
    outputs, so the app's honesty claims are generated from results rather than
    typed into a template and left to rot.

Output:
    webapp/data/*.json

Usage:
    python scripts/build_app_data.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from _paths import PROCESSED_DIR, REFERENCE_DIR, ROOT

sys.path.insert(0, str(ROOT / "src"))
from data import Dataset  # noqa: E402
from runlog import log_run  # noqa: E402
from taxonomy import display_name, match_key  # noqa: E402

APP_DATA_DIR = ROOT / "webapp" / "data"
EXPERIMENTS_DIR = ROOT / "experiments"
MODELS_DIR = ROOT / "models"

N_KEY_TAXA = 24
N_EXAMPLES_PER_CONDITION = 2


def write_json(path, payload) -> None:
    """Write atomically: temp file in the same directory, then os.replace.

    A running API reads these files on demand. Writing in place lets a request
    that lands mid-write read a truncated file and fail with a JSON decode
    error -- observed as intermittent 500s on /api/model-info while rebuilding.
    os.replace is atomic on both POSIX and Windows, so a reader sees either the
    old file or the new one, never a partial one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    with log_run("build_app_data") as run:
        ds = Dataset()
        labels = pd.read_parquet(PROCESSED_DIR / "outcome_labels.parquet")
        labels = labels.loc[labels.index.intersection(ds.X.index)]

        meta = json.loads((MODELS_DIR / "metabolite_predictor.json").read_text(encoding="utf-8"))
        model_features = meta["feature_cols"]
        X = ds.X[model_features]

        # --- population reference (healthy controls only) ---
        healthy_idx = labels[labels["role"] == "control"].index.intersection(X.index)
        healthy = X.loc[healthy_idx]
        reference_mean = healthy.mean().values.astype(np.float32)
        reference_sd = healthy.std().replace(0, np.nan).fillna(1e-6).values.astype(np.float32)

        write_json(APP_DATA_DIR / "population_reference.json", {
            "feature_cols": model_features,
            "mean": reference_mean.round(6).tolist(),
            "sd": reference_sd.round(6).tolist(),
            "n_healthy_samples": int(len(healthy)),
            "note": "Mean/SD in arcsine-sqrt space over healthy-control samples only.",
        })

        # --- key taxa: prevalent AND used by the model AND described in the literature ---
        assoc = pd.read_csv(REFERENCE_DIR / "taxon_disease_associations.csv")
        studied = set(assoc[assoc["n_gut_reports"] >= 2]["genus"])

        from scipy import sparse
        coef = sparse.load_npz(MODELS_DIR / "metabolite_predictor.npz").tocsc()
        feature_influence = np.asarray(np.abs(coef).sum(axis=0)).ravel()

        prevalence = (X != 0).mean().values
        candidates = []
        for i, col in enumerate(model_features):
            genus = match_key(col)
            if not genus or genus not in studied or prevalence[i] < 0.30 or feature_influence[i] <= 0:
                continue
            candidates.append({
                "feature": col,
                "genus": genus,
                "display_name": display_name(col),
                "prevalence": round(float(prevalence[i]), 3),
                "influence": float(feature_influence[i]),
                "healthy_mean_abundance": round(float(np.sin(reference_mean[i]) ** 2), 6),
                "n_disease_associations": int((assoc["genus"] == genus).sum()),
            })

        key_taxa = sorted(candidates, key=lambda c: -c["influence"])
        # One entry per genus -- GTDB splits several of them across lineages.
        seen, deduped = set(), []
        for c in key_taxa:
            if c["genus"] in seen:
                continue
            seen.add(c["genus"])
            deduped.append(c)
        key_taxa = deduped[:N_KEY_TAXA]
        write_json(APP_DATA_DIR / "key_taxa.json", key_taxa)

        # --- example profiles: real samples spanning conditions ---
        examples = []
        rng = np.random.default_rng(0)
        for condition, grp in labels[labels["role"].isin(["case", "control"])].groupby("condition"):
            available = grp.index.intersection(X.index)
            if len(available) == 0:
                continue
            chosen = available[rng.permutation(len(available))[:N_EXAMPLES_PER_CONDITION]]
            for idx in chosen:
                cohort, sample_id = idx
                row = X.loc[idx]
                nonzero = row[row > 0]
                examples.append({
                    "id": f"{cohort}:{sample_id}",
                    "cohort": cohort,
                    "condition": condition,
                    "role": grp.loc[idx, "role"],
                    # Only nonzero features travel to the client; the rest are
                    # zeros the backend reconstructs, which keeps this ~40x smaller.
                    "composition": {model_features[i]: round(float(v), 5)
                                    for i, v in zip(np.flatnonzero(row.values), nonzero.values)},
                })
        write_json(APP_DATA_DIR / "example_profiles.json", examples)

        # --- metabolite population distributions ---
        # A predicted value of "9.15" is meaningless to a reader; "higher than
        # 78% of real samples" is not. Stored as 101 quantiles per metabolite,
        # computed over measured values only (a missing value means the cohort's
        # panel didn't include it, not that the level was zero), which supports
        # exact percentile lookup at a fraction of the size of the raw column.
        quantile_grid = np.arange(0, 101) / 100.0
        distributions = {}
        for t in meta["targets"]:
            hmdb_id = t["hmdb_id"]
            if hmdb_id not in ds.Y.columns:
                continue
            measured = ds.Y[hmdb_id].values[ds.mask[hmdb_id].values]
            measured = measured[~np.isnan(measured)]
            if len(measured) < 20:
                continue
            distributions[hmdb_id] = {
                "quantiles": np.round(np.quantile(measured, quantile_grid), 4).tolist(),
                "n_measured": int(len(measured)),
            }
        write_json(APP_DATA_DIR / "metabolite_distributions.json", distributions)

        # --- population map: 2D PCA, plus components for live projection ---
        pca = PCA(n_components=2, random_state=0)
        coords = pca.fit_transform(X.values)
        label_by_idx = labels["condition"].to_dict()
        role_by_idx = labels["role"].to_dict()
        points = [
            {
                "x": round(float(coords[i, 0]), 4),
                "y": round(float(coords[i, 1]), 4),
                "cohort": idx[0],
                "condition": label_by_idx.get(idx, "unlabeled"),
                "role": role_by_idx.get(idx, "excluded"),
            }
            for i, idx in enumerate(X.index)
        ]
        write_json(APP_DATA_DIR / "population_map.json", {
            "points": points,
            "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            "components": np.round(pca.components_, 6).tolist(),
            "mean": np.round(pca.mean_, 6).tolist(),
        })

        # --- model summary, generated from real experiment outputs ---
        baseline = pd.read_csv(EXPERIMENTS_DIR / "elastic_net_baseline_metrics.csv")
        comparison_path = EXPERIMENTS_DIR / "baseline_comparison.csv"
        outcome_path = EXPERIMENTS_DIR / "outcome_model_metrics.csv"
        summary = {
            "dataset": {
                "n_samples": int(len(ds.X)),
                "n_subjects": int(pd.read_parquet(PROCESSED_DIR / "splits.parquet").shape[0]),
                "n_cohorts": int(ds.X.index.get_level_values("cohort").nunique()),
                "n_taxa_features": int(len(model_features)),
                "n_metabolite_targets": int(len(meta["targets"])),
            },
            "metabolite_model": {
                "type": "elastic net, one model per metabolite",
                "median_test_pearson_r": round(float(baseline["pearson_test"].median()), 4),
                "mean_test_pearson_r": round(float(baseline["pearson_test"].mean()), 4),
                "n_targets_r_above_0.5": int((baseline["pearson_test"] >= 0.5).sum()),
                "n_targets_r_above_0.3": int((baseline["pearson_test"] >= 0.3).sum()),
                "n_targets_r_below_0.2": int((baseline["pearson_test"] < 0.2).sum()),
                "trained_on_samples": meta["trained_on"]["n_samples"],
            },
        }
        if comparison_path.exists():
            comp = pd.read_csv(comparison_path)
            if "pearson_test_mlp" in comp.columns:
                summary["mlp_comparison"] = {
                    "median_test_r_elastic_net": round(float(comp["pearson_test_en"].median()), 4)
                    if "pearson_test_en" in comp.columns else None,
                    "median_test_r_mlp": round(float(comp["pearson_test_mlp"].median()), 4),
                }
        # Backtests, where they've been run. These are the numbers that qualify
        # every other number on the site, so the app reads them directly rather
        # than relying on anyone to remember to restate them.
        backtests = {}
        cross_path = EXPERIMENTS_DIR / "backtest_cross_cohort.csv"
        if cross_path.exists():
            cross = pd.read_csv(cross_path)
            per_cohort = cross.groupby("held_out_cohort")["pearson_test"].median()
            backtests["cross_cohort"] = {
                "median_r": round(float(cross["pearson_test"].median()), 4),
                "n_cohorts": int(per_cohort.notna().sum()),
                "n_targets_sampled": int(cross["target"].nunique()),
                "best_cohort_r": round(float(per_cohort.max()), 4),
                "worst_cohort_r": round(float(per_cohort.min()), 4),
            }
        negative_path = EXPERIMENTS_DIR / "backtest_negative_control.csv"
        if negative_path.exists():
            neg = pd.read_csv(negative_path)
            backtests["negative_control"] = {
                "median_real_r": round(float(neg["pearson_real"].median()), 4),
                "median_shuffled_r": round(float(neg["pearson_shuffled"].median()), 4),
                "n_targets": int(len(neg)),
                "passed": bool(abs(neg["pearson_shuffled"].median()) < 0.1),
            }
        if backtests:
            summary["backtests"] = backtests

        if outcome_path.exists():
            om = pd.read_csv(outcome_path)
            by_model = om.groupby("model")["auroc"].median().round(3).to_dict()
            summary["outcome_model"] = {
                "median_auroc_within_cohort": by_model.get("within_cohort"),
                "median_auroc_pooled": by_model.get("pooled"),
                "median_auroc_cohort_identity_only": by_model.get("cohort_identity"),
                "n_validated_conditions": int(
                    ((om["model"] == "within_cohort") & (om["auroc_ci_low"] > 0.5)).sum()
                ),
                "n_condition_cohort_models": int((om["model"] == "within_cohort").sum()),
            }
        write_json(APP_DATA_DIR / "model_summary.json", summary)

        for name in ["population_reference", "key_taxa", "example_profiles", "population_map",
                     "model_summary", "metabolite_distributions"]:
            run.artifact(APP_DATA_DIR / f"{name}.json")
        run.record(
            n_key_taxa=len(key_taxa),
            n_examples=len(examples),
            n_map_points=len(points),
            n_distributions=len(distributions),
            pca_explained_variance=round(float(pca.explained_variance_ratio_.sum()), 4),
            n_healthy_reference=int(len(healthy)),
        )

        print(f"-> {APP_DATA_DIR}")
        print(f"   population_reference.json  {len(healthy)} healthy samples, {len(model_features)} features")
        print(f"   key_taxa.json              {len(key_taxa)} taxa")
        print(f"   example_profiles.json      {len(examples)} real samples across "
              f"{labels['condition'].nunique()} conditions")
        print(f"   population_map.json        {len(points)} points, "
              f"{pca.explained_variance_ratio_.sum():.1%} variance in 2D")
        print(f"   model_summary.json")
        print("\nKey taxa selected for the simulator:")
        for t in key_taxa[:12]:
            print(f"   {t['genus']:24s} prevalence={t['prevalence']:.0%}  "
                  f"assoc={t['n_disease_associations']:3d}  healthy_abundance={t['healthy_mean_abundance']:.4f}")


if __name__ == "__main__":
    main()
