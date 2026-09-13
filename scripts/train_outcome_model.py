"""
Phase 2 step 6: the outcome model -- can a gut taxonomic profile tell us which
condition a person's sample resembles, and how much of that is real?

This is the piece that turns metabolite prediction into something a person can
interpret ("this profile resembles the IBD reference group"), so it is also the
piece most likely to mislead if built carelessly. Three models are trained per
condition, deliberately, because only the comparison between them is honest:

  1. `within_cohort`   -- case vs. control inside a single cohort, held-out
                          patients from that same cohort. Batch is constant by
                          construction, so whatever AUROC survives here is
                          biology. **This is the headline number.**

  2. `pooled`          -- the naive version: train on every cohort at once and
                          evaluate on held-out patients. This is what you get if
                          you don't think about batch effects, and it usually
                          looks better than (1).

  3. `cohort_identity` -- the control that explains why. Features are ONLY a
                          one-hot encoding of which cohort the sample came from,
                          no biology at all. Because each cohort studies one
                          disease on one platform, this scores far above chance.
                          Any gap between (2) and (3) is what the microbiome
                          actually added; where (2) ~= (3), the pooled model
                          learned the lab, not the patient.

Splits are by subject_id (reusing make_splits.py's patient-level assignment),
so repeat samples from one person never straddle train and test.

Output:
    experiments/outcome_model_metrics.csv
    models/outcome_models.joblib          (within-cohort models, for serving)

Usage:
    python scripts/train_outcome_model.py
    python scripts/train_outcome_model.py --min-cases 20
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _paths import PROCESSED_DIR, ROOT

sys.path.insert(0, str(ROOT / "src"))
from data import Dataset  # noqa: E402
from runlog import log_run  # noqa: E402

EXPERIMENTS_DIR = ROOT / "experiments"
MODELS_DIR = ROOT / "models"


def bootstrap_auroc_ci(y_true, prob, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for AUROC.

    Several held-out arms here have only 8-20 patients. A bare "AUROC 0.96" off
    12 samples is close to meaningless, and printing it without an interval is
    how a noisy result gets quoted as a finding. Resamples the test set with
    replacement, skipping draws that end up single-class.
    """
    rng = np.random.default_rng(seed)
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], prob[idx]))
    if not scores:
        return float("nan"), float("nan")
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def fit_eval(X_tr, y_tr, X_te, y_te, seed: int = 0) -> tuple[object, dict]:
    """L2 logistic regression on standardized features; AUROC/AUPRC on held-out rows.

    Regularized linear rather than a tree ensemble because n_features (5k) far
    exceeds n_samples per condition (tens to low hundreds) -- the same regime
    where the repo's elastic net already beats a deep net on the metabolite task.
    """
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return None, {}
    model = make_pipeline(
        StandardScaler(),
        # l1_ratio=0 is sklearn >=1.8's spelling of the former penalty="l2".
        LogisticRegression(l1_ratio=0, C=0.1, max_iter=2000, class_weight="balanced", random_state=seed),
    )
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]
    ci_low, ci_high = bootstrap_auroc_ci(y_te, prob, seed=seed)
    return model, {
        "auroc": float(roc_auc_score(y_te, prob)),
        "auroc_ci_low": round(ci_low, 3),
        "auroc_ci_high": round(ci_high, 3),
        "auprc": float(average_precision_score(y_te, prob)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_test_positive": int(y_te.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-cases", type=int, default=20, help="Skip conditions with fewer cases than this")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with log_run("train_outcome_model", params={"min_cases": args.min_cases, "seed": args.seed}) as run:
        ds = Dataset()
        labels = pd.read_parquet(PROCESSED_DIR / "outcome_labels.parquet")

        X = ds.X
        labels = labels.loc[labels.index.intersection(X.index)]
        # cohort is an index level on outcome_labels.parquet; grouping/filtering
        # below is clearer against a plain column.
        labels["cohort"] = labels.index.get_level_values("cohort")
        split = ds.sample_split

        usable = labels[labels["role"].isin(["case", "control"]) & labels["comparison_group"]]
        conditions = [
            c for c, n in usable[usable["role"] == "case"]["condition"].value_counts().items()
            if n >= args.min_cases
        ]
        print(f"Features: {X.shape[1]}, labeled samples: {len(usable)}, "
              f"conditions with >={args.min_cases} cases: {len(conditions)}")

        results, fitted = [], {}

        for condition in conditions:
            case_rows = usable[(usable["condition"] == condition)]
            cohorts = sorted(case_rows["cohort"].unique())

            # --- 1. within-cohort: the honest number ---
            for cohort in cohorts:
                arm = usable[usable["cohort"] == cohort]
                arm = arm[(arm["condition"] == condition) | (arm["role"] == "control")]
                y = (arm["condition"] == condition).astype(int)
                idx = arm.index
                tr = idx[split.loc[idx].isin(["train", "val"])]
                te = idx[split.loc[idx] == "test"]
                if len(te) < 8 or y.loc[tr].sum() < 5:
                    continue
                model, metrics = fit_eval(X.loc[tr].values, y.loc[tr].values, X.loc[te].values, y.loc[te].values, args.seed)
                if not metrics:
                    continue
                results.append({"condition": condition, "model": "within_cohort", "cohort": cohort, **metrics})
                fitted[(condition, cohort)] = model
                print(f"  {condition:28s} {cohort:34s} AUROC={metrics['auroc']:.3f} "
                      f"[{metrics['auroc_ci_low']:.2f}-{metrics['auroc_ci_high']:.2f}]  n_test={metrics['n_test']}")

            # --- 2. pooled across cohorts: the naive number ---
            pooled = usable[(usable["condition"] == condition) | (usable["role"] == "control")]
            y = (pooled["condition"] == condition).astype(int)
            idx = pooled.index
            tr = idx[split.loc[idx].isin(["train", "val"])]
            te = idx[split.loc[idx] == "test"]
            if len(te) >= 8 and y.loc[tr].sum() >= 5:
                _, metrics = fit_eval(X.loc[tr].values, y.loc[tr].values, X.loc[te].values, y.loc[te].values, args.seed)
                if metrics:
                    results.append({"condition": condition, "model": "pooled", "cohort": "ALL", **metrics})

                # --- 3. cohort identity only: how much of (2) is just batch ---
                cohort_dummies = pd.get_dummies(pooled["cohort"]).astype(float)
                _, batch_metrics = fit_eval(
                    cohort_dummies.loc[tr].values, y.loc[tr].values,
                    cohort_dummies.loc[te].values, y.loc[te].values, args.seed,
                )
                if batch_metrics:
                    results.append({"condition": condition, "model": "cohort_identity", "cohort": "ALL", **batch_metrics})

        if not results:
            print("No conditions produced a usable model.")
            run.note("no usable conditions")
            return

        df = pd.DataFrame(results)
        EXPERIMENTS_DIR.mkdir(exist_ok=True)
        out = EXPERIMENTS_DIR / "outcome_model_metrics.csv"
        df.to_csv(out, index=False)
        run.artifact(out)

        MODELS_DIR.mkdir(exist_ok=True)
        import joblib
        model_path = MODELS_DIR / "outcome_models.joblib"
        joblib.dump({"models": fitted, "feature_cols": ds.feature_cols}, model_path, compress=3)
        run.artifact(model_path)

        summary = df.groupby("model")["auroc"].agg(["median", "mean", "count"]).round(3)
        within = df[df["model"] == "within_cohort"]["auroc"]
        pooled_auroc = df[df["model"] == "pooled"]["auroc"]
        batch_auroc = df[df["model"] == "cohort_identity"]["auroc"]

        run.record(
            median_within_cohort_auroc=float(within.median()) if len(within) else None,
            median_pooled_auroc=float(pooled_auroc.median()) if len(pooled_auroc) else None,
            median_cohort_identity_auroc=float(batch_auroc.median()) if len(batch_auroc) else None,
            n_conditions=len(conditions),
            n_within_cohort_models=int(len(within)),
        )

        print(f"\n-> {out}")
        print(f"-> {model_path}")
        print("\nAUROC by model type:")
        print(summary.to_string())
        print(
            "\nRead this as: `cohort_identity` uses no biology at all -- it only knows which\n"
            "lab ran the sample. Wherever `pooled` fails to clear it by a wide margin, the\n"
            "pooled model is mostly reading batch signature, which is why `within_cohort`\n"
            "is the number this project reports."
        )


if __name__ == "__main__":
    main()
