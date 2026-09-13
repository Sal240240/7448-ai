"""
Phase 2 step 7: fit the metabolite predictor that actually gets served.

`train_elastic_net.py` answers "how well does this approach work" -- it fits
ElasticNetCV per metabolite and records metrics, but never persists a model, so
there is nothing for an application to load. This fits the deployable version.

Two differences from the baseline script, both deliberate:

  - **No CV search.** The baseline already selected (alpha, l1_ratio) per target
    and recorded them in experiments/elastic_net_baseline_metrics.csv. Refitting
    with those fixed is ~90x less compute than redoing a 3-l1_ratio x 10-alpha x
    3-fold search, and reuses hyperparameters that were chosen on training data
    only.

  - **Fits on train+val.** The baseline holds val out to compare architectures;
    that comparison is done, so the served model gets the extra ~260 patients.
    Test stays untouched -- the reported accuracy still comes from patients this
    model has never seen.

The artifact is a sparse coefficient matrix rather than 1,098 pickled estimator
objects: elastic net solutions are mostly zeros, it loads in milliseconds, a
prediction is one sparse matmul, and -- the reason that matters here -- the
coefficients *are* the explanation. The "which microbes drove this prediction"
view in the webapp reads these directly rather than approximating them with a
post-hoc attribution method.

Output:
    models/metabolite_predictor.npz    (sparse coefficients + intercepts)
    models/metabolite_predictor.json   (feature/target names, per-target accuracy)

Usage:
    python scripts/train_production_model.py
    python scripts/train_production_model.py --min-test-r 0.2   # only ship targets this good
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.linear_model import ElasticNet

from _paths import ROOT

sys.path.insert(0, str(ROOT / "src"))
from data import Dataset  # noqa: E402
from runlog import log_run  # noqa: E402

EXPERIMENTS_DIR = ROOT / "experiments"
MODELS_DIR = ROOT / "models"


def fit_one(target: str, alpha: float, l1_ratio: float, X, y, mask) -> dict | None:
    """Fit one metabolite's model on its measured rows only.

    A missing metabolite value means "this cohort's panel didn't include this
    compound," not "zero" -- so each target trains on a different subset of rows.
    """
    m = mask[target].values
    if m.sum() < 20:
        return None
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
    model.fit(X.values[m], y[target].values[m])
    coef = model.coef_
    nz = np.flatnonzero(coef)
    return {
        "target": target,
        "indices": nz.astype(np.int32),
        "values": coef[nz].astype(np.float32),
        "intercept": float(model.intercept_),
        "n_train": int(m.sum()),
        "n_nonzero": int(len(nz)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-jobs", type=int, default=3, help="Parallel fits (default 3, to leave the machine usable)")
    parser.add_argument("--min-test-r", type=float, default=None,
                        help="Only include targets whose baseline test Pearson r clears this")
    parser.add_argument("--limit", type=int, default=None, help="Debug: first N targets only")
    args = parser.parse_args()

    with log_run("train_production_model", params={"min_test_r": args.min_test_r, "n_jobs": args.n_jobs}) as run:
        metrics_path = EXPERIMENTS_DIR / "elastic_net_baseline_metrics.csv"
        if not metrics_path.exists():
            print(f"{metrics_path} not found -- run train_elastic_net.py first.")
            return
        baseline = pd.read_csv(metrics_path)

        if args.min_test_r is not None:
            before = len(baseline)
            baseline = baseline[baseline["pearson_test"] >= args.min_test_r]
            print(f"Filtered to {len(baseline)}/{before} targets with baseline test r >= {args.min_test_r}")
        if args.limit:
            baseline = baseline.head(args.limit)

        ds = Dataset()
        # Train on train+val; test stays held out so reported accuracy is honest.
        fit_idx = ds.sample_split[ds.sample_split.isin(["train", "val"])].index
        X, Y, M = ds.X.loc[fit_idx], ds.Y.loc[fit_idx], ds.mask.loc[fit_idx]

        targets = [t for t in baseline["target"] if t in Y.columns]
        params = baseline.set_index("target")[["alpha", "l1_ratio"]].to_dict("index")

        print(f"Fitting {len(targets)} metabolite models on {len(fit_idx)} samples x {X.shape[1]} features")
        t0 = time.time()
        results = Parallel(n_jobs=args.n_jobs, verbose=1)(
            delayed(fit_one)(t, params[t]["alpha"], params[t]["l1_ratio"], X, Y, M) for t in targets
        )
        results = [r for r in results if r is not None]
        elapsed = time.time() - t0
        print(f"Fit {len(results)} models in {elapsed:.0f}s")

        # Assemble one sparse (n_targets x n_features) coefficient matrix.
        rows, cols, vals = [], [], []
        for i, r in enumerate(results):
            rows.extend([i] * len(r["indices"]))
            cols.extend(r["indices"].tolist())
            vals.extend(r["values"].tolist())
        coef = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(len(results), X.shape[1]), dtype=np.float32
        )
        intercepts = np.array([r["intercept"] for r in results], dtype=np.float32)

        MODELS_DIR.mkdir(exist_ok=True)
        npz_path = MODELS_DIR / "metabolite_predictor.npz"
        sparse.save_npz(npz_path, coef)
        np.save(MODELS_DIR / "metabolite_predictor_intercepts.npy", intercepts)
        run.artifact(npz_path)

        accuracy = baseline.set_index("target")[["pearson_test", "mae_test", "n_test"]].to_dict("index")
        meta = {
            "feature_cols": list(X.columns),
            "targets": [
                {
                    "hmdb_id": r["target"],
                    "n_train": r["n_train"],
                    "n_nonzero_coef": r["n_nonzero"],
                    # Held-out accuracy from the baseline run, carried alongside the
                    # model so the app can never show a prediction without its error bar.
                    "test_pearson_r": _clean_float(accuracy.get(r["target"], {}).get("pearson_test")),
                    "test_mae": _clean_float(accuracy.get(r["target"], {}).get("mae_test")),
                    "n_test": int(accuracy.get(r["target"], {}).get("n_test") or 0),
                }
                for r in results
            ],
            "trained_on": {"n_samples": int(len(fit_idx)), "splits": ["train", "val"]},
            "n_features": int(X.shape[1]),
        }
        json_path = MODELS_DIR / "metabolite_predictor.json"
        json_path.write_text(json.dumps(meta), encoding="utf-8")
        run.artifact(json_path)

        density = coef.nnz / (coef.shape[0] * coef.shape[1])
        usable_r = [t["test_pearson_r"] for t in meta["targets"] if t["test_pearson_r"] is not None]
        run.record(
            n_targets=len(results),
            n_features=int(X.shape[1]),
            n_train_samples=int(len(fit_idx)),
            coef_density=round(density, 5),
            median_test_r=round(float(np.median(usable_r)), 4) if usable_r else None,
            fit_seconds=round(elapsed, 1),
        )

        print(f"\n-> {npz_path}  ({coef.shape[0]} x {coef.shape[1]}, {coef.nnz} nonzero, density {density:.3%})")
        print(f"-> {json_path}")
        if usable_r:
            print(f"\nHeld-out accuracy of shipped targets: median r = {np.median(usable_r):.3f}")
            print(f"  targets with r >= 0.5: {sum(r >= 0.5 for r in usable_r)}")
            print(f"  targets with r >= 0.3: {sum(r >= 0.3 for r in usable_r)}")
            print(f"  targets with r <  0.2: {sum(r < 0.2 for r in usable_r)}  (weak -- flag these in any UI)")


def _clean_float(value) -> float | None:
    """NaN isn't valid JSON; the app needs an explicit null for 'no test estimate'."""
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else round(value, 4)


if __name__ == "__main__":
    main()
