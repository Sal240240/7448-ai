"""
Phase 1 baseline: MelonnPan-style elastic net, one model per metabolite.

This is the comparison point spec section 5.1 requires before any deep
architecture is allowed to claim it "earns its complexity": "if you don't
beat the 2019 linear baseline, the deep model isn't earning its complexity
yet." MelonnPan itself fits an independent elastic net per metabolite via
glmnet; this mirrors that (scikit-learn ElasticNetCV instead of glmnet, same
idea) rather than a joint multi-task linear model, since coverage varies
wildly per metabolite (median ~13% of samples, since cohorts used different
metabolomics panels -- see data/processed/metabolites_measured.parquet) and
a joint fit can't handle a different missingness pattern per target.

Only fits metabolites with at least --min-train-measured measured samples in
the training split -- below that, an elastic net's own cross-validated alpha
selection is unreliable, and the number is not worth reporting as a "beat" or
"loss" against it. This is a real, spec-relevant finding on its own: only
~1,072 of 1,384 curated metabolites clear a 200-sample floor, and even fewer
clear denser thresholds. See README's Phase 1 section for the full breakdown.

Output:
    experiments/elastic_net_baseline_metrics.csv  (per-target r/MAE/MSE on val + test)

Usage:
    python scripts/train_elastic_net.py --min-train-measured 100
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNetCV

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import Dataset  # noqa: E402

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


def fit_and_eval_one(target: str, X_train, y_train_full, mask_train_full, X_val, y_val_full, mask_val_full,
                      X_test, y_test_full, mask_test_full):
    m_train = mask_train_full[target].values
    if m_train.sum() < 20:
        return None

    Xt, yt = X_train.values[m_train], y_train_full[target].values[m_train]

    model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=10, cv=3, max_iter=5000, n_jobs=1)
    model.fit(Xt, yt)

    result = {"target": target, "n_train": int(m_train.sum()),
              "alpha": model.alpha_, "l1_ratio": model.l1_ratio_}

    for split_name, X_s, y_full, mask_full in [
        ("val", X_val, y_val_full, mask_val_full),
        ("test", X_test, y_test_full, mask_test_full),
    ]:
        m = mask_full[target].values
        result[f"n_{split_name}"] = int(m.sum())
        if m.sum() < 3:
            result[f"pearson_{split_name}"] = np.nan
            result[f"mae_{split_name}"] = np.nan
            continue
        y_true = y_full[target].values[m]
        y_pred = model.predict(X_s.values[m])
        result[f"pearson_{split_name}"] = (
            np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan
        )
        result[f"mae_{split_name}"] = float(np.mean(np.abs(y_true - y_pred)))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-train-measured", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=None, help="Debug: only fit the first N qualifying targets")
    args = parser.parse_args()

    ds = Dataset()
    X_train, Y_train, M_train = ds.xy("train")
    X_val, Y_val, M_val = ds.xy("val")
    X_test, Y_test, M_test = ds.xy("test")

    n_measured_train = M_train.sum()
    qualifying = n_measured_train[n_measured_train >= args.min_train_measured].index.tolist()
    if args.limit:
        qualifying = qualifying[: args.limit]

    print(f"Features: {ds.n_features}, total targets: {ds.n_targets}, "
          f"qualifying (>= {args.min_train_measured} measured train samples): {len(qualifying)}")

    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(fit_and_eval_one)(t, X_train, Y_train, M_train, X_val, Y_val, M_val, X_test, Y_test, M_test)
        for t in qualifying
    )
    results = [r for r in results if r is not None]
    print(f"\nFit {len(results)} targets in {time.time() - t0:.1f}s")

    df = pd.DataFrame(results)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    out = EXPERIMENTS_DIR / "elastic_net_baseline_metrics.csv"
    df.to_csv(out, index=False)

    print(f"\n-> {out}")
    print(f"Median val Pearson r:  {df['pearson_val'].median():.3f}  (n={df['pearson_val'].notna().sum()})")
    print(f"Median test Pearson r: {df['pearson_test'].median():.3f}  (n={df['pearson_test'].notna().sum()})")
    print(f"Mean val Pearson r:    {df['pearson_val'].mean():.3f}")
    print(f"Mean test Pearson r:   {df['pearson_test'].mean():.3f}")


if __name__ == "__main__":
    main()
