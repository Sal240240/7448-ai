"""
Phase 1 comparison: does the MLP earn its complexity over the elastic net
baseline (spec section 5.1)?

Joins both metrics CSVs on the *same* target set -- the elastic net only
covers targets with >= --min-train-measured measured training samples
(qualifying-target selection lives in train_elastic_net.py), so the MLP's
broader coverage (it can produce a number for every target, since its loss
is masked rather than requiring a per-target fit) is restricted down to that
same set here for an apples-to-apples comparison. The MLP's numbers on the
targets *outside* that set are reported separately, since there's no linear
baseline to compare them against.

Output:
    experiments/baseline_comparison.csv

Usage:
    python scripts/compare_baselines.py
"""
from pathlib import Path

import pandas as pd

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


def main() -> None:
    en = pd.read_csv(EXPERIMENTS_DIR / "elastic_net_baseline_metrics.csv")
    mlp = pd.read_csv(EXPERIMENTS_DIR / "mlp_baseline_metrics.csv")

    merged = en.merge(mlp, on="target", suffixes=("_en", "_mlp"))
    merged["r_diff_test"] = merged["pearson_test_mlp"] - merged["pearson_test_en"]
    merged["r_diff_val"] = merged["pearson_val_mlp"] - merged["pearson_val_en"]

    out = EXPERIMENTS_DIR / "baseline_comparison.csv"
    merged.to_csv(out, index=False)

    n = len(merged)
    en_wins = (merged["r_diff_test"] < 0).sum()
    mlp_wins = (merged["r_diff_test"] > 0).sum()

    print(f"Compared on {n} targets (elastic net's qualifying set)\n")
    print(f"{'metric':<28}{'elastic net':>14}{'MLP':>14}")
    print(f"{'median val Pearson r':<28}{merged['pearson_val_en'].median():>14.3f}{merged['pearson_val_mlp'].median():>14.3f}")
    print(f"{'median test Pearson r':<28}{merged['pearson_test_en'].median():>14.3f}{merged['pearson_test_mlp'].median():>14.3f}")
    print(f"{'mean val Pearson r':<28}{merged['pearson_val_en'].mean():>14.3f}{merged['pearson_val_mlp'].mean():>14.3f}")
    print(f"{'mean test Pearson r':<28}{merged['pearson_test_en'].mean():>14.3f}{merged['pearson_test_mlp'].mean():>14.3f}")
    print(f"{'median val MAE':<28}{merged['mae_val_en'].median():>14.3f}{merged['mae_val_mlp'].median():>14.3f}")
    print(f"{'median test MAE':<28}{merged['mae_test_en'].median():>14.3f}{merged['mae_test_mlp'].median():>14.3f}")
    print(f"\nMLP beats elastic net (test r) on {mlp_wins}/{n} targets ({100*mlp_wins/n:.0f}%)")
    print(f"Elastic net beats MLP (test r) on {en_wins}/{n} targets ({100*en_wins/n:.0f}%)")

    mlp_only = mlp[~mlp["target"].isin(en["target"])]
    print(f"\nMLP also produced predictions for {len(mlp_only)} additional targets the elastic net "
          f"couldn't fit (< {100} measured train samples). Their median test r: "
          f"{mlp_only['pearson_test'].median():.3f} (n={mlp_only['pearson_test'].notna().sum()}) "
          f"-- take this set with more caution, less held-out data per target.")

    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
