"""
Patient-level train/val/test split (spec section 5.3).

Splits by subject_id, not sample_id -- if a subject contributed multiple
samples they all land in the same split, otherwise patient identity leaks
into the "held-out" set and generalization numbers become meaningless.
Stratifies at the cohort level (assigns whole subjects, grouped by cohort,
so every split gets a representative mix of cohorts) rather than a single
random shuffle, which could otherwise starve small cohorts from one split.

Output:
    data/processed/splits.parquet  (index: subject_id; column: split in {train, val, test})

Usage:
    python scripts/make_splits.py --val-frac 0.15 --test-frac 0.15 --seed 0
"""
import argparse

import numpy as np
import pandas as pd

from _paths import PROCESSED_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet")
    subjects_by_cohort = manifest.reset_index().groupby("cohort")["subject_id"].unique()

    rng = np.random.default_rng(args.seed)
    assignment = {}
    for cohort, subjects in subjects_by_cohort.items():
        subjects = np.array(subjects, dtype=object)
        rng.shuffle(subjects)
        n = len(subjects)
        n_test = max(1, round(n * args.test_frac)) if n >= 4 else 0
        n_val = max(1, round(n * args.val_frac)) if n >= 4 else 0
        test_ids, val_ids, train_ids = subjects[:n_test], subjects[n_test:n_test + n_val], subjects[n_test + n_val:]
        for sid in train_ids:
            assignment[sid] = "train"
        for sid in val_ids:
            assignment[sid] = "val"
        for sid in test_ids:
            assignment[sid] = "test"
        print(f"{cohort:35s} n_subjects={n:4d}  train={len(train_ids):4d} val={len(val_ids):3d} test={len(test_ids):3d}")

    splits = pd.Series(assignment, name="split").rename_axis("subject_id").to_frame()
    out = PROCESSED_DIR / "splits.parquet"
    splits.to_parquet(out)

    print(f"\nTotal subjects: {len(splits)}")
    print(splits["split"].value_counts())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
