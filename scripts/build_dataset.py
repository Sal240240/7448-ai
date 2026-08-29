"""
Phase 0 step 3: join taxonomy features with metabolite targets into the
final modeling-ready tables.

Joins on (cohort, sample_id). Samples with taxonomy but no metabolomics (or
vice versa) are dropped -- we can only train/evaluate on paired samples.
Also emits a per-sample manifest (cohort, sample_id, subject_id, study_group)
pulled from metadata.tsv, which downstream training needs for the
patient-level train/val/test split (spec section 5.3 -- split by subject_id,
never by sample_id, or patient identity leaks across the split).

Output:
    data/processed/features.parquet   (genus-level taxonomy; the common feature space)
    data/processed/targets.parquet    (log1p metabolite abundance, HMDB-indexed)
    data/processed/targets_mask.parquet (which targets were actually measured per cohort)
    data/processed/manifest.parquet   (cohort, sample_id, subject_id, study_group)

Usage:
    python scripts/build_dataset.py
"""
import pandas as pd

from _paths import PROCESSED_DIR, list_cohort_dirs


def load_manifest() -> pd.DataFrame:
    rows = []
    for cohort_dir in list_cohort_dirs():
        meta = pd.read_csv(cohort_dir / "metadata.tsv", sep="\t", dtype={"Sample": str, "Subject": str})
        meta = meta.rename(columns={"Sample": "sample_id", "Subject": "subject_id", "Study.Group": "study_group"})
        meta["cohort"] = cohort_dir.name
        if "study_group" not in meta.columns:
            # e.g. POYET_BIO_ML_2019: a single-subject healthy longitudinal cohort, no case/control groups
            meta["study_group"] = pd.NA
        meta["study_group"] = meta["study_group"].astype(str).replace("nan", pd.NA).replace("<NA>", pd.NA)
        rows.append(meta[["cohort", "sample_id", "subject_id", "study_group"]])
    return pd.concat(rows, axis=0).set_index(["cohort", "sample_id"])


def main() -> None:
    taxonomy = pd.read_parquet(PROCESSED_DIR / "taxonomy_genus.parquet")
    targets = pd.read_parquet(PROCESSED_DIR / "metabolites.parquet")
    mask = pd.read_parquet(PROCESSED_DIR / "metabolites_measured.parquet")
    manifest = load_manifest()

    paired_index = taxonomy.index.intersection(targets.index)
    print(f"Taxonomy samples: {len(taxonomy)}, metabolomics samples: {len(targets)}, paired: {len(paired_index)}")

    features = taxonomy.loc[paired_index]
    targets = targets.loc[paired_index]
    mask = mask.loc[paired_index]
    manifest = manifest.loc[manifest.index.intersection(paired_index)].loc[paired_index]

    n_subjects = manifest["subject_id"].nunique()
    print(f"Unique subjects across paired samples: {n_subjects}")

    features.to_parquet(PROCESSED_DIR / "features.parquet")
    targets.to_parquet(PROCESSED_DIR / "targets.parquet")
    mask.to_parquet(PROCESSED_DIR / "targets_mask.parquet")
    manifest.to_parquet(PROCESSED_DIR / "manifest.parquet")

    print(f"\n-> features.parquet   {features.shape}")
    print(f"-> targets.parquet    {targets.shape}")
    print(f"-> targets_mask.parquet {mask.shape}")
    print(f"-> manifest.parquet   {manifest.shape}")


if __name__ == "__main__":
    main()
