"""
Phase 0 step 2a: combine per-cohort taxonomic profiles into one aligned feature table.

Reads data/raw/borenstein/<COHORT>/genera.tsv (present for all cohorts) and,
where available, species.tsv, applies the arcsine-square-root transform used
throughout this literature for relative-abundance data (spec section 3.2.4),
and unions columns across cohorts so every cohort ends up with the same
feature space (missing taxa filled with 0 -- "not detected", which is the
correct interpretation for relative-abundance profiling, unlike metabolites
where a gap can mean "not measured" -- see standardize_metabolites.py).

Output:
    data/processed/taxonomy_genus.parquet   (index: cohort, sample_id)
    data/processed/taxonomy_species.parquet (subset of cohorts that have species-level data)

Usage:
    python scripts/standardize_taxonomy.py
"""
import numpy as np
import pandas as pd

from _paths import RAW_DIR, PROCESSED_DIR, list_cohort_dirs


def arcsine_sqrt(df: pd.DataFrame) -> pd.DataFrame:
    """Variance-stabilizing transform for compositional (relative-abundance) data."""
    return np.arcsin(np.sqrt(df.clip(lower=0, upper=1)))


def load_level(level: str) -> pd.DataFrame:
    frames = []
    for cohort_dir in list_cohort_dirs():
        fpath = cohort_dir / f"{level}.tsv"
        if not fpath.exists():
            fpath_zip = cohort_dir / f"{level}.tsv.zip"
            if not fpath_zip.exists():
                continue
            fpath = fpath_zip
        df = pd.read_csv(fpath, sep="\t", dtype={"Sample": str})
        df = df.rename(columns={"Sample": "sample_id"}).set_index("sample_id")
        df.insert(0, "cohort", cohort_dir.name)
        frames.append(df)
        print(f"  loaded {level} for {cohort_dir.name}: {df.shape[0]} samples x {df.shape[1] - 1} taxa")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=0)
    combined = combined.set_index("cohort", append=True).reorder_levels(["cohort", "sample_id"])

    taxa_cols = combined.columns
    combined[taxa_cols] = combined[taxa_cols].fillna(0.0)
    combined[taxa_cols] = arcsine_sqrt(combined[taxa_cols].astype("float64")).astype("float32")
    return combined


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Genus-level taxonomy (all cohorts):")
    genus = load_level("genera")
    out = PROCESSED_DIR / "taxonomy_genus.parquet"
    genus.to_parquet(out)
    print(f"-> {out}  shape={genus.shape}")

    print("\nSpecies-level taxonomy (subset of cohorts):")
    species = load_level("species")
    if not species.empty:
        out = PROCESSED_DIR / "taxonomy_species.parquet"
        species.to_parquet(out)
        print(f"-> {out}  shape={species.shape}")
    else:
        print("No species-level files found -- run fetch_borenstein.py without --skip-species.")


if __name__ == "__main__":
    main()
