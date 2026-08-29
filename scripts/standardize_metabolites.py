"""
Phase 0 step 2b: map metabolites to HMDB IDs and combine across cohorts.

The Borenstein collection ships an mtb.map.tsv per cohort mapping each
platform-specific compound (e.g. "HILIC-neg_Cluster_0480: 1-3-7-trimethylurate")
to an HMDB ID where annotation confidence allows -- this does the job spec
section 3.2.2 asks for (a single cross-cohort metabolite ontology), we just
need to apply it and drop unannotated features (spec section 4: "don't try to
predict unannotated spectral features in v1 -- you can't validate what you
can't name").

Values are log1p-transformed (spec section 3.2.4). Missing values are left as
NaN rather than 0 and tracked in a companion "measured" mask, because a gap
here means "this cohort's metabolomics panel didn't include this compound",
not "this compound is absent" -- collapsing that to 0 would poison training
with false negatives. Downstream training should mask the loss accordingly.

When multiple platform compounds map to the same HMDB ID within one cohort
(seen on some panels, e.g. redundant adducts/clusters), they're averaged.

Output:
    data/processed/metabolites.parquet       (index: cohort, sample_id; columns: HMDB IDs; log1p abundance)
    data/processed/metabolites_measured.parquet  (same shape, boolean: was this HMDB ID on this cohort's panel)

Usage:
    python scripts/standardize_metabolites.py
"""
import numpy as np
import pandas as pd

from _paths import PROCESSED_DIR, list_cohort_dirs


def load_cohort_metabolites(cohort_dir) -> pd.DataFrame | None:
    map_path = cohort_dir / "mtb.map.tsv"
    mtb_path = cohort_dir / "mtb.tsv"
    if not mtb_path.exists():
        mtb_path = cohort_dir / "mtb.tsv.zip"
    if not map_path.exists() or not mtb_path.exists():
        return None

    cmap = pd.read_csv(map_path, sep="\t")
    cmap = cmap.dropna(subset=["HMDB"])
    compound_to_hmdb = dict(zip(cmap["Compound"], cmap["HMDB"]))

    mtb = pd.read_csv(mtb_path, sep="\t", dtype={"Sample": str}).rename(columns={"Sample": "sample_id"}).set_index("sample_id")
    keep_cols = [c for c in mtb.columns if c in compound_to_hmdb]
    mtb = mtb[keep_cols].rename(columns=compound_to_hmdb)

    # collapse duplicate HMDB ids (multiple compounds annotated to the same metabolite)
    mtb = mtb.T.groupby(level=0).mean().T
    return mtb


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for cohort_dir in list_cohort_dirs():
        mtb = load_cohort_metabolites(cohort_dir)
        if mtb is None:
            print(f"  ! skipping {cohort_dir.name}: missing mtb.tsv/mtb.map.tsv")
            continue
        mtb.insert(0, "cohort", cohort_dir.name)
        frames.append(mtb)
        print(f"  loaded metabolites for {cohort_dir.name}: {mtb.shape[0]} samples x {mtb.shape[1] - 1} HMDB-annotated compounds")

    if not frames:
        print("No cohorts found -- run fetch_borenstein.py first.")
        return

    combined = pd.concat(frames, axis=0)
    combined = combined.set_index("cohort", append=True).reorder_levels(["cohort", "sample_id"])

    measured = combined.notna()
    values = np.log1p(combined.clip(lower=0)).astype("float32")

    out_values = PROCESSED_DIR / "metabolites.parquet"
    out_mask = PROCESSED_DIR / "metabolites_measured.parquet"
    values.to_parquet(out_values)
    measured.to_parquet(out_mask)

    print(f"\n-> {out_values}  shape={values.shape}")
    print(f"-> {out_mask}  shape={measured.shape}")
    n_cohorts = combined.index.get_level_values("cohort").nunique()
    coverage = measured.mean().sort_values(ascending=False)
    print(f"\nMetabolites measured in all {n_cohorts} cohorts: {(coverage == 1.0).sum()}")
    print(f"Metabolites measured in only 1 cohort: {(coverage <= coverage.min() + 1e-9).sum()}")


if __name__ == "__main__":
    main()
