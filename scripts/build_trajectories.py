"""
Phase 2 step 10: extract the longitudinal structure hiding in the cohort
metadata.

Six of the fourteen cohorts sampled the same person repeatedly -- iHMP weekly
for up to a year, MARS across eight visits, KOSTIC following infants from
6 weeks to 3 years. That is the only material in this project capable of saying
anything about *change over time* rather than a single snapshot, and none of it
survives into the modeling tables, which carry only subject_id and study_group.

Each cohort encodes time differently, and the differences matter:

    iHMP      week_num          weeks since enrolment   -> days
    MARS      Timepoint         visit index only, real spacing unknown
    KOSTIC    Age (days)        infant age              -> days
    HE        Age (months)      infant age              -> days (x30.44)
    POYET     Collection_Date   calendar date           -> days since subject's first
    WANDRO    day               days since birth/admission

Where only an ordinal index exists (MARS), `days_elapsed` is left null and
`has_absolute_time` is False rather than inventing a spacing. Anything
downstream that needs a real rate of change has to exclude those subjects, and
can now tell which they are.

Output:
    data/processed/trajectories.parquet   (one row per sample, ordered within subject)

Usage:
    python scripts/build_trajectories.py
"""
import sys

import numpy as np
import pandas as pd

from _paths import PROCESSED_DIR, ROOT, BORENSTEIN_DIR

sys.path.insert(0, str(ROOT / "src"))
from runlog import log_run  # noqa: E402

DAYS_PER_MONTH = 30.44

# cohort -> (column, kind). kind says how to turn the column into elapsed days.
TIME_SPECS = {
    "iHMP_IBDMDB_2019": ("week_num", "weeks"),
    "MARS_IBS_2020": ("Timepoint", "ordinal"),
    "KOSTIC_INFANTS_DIABETES_2015": ("Age", "days"),
    "HE_INFANTS_MFGM_2019": ("Age", "months"),
    "POYET_BIO_ML_2019": ("Collection_Date", "date"),
    "WANDRO_PRETERMS_2018": ("day", "days"),
}


def to_days(series: pd.Series, kind: str) -> tuple[pd.Series, bool]:
    """Convert a cohort's time column to numeric days; flag whether it's real time."""
    if kind == "date":
        parsed = pd.to_datetime(series, format="%d-%b-%Y", errors="coerce")
        return parsed.map(lambda d: d.toordinal() if pd.notna(d) else np.nan), True
    numeric = pd.to_numeric(series, errors="coerce")
    if kind == "weeks":
        return numeric * 7.0, True
    if kind == "months":
        return numeric * DAYS_PER_MONTH, True
    if kind == "days":
        return numeric, True
    return numeric, False  # ordinal: an index, not a duration


def main() -> None:
    with log_run("build_trajectories") as run:
        manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet").reset_index()

        rows = []
        for cohort, (column, kind) in TIME_SPECS.items():
            meta_path = BORENSTEIN_DIR / cohort / "metadata.tsv"
            if not meta_path.exists():
                print(f"  ! {cohort}: no metadata.tsv")
                continue
            meta = pd.read_csv(meta_path, sep="\t", dtype=str)
            if column not in meta.columns or "Sample" not in meta.columns:
                print(f"  ! {cohort}: missing '{column}' or 'Sample'")
                continue

            times, absolute = to_days(meta[column], kind)
            frame = pd.DataFrame({
                "cohort": cohort,
                "sample_id": meta["Sample"].astype(str),
                "raw_time": times.values,
                "has_absolute_time": absolute,
            })
            rows.append(frame)
            print(f"  {cohort:34s} {column:16s} ({kind:8s}) "
                  f"{frame['raw_time'].notna().sum():4d}/{len(frame)} samples timed")

        if not rows:
            print("No longitudinal metadata found.")
            run.note("no cohorts yielded time data")
            return

        times = pd.concat(rows, ignore_index=True)
        merged = manifest.merge(times, on=["cohort", "sample_id"], how="inner")
        merged = merged.dropna(subset=["raw_time"])

        # Raw subject IDs are unique only within a cohort -- two cohorts in this
        # collection both number participants A001, A002, ... and share 90 IDs.
        # Grouping on the bare ID would splice two different people's samples
        # into one fabricated trajectory, so every grouping below uses this key.
        merged["subject_key"] = (
            merged["cohort"].astype(str) + "::" + merged["subject_id"].astype(str)
        )

        # Re-express as time since each subject's own first sample, so subjects
        # are comparable regardless of whether the source counted from birth,
        # enrolment, or a calendar date.
        merged["days_elapsed"] = merged.groupby("subject_key")["raw_time"].transform(
            lambda s: s - s.min()
        )
        merged.loc[~merged["has_absolute_time"], "days_elapsed"] = np.nan
        merged["visit_order"] = merged.groupby("subject_key")["raw_time"].rank(method="first").astype(int)
        merged["n_visits"] = merged.groupby("subject_key")["sample_id"].transform("size")

        # A single timed sample is not a trajectory.
        merged = merged[merged["n_visits"] >= 2].copy()
        merged = merged.sort_values(["cohort", "subject_key", "visit_order"])

        out = PROCESSED_DIR / "trajectories.parquet"
        merged.set_index(["cohort", "sample_id"]).to_parquet(out)
        run.artifact(out)

        with_abs = merged[merged["has_absolute_time"]]
        run.record(
            n_samples=len(merged),
            n_subjects=int(merged["subject_key"].nunique()),
            n_cohorts=int(merged["cohort"].nunique()),
            n_subjects_absolute_time=int(with_abs["subject_key"].nunique()),
            median_visits=float(merged.groupby("subject_key").size().median()),
        )

        print(f"\n-> {out}")
        print(f"   {len(merged)} samples from {merged['subject_key'].nunique()} subjects "
              f"with >=2 timepoints, across {merged['cohort'].nunique()} cohorts")
        print(f"   {with_abs['subject_key'].nunique()} subjects have real elapsed time "
              f"(the rest have visit order only)")
        print("\nPer cohort:")
        summary = merged.groupby("cohort").agg(
            subjects=("subject_key", "nunique"),
            samples=("sample_id", "size"),
            median_visits=("n_visits", "median"),
            median_span_days=("days_elapsed", "max"),
        )
        print(summary.round(1).to_string())


if __name__ == "__main__":
    main()
