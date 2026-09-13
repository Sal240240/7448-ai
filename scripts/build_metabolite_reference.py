"""
Phase 2 step 1: consolidate the per-cohort mtb.map.tsv files into one
cross-cohort metabolite reference table.

The modeling pipeline reduces every metabolite to a bare HMDB ID, which is the
right call for training (it's the only stable cross-cohort key) but useless for
anything a human reads: "HMDB0000039" means nothing, "butyrate" means a lot.
This pulls the human-readable layer out once, so the outcome layer, the
backtests, and the webapp all read the same names instead of each re-parsing
14 raw files.

The 14 annotation files do NOT share a schema -- only `Compound`, `HMDB`,
`KEGG`, and `High.Confidence.Annotation` appear in all of them. Names live in
`Compound.Name` in 5 cohorts and are embedded in the `Compound` string
("HILIC-neg_Cluster_0480: 1-3-7-trimethylurate") in the rest. Chemical class
is worse: `Putative.Chemical.Class` exists only in FRANZOSA and KIM uses a
different taxonomy entirely (`Superpathway`/`Subpathway`). So cohort-provided
class is kept only as a hint -- the authoritative classification comes from
KEGG BRITE in fetch_kegg.py, which covers every metabolite carrying a KEGG ID
rather than just one cohort's panel.

Cross-cohort name conflicts (same HMDB ID, 14 annotation files) resolve to the
most common spelling, tie-broken toward the shortest, since platform-specific
names tend to be the longer decorated ones.

Output:
    data/reference/metabolite_reference.csv   (checked in; one row per HMDB ID)

Usage:
    python scripts/build_metabolite_reference.py
"""
from collections import Counter

import pandas as pd

from _paths import PROCESSED_DIR, REFERENCE_DIR, list_cohort_dirs

# Columns that hold a human-readable class, in priority order. Different
# cohorts use different vocabularies here; see module docstring.
CLASS_COLUMNS = ["Putative.Chemical.Class", "Subpathway", "Superpathway"]
NULLISH = {"", "NA", "N/A", "nan", "None", "-"}


def _clean(value) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    return "" if value in NULLISH else value


def _compound_to_name(compound: str) -> str:
    """Platform compound IDs often embed the name after a colon; otherwise use as-is."""
    compound = _clean(compound)
    if ": " in compound:
        return compound.split(": ", 1)[1].strip()
    return compound


def _consensus(values: list[str], prefer_short: bool = False) -> str:
    counts = Counter(v for v in map(_clean, values) if v)
    if not counts:
        return ""
    top = max(counts.values())
    winners = [v for v, c in counts.items() if c == top]
    return sorted(winners, key=len)[0] if prefer_short else sorted(winners)[0]


def load_cohort_map(cohort_dir) -> pd.DataFrame | None:
    map_path = cohort_dir / "mtb.map.tsv"
    if not map_path.exists():
        return None
    cmap = pd.read_csv(map_path, sep="\t", dtype=str)
    if "HMDB" not in cmap.columns:
        return None
    cmap = cmap.dropna(subset=["HMDB"])

    name_source = cmap["Compound.Name"] if "Compound.Name" in cmap.columns else cmap.get("Compound", "")
    class_col = next((c for c in CLASS_COLUMNS if c in cmap.columns), None)

    out = pd.DataFrame({
        "hmdb_id": cmap["HMDB"].map(_clean),
        "kegg_id": cmap["KEGG"].map(_clean) if "KEGG" in cmap.columns else "",
        "name": [
            _clean(n) or _compound_to_name(c)
            for n, c in zip(name_source, cmap.get("Compound", pd.Series([""] * len(cmap))))
        ],
        "class_hint": cmap[class_col].map(_clean) if class_col else "",
        "high_confidence": (
            cmap["High.Confidence.Annotation"].astype(str).str.upper().eq("TRUE")
            if "High.Confidence.Annotation" in cmap.columns else False
        ),
        "cohort": cohort_dir.name,
    })
    return out[out["hmdb_id"] != ""]


def main() -> None:
    frames = []
    for cohort_dir in list_cohort_dirs():
        cmap = load_cohort_map(cohort_dir)
        if cmap is None:
            print(f"  ! {cohort_dir.name}: no usable mtb.map.tsv, skipping")
            continue
        frames.append(cmap)
        named = (cmap["name"] != "").sum()
        print(f"  {cohort_dir.name:36s} {len(cmap):5d} annotated  ({named} named, "
              f"{(cmap['kegg_id'] != '').sum()} with KEGG)")

    if not frames:
        print("No annotation files found -- run fetch_borenstein.py first.")
        return

    combined = pd.concat(frames, ignore_index=True)

    records = []
    for hmdb, grp in combined.groupby("hmdb_id"):
        records.append({
            "hmdb_id": hmdb,
            "name": _consensus(grp["name"].tolist(), prefer_short=True),
            "kegg_id": _consensus(grp["kegg_id"].tolist()),
            "class_hint": _consensus(grp["class_hint"].tolist()),
            "high_confidence": bool(grp["high_confidence"].any()),
            "n_cohorts_annotated": grp["cohort"].nunique(),
        })

    ref = pd.DataFrame(records).sort_values("hmdb_id").reset_index(drop=True)

    # Flag which of these actually survived into the modeling targets -- the raw
    # maps include compounds dropped during standardization, and a reference
    # table listing metabolites the model can't predict would be misleading.
    targets_path = PROCESSED_DIR / "targets.parquet"
    if targets_path.exists():
        modeled = set(pd.read_parquet(targets_path).columns)
        ref["in_model_targets"] = ref["hmdb_id"].isin(modeled)
    else:
        ref["in_model_targets"] = pd.NA
        print("\n  ! targets.parquet not found -- run build_dataset.py to flag modeled targets")

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    out = REFERENCE_DIR / "metabolite_reference.csv"
    ref.to_csv(out, index=False)

    print(f"\n-> {out}  ({len(ref)} unique HMDB IDs)")
    print(f"   named:                 {(ref['name'] != '').sum()}")
    print(f"   with a KEGG ID:        {(ref['kegg_id'] != '').sum()}")
    print(f"   with a class hint:     {(ref['class_hint'] != '').sum()}")
    print(f"   high-confidence:       {ref['high_confidence'].sum()}")
    if targets_path.exists():
        print(f"   in model targets:      {ref['in_model_targets'].sum()} / {len(modeled)}")


if __name__ == "__main__":
    main()
