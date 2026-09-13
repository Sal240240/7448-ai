"""
Phase 2 step 3: pull Disbiome's curated microbe-disease associations.

Disbiome (disbiome.ugent.be) is a hand-curated database of published
observations of the form "organism X was reported Elevated/Reduced in disease
Y, measured by method Z, in sample type S, citation P". ~10,900 such records
across 375 diseases and 1,615 organisms. That is exactly the layer this
project is missing: the models predict metabolite abundances, but nothing in
the repo yet connects a microbial or metabolic shift to a health outcome a
person would recognize.

Every association carried forward keeps its publication reference, so the
webapp can show "3 studies reported this genus reduced in Crohn's disease"
with citations rather than an unsourced claim. Records are observational
associations from the literature -- they are not causal, and nothing
downstream should render them as such.

Two mapping problems this handles:

  - **Naming.** Our feature columns are full GTDB lineages
    ("d__Bacteria;...;g__Blautia_A"); Disbiome uses plain names ("Blautia",
    "Faecalibacterium prausnitzii"). We match at genus level, stripping GTDB's
    alphabetic suffixes (Blautia_A -> Blautia), which exist because GTDB splits
    some genera that older taxonomies (and Disbiome) treat as one.

  - **Sample type.** Disbiome covers all body sites. A genus elevated in
    saliva during periodontitis says nothing about a stool sample, so gut
    relevance is flagged per record and reported separately rather than
    silently mixed in.

Raw endpoint dumps are cached under data/external/disbiome/.

Output:
    data/reference/taxon_disease_associations.csv  (checked in; genus x disease)

Usage:
    python scripts/fetch_disbiome.py
    python scripts/fetch_disbiome.py --force
"""
import argparse
import json
import re

import pandas as pd
import requests

from _paths import EXTERNAL_DIR, PROCESSED_DIR, REFERENCE_DIR

API_ROOT = "https://disbiome.ugent.be:8080"
ENDPOINTS = ["experiment", "disease", "organism", "sample", "method", "publication"]
CACHE_DIR = EXTERNAL_DIR / "disbiome"

# Disbiome curates methodological-quality flags per publication ("y"/"n"). These
# are the difference between "3 studies reported this" and "3 studies reported
# this, and 2 of them matched controls for confounders" -- a tool that shows
# health associations without showing how well-evidenced they are is just
# laundering weak findings into confident-looking output.
QUALITY_FLAGS = [
    "controls_matched_for_possible_confounding_factors",
    "sample_size_justified",
    "specific_test_statistics_reported",
    "inclusion_exclusion_criteria_stated",
    "measure_of_variance_reported",
    "conflict_of_interest_statement_given",
]
PMID_PATTERN = re.compile(r"(\d{6,9})")

# Disbiome sample-type names that correspond to the gut lumen / stool, i.e. the
# compartment our cohorts actually sequenced.
GUT_SAMPLE_PATTERN = re.compile(
    r"faec|fec|stool|gut|intestin|colon|ileum|ileal|rect|sigmoid|caec|cecum|duoden|jejun",
    re.IGNORECASE,
)
GTDB_SUFFIX = re.compile(r"_[A-Z]+$")


def normalize_genus(name: str) -> str:
    """GTDB splits some genera with an alphabetic suffix (Blautia_A); Disbiome doesn't."""
    return GTDB_SUFFIX.sub("", str(name).strip())


def genus_from_lineage(lineage: str) -> str:
    for rank in str(lineage).split(";"):
        if rank.startswith("g__"):
            return normalize_genus(rank[3:])
    return ""


def genus_from_organism(organism_name: str) -> str:
    """Disbiome organism names are 'Genus', 'Genus species', or occasionally a strain."""
    token = str(organism_name).strip().split()
    return normalize_genus(token[0]) if token else ""


def pmid_from_url(url) -> str:
    """Disbiome stores 'https://www.ncbi.nlm.nih.gov/pubmed/25446201\\n' rather than a bare ID."""
    match = PMID_PATTERN.search(str(url or ""))
    return match.group(1) if match else ""


def clean_text(value) -> str:
    """Normalize a Disbiome string field for output.

    Titles legitimately contain typographic punctuation (U+2019 in "Crohn's
    disease"), which a Windows console renders as a replacement glyph but which
    is correct in the data and in a browser -- so this deliberately does not
    "repair" non-ASCII characters.
    """
    return str(value or "").strip()


def build_publications(publication_records: list[dict]) -> pd.DataFrame:
    """Citation + methodological-quality table, one row per Disbiome publication."""
    rows = []
    for pub in publication_records:
        flags = [str(pub.get(f, "")).strip().lower() == "y" for f in QUALITY_FLAGS]
        rows.append({
            "publication_id": pub.get("publication_id"),
            "pmid": pmid_from_url(pub.get("pubmed_url")),
            "title": clean_text(pub.get("title")),
            "first_author": clean_text(pub.get("first_author")),
            "journal": clean_text(pub.get("outlet")),
            "year": pub.get("year_of_publication"),
            "doi": (pub.get("doi") or "").strip(),
            "quality_flags_met": sum(flags),
            "quality_flags_total": len(QUALITY_FLAGS),
        })
    return pd.DataFrame(rows)


def fetch_endpoint(session: requests.Session, endpoint: str, force: bool) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{endpoint}.json"
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    resp = session.get(f"{API_ROOT}/{endpoint}", timeout=120)
    resp.raise_for_status()
    data = resp.json()
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": "7448-ai-research-pipeline"})

    data = {}
    for endpoint in ENDPOINTS:
        try:
            data[endpoint] = fetch_endpoint(session, endpoint, args.force)
            print(f"  {endpoint:12s} {len(data[endpoint]):6d} records")
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"  ! {endpoint}: {e}")
            data[endpoint] = []

    experiments = pd.DataFrame(data.get("experiment", []))
    if experiments.empty:
        print("\nNo experiment records retrieved -- nothing to build.")
        return

    publications = build_publications(data.get("publication", []))
    pub_by_id = publications.set_index("publication_id").to_dict("index") if len(publications) else {}

    experiments["genus"] = experiments["organism_name"].map(genus_from_organism)
    experiments["is_gut"] = experiments["sample_name"].fillna("").str.contains(GUT_SAMPLE_PATTERN)
    experiments["is_human"] = experiments["host_type"].fillna("").str.strip().str.lower().eq("human")
    experiments["outcome"] = experiments["qualitative_outcome"].fillna("").str.strip().str.title()

    print(f"\nRecords: {len(experiments)} total, {experiments['is_human'].sum()} human, "
          f"{(experiments['is_human'] & experiments['is_gut']).sum()} human + gut sample")

    usable = experiments[
        experiments["is_human"]
        & (experiments["genus"] != "")
        & experiments["outcome"].isin(["Elevated", "Reduced"])
    ].copy()

    rows = []
    for (genus, disease), grp in usable.groupby(["genus", "disease_name"]):
        gut = grp[grp["is_gut"]]
        pub_ids = sorted({int(p) for p in grp["publication_id"].dropna()})
        backing = [pub_by_id[p] for p in pub_ids if p in pub_by_id]
        pmids = [b["pmid"] for b in backing if b["pmid"]]
        years = [b["year"] for b in backing if b["year"]]
        n_elevated = int((grp["outcome"] == "Elevated").sum())
        n_reduced = int((grp["outcome"] == "Reduced").sum())
        rows.append({
            "genus": genus,
            "disease": disease,
            "n_reports": len(grp),
            "n_gut_reports": len(gut),
            "n_elevated": n_elevated,
            "n_reduced": n_reduced,
            # Net direction, with consistency as the share of reports agreeing --
            # a 5-0 split is a very different claim from a 3-2 split, and the
            # webapp needs to be able to say which it's showing.
            "direction": "elevated" if n_elevated > n_reduced else ("reduced" if n_reduced > n_elevated else "mixed"),
            "consistency": round(max(n_elevated, n_reduced) / max(len(grp), 1), 3),
            "n_publications": len(pub_ids),
            "pubmed_ids": ";".join(pmids[:10]),
            "latest_year": max(years) if years else "",
            # Mean share of Disbiome's methodological-quality flags met by the
            # backing studies -- surfaced so weak evidence reads as weak.
            "mean_study_quality": (
                round(sum(b["quality_flags_met"] for b in backing) / (len(backing) * len(QUALITY_FLAGS)), 3)
                if backing else ""
            ),
        })

    assoc = pd.DataFrame(rows).sort_values(["n_reports", "genus"], ascending=[False, True]).reset_index(drop=True)

    # How much of this actually lands on taxa our model can see?
    features_path = PROCESSED_DIR / "features.parquet"
    if features_path.exists():
        our_genera = {genus_from_lineage(c) for c in pd.read_parquet(features_path).columns}
        our_genera.discard("")
        assoc["in_feature_space"] = assoc["genus"].isin(our_genera)
        matched = assoc[assoc["in_feature_space"]]
        print(f"\nFeature-space coverage:")
        print(f"  distinct genera in our features:     {len(our_genera)}")
        print(f"  distinct genera in Disbiome (human): {assoc['genus'].nunique()}")
        print(f"  overlap:                             {matched['genus'].nunique()}")
        print(f"  associations on matched genera:      {len(matched)} of {len(assoc)}")
    else:
        assoc["in_feature_space"] = pd.NA
        print("\n  ! features.parquet not found -- run build_dataset.py to compute coverage")

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    out = REFERENCE_DIR / "taxon_disease_associations.csv"
    assoc.to_csv(out, index=False)

    pubs_out = REFERENCE_DIR / "publications.csv"
    publications.sort_values("publication_id").to_csv(pubs_out, index=False)
    print(f"\n-> {pubs_out}  ({len(publications)} publications, "
          f"{(publications['pmid'] != '').sum()} with a resolvable PMID)")

    print(f"-> {out}  ({len(assoc)} genus-disease associations)")
    print(f"   gut-sample-backed: {(assoc['n_gut_reports'] > 0).sum()}")
    print("\nMost-reported diseases:")
    print(assoc.groupby("disease")["n_reports"].sum().sort_values(ascending=False).head(12).to_string())


if __name__ == "__main__":
    main()
