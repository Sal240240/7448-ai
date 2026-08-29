"""
Phase 0 step 1: pull the Borenstein Lab curated microbiome-metabolome collection.

Source: github.com/borenstein-lab/microbiome-metabolome-curated-data
The repo is ~1.3GB including .RData files we don't need, so this pulls only
the per-cohort .tsv files (taxonomy, metabolites, metabolite-HMDB map, metadata)
directly from raw.githubusercontent.com instead of a full git clone.

Usage:
    python scripts/fetch_borenstein.py                # all cohorts
    python scripts/fetch_borenstein.py --cohorts FRANZOSA_IBD_2019 iHMP_IBDMDB_2019
    python scripts/fetch_borenstein.py --include-genera
"""
import argparse
import sys
import time
from pathlib import Path

import requests

REPO = "borenstein-lab/microbiome-metabolome-curated-data"
BRANCH = "main"
API_ROOT = f"https://api.github.com/repos/{REPO}/contents"
RAW_ROOT = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
DATA_SUBDIR = "data/processed_data"

# genera.tsv exists for every cohort; species.tsv only exists for ~6/14 (the
# metagenomic cohorts -- the 16S ones only report genus level), so genera is
# the common denominator for cross-cohort feature alignment. species.tsv is
# fetched opportunistically where available for finer-grained modeling.
CORE_FILES = ["metadata.tsv", "genera.tsv", "mtb.tsv", "mtb.map.tsv"]
OPTIONAL_FILES = ["species.tsv"]

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "borenstein"


def list_cohorts(session: requests.Session) -> list[str]:
    resp = session.get(f"{API_ROOT}/{DATA_SUBDIR}", timeout=30)
    resp.raise_for_status()
    entries = resp.json()
    return sorted(e["name"] for e in entries if e["type"] == "dir")


def list_cohort_files(session: requests.Session, cohort: str) -> set[str]:
    resp = session.get(f"{API_ROOT}/{DATA_SUBDIR}/{cohort}", timeout=30)
    resp.raise_for_status()
    return {e["name"] for e in resp.json() if e["type"] == "file"}


def download_file(session: requests.Session, cohort: str, filename: str, dest: Path, retries: int = 3) -> None:
    url = f"{RAW_ROOT}/{DATA_SUBDIR}/{cohort}/{filename}"
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            return
        except requests.RequestException as e:
            if attempt == retries:
                raise
            print(f"    retry {attempt}/{retries} for {filename} ({e})", file=sys.stderr)
            time.sleep(2 * attempt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohorts", nargs="*", default=None, help="Subset of cohort names to fetch (default: all)")
    parser.add_argument("--skip-species", action="store_true", help="Don't fetch species.tsv even where available (genera.tsv is always fetched)")
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": "7448-ai-data-fetch"})

    available = list_cohorts(session)
    cohorts = args.cohorts if args.cohorts else available
    unknown = set(cohorts) - set(available)
    if unknown:
        print(f"Unknown cohort(s): {unknown}\nAvailable: {available}", file=sys.stderr)
        sys.exit(1)

    files_wanted = list(CORE_FILES) + ([] if args.skip_species else OPTIONAL_FILES)

    print(f"Fetching {len(cohorts)} cohort(s) into {OUT_DIR}")
    for cohort in cohorts:
        print(f"[{cohort}]")
        present = list_cohort_files(session, cohort)
        for fname in files_wanted:
            candidates = [fname, fname + ".zip"]
            actual = next((c for c in candidates if c in present), None)
            if actual is None:
                optional_note = " (optional, not present for this cohort)" if fname in OPTIONAL_FILES else ""
                print(f"  ! {fname} not found for {cohort}{optional_note}, skipping")
                continue
            dest = OUT_DIR / cohort / actual
            if dest.exists() and not args.force:
                print(f"  = {actual} already present")
                continue
            print(f"  > downloading {actual}")
            download_file(session, cohort, actual, dest)

    print("Done.")


if __name__ == "__main__":
    main()
