"""
One command that answers "what does this project currently know, what has it
run, and what came out of it."

Without this, the state of the pipeline lives across a dozen parquet files, a
few CSVs, and whichever terminal a training job happened to be started in.
This reads all of it and prints a single status page: which data tables exist
and how big they are, which reference/annotation layers have been pulled, what
models are trained, and the most recent runs from the run log.

Usage:
    python scripts/status.py
    python scripts/status.py --runs 20      # show more run history
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from _paths import EXTERNAL_DIR, PROCESSED_DIR, REFERENCE_DIR, ROOT

sys.path.insert(0, str(ROOT / "src"))
from runlog import read_runs  # noqa: E402

EXPERIMENTS_DIR = ROOT / "experiments"
MODELS_DIR = ROOT / "models"


def _size(path: Path) -> str:
    if not path.exists():
        return "-"
    n = path.stat().st_size if path.is_file() else sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _parquet_shape(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        import pyarrow.parquet as pq
        meta = pq.read_metadata(path)
        return f"{meta.num_rows} x {meta.num_columns}"
    except Exception as e:  # noqa: BLE001 -- status output shouldn't crash on one bad file
        return f"unreadable ({type(e).__name__})"


def _csv_rows(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        return f"{len(pd.read_csv(path))} rows"
    except Exception as e:  # noqa: BLE001
        return f"unreadable ({type(e).__name__})"


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=10, help="How many recent runs to show")
    args = parser.parse_args()

    print("=" * 68)
    print("7448 AI -- pipeline status")
    print("=" * 68)

    section("Modeling tables (data/processed/)")
    for name in ["features", "targets", "targets_mask", "manifest", "splits",
                 "taxonomy_genus", "taxonomy_species", "metabolites"]:
        path = PROCESSED_DIR / f"{name}.parquet"
        print(f"  {name:20s} {_parquet_shape(path):>18s}  {_size(path):>8s}")

    section("Reference layers (data/reference/ -- checked in)")
    for name in ["metabolite_reference", "metabolite_kegg", "taxon_disease_associations",
                 "publications", "outcome_label_map"]:
        path = REFERENCE_DIR / f"{name}.csv"
        print(f"  {name:30s} {_csv_rows(path):>14s}  {_size(path):>8s}")

    section("External caches (data/external/ -- gitignored)")
    for sub in ["disbiome", "kegg", "pubmed"]:
        path = EXTERNAL_DIR / sub
        n_files = len(list(path.rglob("*"))) if path.exists() else 0
        print(f"  {sub:20s} {n_files:6d} files  {_size(path):>8s}")

    section("Trained models (models/)")
    if MODELS_DIR.exists():
        artifacts = sorted(MODELS_DIR.rglob("*"))
        files = [f for f in artifacts if f.is_file()]
        if files:
            for f in files:
                print(f"  {str(f.relative_to(MODELS_DIR)):40s} {_size(f):>8s}")
        else:
            print("  (none yet)")
    else:
        print("  (none yet -- models/ does not exist)")

    section("Experiment outputs (experiments/)")
    if EXPERIMENTS_DIR.exists():
        for f in sorted(EXPERIMENTS_DIR.glob("*.csv")):
            print(f"  {f.name:40s} {_csv_rows(f):>14s}")
    else:
        print("  (none yet)")

    section(f"Recent runs (last {args.runs})")
    runs = read_runs(limit=args.runs)
    if not runs:
        print("  (no runs logged yet -- experiments/run_log.jsonl is empty)")
    for run in runs:
        flag = {"ok": "  ", "failed": "!!", "interrupted": "~~"}.get(run["status"], "??")
        print(f"  {flag} {run['started_at']}  {run['step']:26s} {run['duration_s']:8.1f}s  {run['status']}")
        for key, value in list(run.get("metrics", {}).items())[:4]:
            shown = f"{value:.4g}" if isinstance(value, float) else value
            print(f"        {key}: {shown}")
        for note in run.get("notes", [])[:2]:
            print(f"        note: {note}")
        if run.get("error"):
            print(f"        error: {run['error']}")

    print()


if __name__ == "__main__":
    main()
