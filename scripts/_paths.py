"""Shared path helpers for the Phase 0 data scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
BORENSTEIN_DIR = RAW_DIR / "borenstein"

# Third-party pulls (Disbiome, KEGG, PubMed) land here; gitignored like raw/.
EXTERNAL_DIR = ROOT / "data" / "external"
# Small, curated, derived tables that ARE checked in -- they're what the webapp
# and the outcome layer read at runtime, and they're small enough to review in
# a diff, unlike the multi-hundred-MB cohort data.
REFERENCE_DIR = ROOT / "data" / "reference"


def list_cohort_dirs():
    if not BORENSTEIN_DIR.exists():
        return []
    return sorted(p for p in BORENSTEIN_DIR.iterdir() if p.is_dir())
