"""Shared path helpers for the Phase 0 data scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
BORENSTEIN_DIR = RAW_DIR / "borenstein"


def list_cohort_dirs():
    if not BORENSTEIN_DIR.exists():
        return []
    return sorted(p for p in BORENSTEIN_DIR.iterdir() if p.is_dir())
