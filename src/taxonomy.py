"""Parsing and display of GTDB lineage strings.

Feature columns look like:
    d__Bacteria;p__Firmicutes_A;c__Clostridia;o__Lachnospirales;f__Lachnospiraceae;g__Roseburia

Two details bite everywhere this is parsed, which is why it lives in one place
rather than being re-implemented per script:

  - **Not every lineage reaches genus.** 16S profiling often resolves only to
    family, leaving a bare `g__`. Those features are real and the model uses
    them, so they need an honest display label ("Lachnospiraceae (family)")
    rather than a blank or a silent drop.

  - **GTDB suffixes.** GTDB splits some genera that older taxonomies treat as
    one, marking them `Blautia_A`, `Firmicutes_A`. Literature databases like
    Disbiome use the unsuffixed name, so matching against them requires
    stripping it -- but display should keep it, because `Blautia_A` and
    `Blautia_B` are genuinely different groups.
"""
from __future__ import annotations

import re

GTDB_SUFFIX = re.compile(r"_[A-Z]+$")
RANK_LABELS = {"g": "genus", "f": "family", "o": "order", "c": "class", "p": "phylum", "d": "domain"}
RANK_ORDER = ("g", "f", "o", "c", "p", "d")


def parse_lineage(lineage: str) -> dict[str, str]:
    """Split a lineage into {rank_prefix: name}, dropping empty ranks."""
    out = {}
    for part in str(lineage).split(";"):
        part = part.strip()
        if "__" not in part:
            continue
        prefix, _, name = part.partition("__")
        if name:
            out[prefix] = name
    return out


def deepest_rank(lineage: str) -> tuple[str, str]:
    """Deepest resolved (rank_prefix, name); ("", lineage) if nothing parses."""
    ranks = parse_lineage(lineage)
    for prefix in RANK_ORDER:
        if prefix in ranks:
            return prefix, ranks[prefix]
    return "", str(lineage)


def genus_name(lineage: str) -> str:
    """Genus as written in GTDB, suffix intact. Empty when unresolved to genus."""
    return parse_lineage(lineage).get("g", "")


def strip_gtdb_suffix(name: str) -> str:
    """Drop GTDB's alphabetic split suffix from an already-bare genus name.

    Distinct from match_key(), which parses a full lineage. Calling match_key on
    a bare "Blautia_A" returns "" (there is no "__" to parse), so callers
    holding a plain genus string need this instead.
    """
    return GTDB_SUFFIX.sub("", str(name).strip())


def match_key(lineage: str) -> str:
    """Genus normalized for joining against literature databases (suffix stripped).

    Empty for lineages that don't reach genus -- those legitimately have no
    counterpart in a genus-keyed database, and inventing one would create
    associations that don't exist.
    """
    genus = genus_name(lineage)
    return GTDB_SUFFIX.sub("", genus) if genus else ""


def display_name(lineage: str) -> str:
    """Precise label: genus exactly as GTDB writes it, suffix included.

    For anywhere the distinction between Blautia_A and Blautia_B matters.
    """
    genus = genus_name(lineage)
    if genus:
        return genus
    prefix, name = deepest_rank(lineage)
    if not prefix:
        return str(lineage)
    return f"{name} ({RANK_LABELS.get(prefix, prefix)})"


def friendly_name(lineage: str) -> str:
    """Reader-facing label, with the GTDB suffix dropped.

    "Faecalibacterium_A" is a database bookkeeping detail: GTDB split a genus
    that the wider literature — and every reader outside microbial taxonomy —
    still calls Faecalibacterium. Showing the suffix in a lay interface adds
    confusion without adding anything the reader can act on, so the precise
    spelling stays available via display_name() and the full lineage, and this
    is what gets rendered.
    """
    genus = genus_name(lineage)
    if genus:
        return GTDB_SUFFIX.sub("", genus)
    return display_name(lineage)
