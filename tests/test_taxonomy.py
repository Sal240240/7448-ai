"""Lineage parsing — the layer everything else keys on.

A silent failure here doesn't crash anything, it just quietly mismatches a
genus against the literature database and produces a wrong-but-plausible
association, which is the worst failure mode this project has.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from taxonomy import deepest_rank, display_name, friendly_name, genus_name, match_key, parse_lineage

FULL = "d__Bacteria;p__Firmicutes_A;c__Clostridia;o__Lachnospirales;f__Lachnospiraceae;g__Blautia_A"
NO_GENUS = "d__Bacteria;p__Firmicutes_A;c__Clostridia;o__Lachnospirales;f__Lachnospiraceae;g__"
SPARSE = "d__Bacteria;p__Firmicutes_A;c__Clostridia;o__;f__;g__"


def test_parse_lineage_drops_empty_ranks():
    parsed = parse_lineage(NO_GENUS)
    assert parsed["f"] == "Lachnospiraceae"
    assert "g" not in parsed


def test_genus_name_keeps_gtdb_suffix():
    assert genus_name(FULL) == "Blautia_A"


def test_match_key_strips_suffix_for_literature_join():
    # Disbiome records "Blautia", never "Blautia_A" -- without stripping, every
    # GTDB-split genus would silently miss its associations.
    assert match_key(FULL) == "Blautia"


def test_match_key_empty_when_unresolved_to_genus():
    # Must be empty, not the family name: a family has no genus-keyed entry in
    # the literature database, and substituting one would fabricate a match.
    assert match_key(NO_GENUS) == ""
    assert match_key(SPARSE) == ""


def test_display_name_preserves_precision():
    assert display_name(FULL) == "Blautia_A"


def test_friendly_name_drops_suffix_for_readers():
    assert friendly_name(FULL) == "Blautia"


def test_unresolved_lineage_falls_back_to_deepest_named_rank():
    assert display_name(NO_GENUS) == "Lachnospiraceae (family)"
    assert friendly_name(NO_GENUS) == "Lachnospiraceae (family)"
    # SPARSE names nothing below class, so class is the deepest available.
    assert display_name(SPARSE) == "Clostridia (class)"


def test_deepest_rank_prefers_finest_resolution():
    assert deepest_rank(FULL) == ("g", "Blautia_A")
    assert deepest_rank(NO_GENUS) == ("f", "Lachnospiraceae")


@pytest.mark.parametrize("value", ["", "nonsense", "d__", ";;;"])
def test_malformed_input_does_not_raise(value):
    # These reach the parser from cohort files we don't control.
    assert isinstance(display_name(value), str)
    assert isinstance(match_key(value), str)


def test_strip_gtdb_suffix_handles_bare_genus_names():
    from taxonomy import strip_gtdb_suffix
    assert strip_gtdb_suffix("Blautia_A") == "Blautia"
    assert strip_gtdb_suffix("Blautia") == "Blautia"
    # match_key is for lineages and correctly returns "" here — the reason this
    # separate helper has to exist.
    assert match_key("Blautia_A") == ""
