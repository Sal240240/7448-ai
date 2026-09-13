"""Outcome association layer.

The weighting logic decides what a reader is shown first, so a bug here changes
which health associations get prominence — the highest-stakes ranking in the
project.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from outcomes import MAX_ABS_Z, MIN_REFERENCE_SD, profile_deviations

FEATURES = [
    "d__Bacteria;p__P;c__C;o__O;f__F;g__Alpha",
    "d__Bacteria;p__P;c__C;o__O;f__F;g__Beta_A",
    "d__Bacteria;p__P;c__C;o__O;f__F;g__",
]


def test_constant_reference_feature_cannot_produce_a_deviation():
    """Regression: a zero-variance taxon got a 1e-6 epsilon SD upstream.

    That cleared a naive 1e-9 guard and turned any nonzero abundance into a
    z-score in the thousands, which would dominate the ranking and fabricate a
    disease signal from a taxon that never varies.
    """
    x = np.array([0.5, 0.0, 0.0])
    mean = np.array([0.0, 0.0, 0.0])
    sd = np.array([1e-6, 1.0, 1.0])  # the epsilon substituted for a zero SD
    deviations = profile_deviations(x, FEATURES, mean, sd)
    assert deviations.get("Alpha", 0.0) == 0.0


def test_deviations_are_clipped_to_a_plausible_range():
    x = np.array([100.0, 0.0, 0.0])
    mean = np.zeros(3)
    sd = np.array([MIN_REFERENCE_SD * 2, 1.0, 1.0])
    deviations = profile_deviations(x, FEATURES, mean, sd)
    assert all(abs(v) <= MAX_ABS_Z for v in deviations.values())


def test_deviations_key_on_suffix_stripped_genus():
    x = np.array([0.0, 2.0, 0.0])
    mean = np.zeros(3)
    sd = np.ones(3)
    deviations = profile_deviations(x, FEATURES, mean, sd)
    # Beta_A must join the literature layer as "Beta".
    assert "Beta" in deviations
    assert deviations["Beta"] == pytest.approx(2.0)


def test_unresolved_lineage_never_becomes_a_deviation():
    x = np.array([0.0, 0.0, 5.0])
    deviations = profile_deviations(x, FEATURES, np.zeros(3), np.ones(3))
    assert len(deviations) == 0, "a family-level feature was given a genus key"
