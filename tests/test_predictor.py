"""Inference and attribution.

The property that matters most: attribution must be exact. The webapp tells
readers "these microbes produced this number", and that claim is only true if
the contributions actually sum to the prediction.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from predictor import MetabolitePredictor, arcsine_sqrt, confidence_band

FEATURES = [
    "d__Bacteria;p__P;c__C;o__O;f__F;g__Alpha",
    "d__Bacteria;p__P;c__C;o__O;f__F;g__Beta_A",
    "d__Bacteria;p__P;c__C;o__O;f__F;g__",
]


@pytest.fixture
def predictor() -> MetabolitePredictor:
    coef = sparse.csr_matrix(np.array([[2.0, -1.0, 0.5], [0.0, 3.0, 0.0]], dtype=np.float32))
    intercepts = np.array([1.0, -0.5], dtype=np.float32)
    targets = [
        {"hmdb_id": "HMDB0000001", "test_pearson_r": 0.72},
        {"hmdb_id": "HMDB0000002", "test_pearson_r": None},
    ]
    return MetabolitePredictor(coef, intercepts, FEATURES, targets)


def test_prediction_matches_manual_linear_algebra(predictor):
    x = np.array([0.5, 0.25, 2.0])
    expected_first = 2.0 * 0.5 + (-1.0) * 0.25 + 0.5 * 2.0 + 1.0
    assert predictor.predict(x)[0] == pytest.approx(expected_first)


def test_contributions_sum_to_prediction_minus_intercept(predictor):
    """The claim the UI makes. If this drifts, the explanation is a fiction."""
    x = np.array([0.5, 0.25, 2.0])
    contributions = predictor.explain(x, "HMDB0000001", top_n=99)
    total = sum(c.contribution for c in contributions)
    assert total == pytest.approx(predictor.predict(x)[0] - predictor.intercepts[0], abs=1e-6)


def test_explain_orders_by_absolute_magnitude(predictor):
    x = np.array([0.5, 0.25, 2.0])
    magnitudes = [abs(c.contribution) for c in predictor.explain(x, "HMDB0000001", top_n=99)]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_explain_unknown_target_returns_empty(predictor):
    assert predictor.explain(np.zeros(3), "HMDB9999999") == []


def test_genus_lookup_resolves_suffixed_lineage(predictor):
    # A caller asking for "Beta" must reach the Beta_A column.
    x = predictor.vector_from_abundances({"Beta": 0.25}, baseline=np.zeros(3))
    assert x[1] == pytest.approx(arcsine_sqrt(np.asarray(0.25)))


def test_adjustments_apply_on_top_of_baseline(predictor):
    """Unspecified taxa must keep their baseline value, not fall to zero."""
    baseline = np.array([0.3, 0.4, 0.5])
    x = predictor.vector_from_abundances({"Alpha": 0.01}, baseline=baseline)
    assert x[1] == pytest.approx(0.4)
    assert x[2] == pytest.approx(0.5)
    assert x[0] != pytest.approx(0.3)


def test_baseline_is_not_mutated_by_adjustment(predictor):
    baseline = np.array([0.3, 0.4, 0.5])
    predictor.vector_from_abundances({"Alpha": 0.01}, baseline=baseline)
    assert baseline[0] == pytest.approx(0.3)


def test_unknown_taxon_is_ignored_not_fatal(predictor):
    x = predictor.vector_from_abundances({"Nonexistent": 0.5}, baseline=np.zeros(3))
    assert np.allclose(x, np.zeros(3))


def test_arcsine_sqrt_clamps_out_of_range_input():
    assert arcsine_sqrt(np.asarray(-1.0)) == pytest.approx(0.0)
    assert arcsine_sqrt(np.asarray(5.0)) == pytest.approx(np.pi / 2)


@pytest.mark.parametrize(
    "r,expected",
    [(0.8, "good"), (0.5, "good"), (0.35, "moderate"), (0.2, "weak"), (0.05, "very_weak"),
     (None, "unvalidated"), (float("nan"), "unvalidated")],
)
def test_confidence_bands(r, expected):
    assert confidence_band(r) == expected


def test_prediction_carries_confidence_for_every_target(predictor):
    """No value may reach the UI without an accuracy label attached."""
    results = predictor.predict_with_explanations(np.array([0.5, 0.25, 2.0]))
    assert len(results) == 2
    assert all(r.confidence for r in results)
    assert results[1].confidence == "unvalidated"  # test_pearson_r was None


def test_feature_at_index_zero_is_addressable(predictor):
    """Regression: `feature_index.get(k) or genus_index.get(k)` drops index 0.

    Index 0 is falsy, so the `or` fell through to the genus lookup and the
    adjustment was silently discarded — a wrong answer with no error.
    """
    first = FEATURES[0]
    x = predictor.vector_from_abundances({first: 0.25}, baseline=np.zeros(3))
    assert x[0] == pytest.approx(arcsine_sqrt(np.asarray(0.25))), "adjustment on feature 0 was dropped"


def test_suffixed_genus_name_is_addressable(predictor):
    """Regression: a bare "Beta_A" routed through match_key() resolved to nothing.

    match_key() parses a full lineage; given a bare genus it returns "", so the
    genus fallback silently dropped every suffixed name a caller supplied.
    """
    x = predictor.vector_from_abundances({"Beta_A": 0.25}, baseline=np.zeros(3))
    assert x[1] == pytest.approx(arcsine_sqrt(np.asarray(0.25)))
