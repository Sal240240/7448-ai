"""API contract and input validation.

Validation is the service's only trust boundary — everything downstream indexes
numpy arrays on the assumption that it held. These tests exercise the boundary
with the shapes an attacker would actually send.

Skipped wholesale when model artifacts aren't built, so a fresh clone doesn't
show spurious failures before the pipeline has been run.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "webapp" / "backend"))

pytest.importorskip("fastapi")

if not (ROOT / "models" / "metabolite_predictor.npz").exists():
    pytest.skip("model artifacts not built; run scripts/train_production_model.py",
                allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# --------------------------------------------------------------- contract ---

def test_health_reports_loaded_model(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_metabolite_targets"] > 0


def test_simulate_returns_predictions_with_confidence(client):
    body = client.post("/api/simulate", json={"adjustments": {}}).json()
    assert body["metabolites"], "expected at least one predicted metabolite"
    for metabolite in body["metabolites"]:
        # The core product promise: no bare numbers.
        assert "confidence" in metabolite
        assert "test_pearson_r" in metabolite


def test_simulate_at_healthy_defaults_is_a_no_op(client):
    """Sending the healthy reference back must not register as a change.

    This caught a real bug: adjustments keyed by genus name resolved to a
    different feature column than the one the default was computed from, so the
    UI showed a large spurious delta before the user touched anything.
    """
    taxa = client.get("/api/taxa").json()
    adjustments = {t["feature"]: t["healthy_abundance"] for t in taxa}
    body = client.post("/api/simulate", json={"adjustments": adjustments}).json()
    assert all(abs(m["delta"]) < 0.01 for m in body["metabolites"])


def test_simulate_responds_to_a_real_change(client):
    taxa = client.get("/api/taxa").json()
    target = next(t for t in taxa if t["genus"] == "Faecalibacterium")
    body = client.post(
        "/api/simulate", json={"adjustments": {target["feature"]: 1e-6}}
    ).json()
    assert any(abs(m["delta"]) > 0.01 for m in body["metabolites"])


def test_drivers_are_named(client):
    body = client.post("/api/simulate", json={"adjustments": {}}).json()
    drivers = [d for m in body["metabolites"] for d in m["drivers"]]
    assert drivers
    assert all(d["taxon"].strip() for d in drivers), "a blank driver name reaches the UI"


def test_taxa_and_examples_are_listable(client):
    assert len(client.get("/api/taxa").json()) > 0
    assert len(client.get("/api/examples").json()) > 0


def test_examples_omit_bulky_composition(client):
    # Compositions are fetched by reference; shipping them in the list would
    # bloat every page load.
    assert all("composition" not in e for e in client.get("/api/examples").json())


def test_model_info_exposes_honest_metrics(client):
    body = client.get("/api/model-info").json()
    assert body["metabolite_model"]["median_test_pearson_r"] > 0
    # The count of poorly-predicted targets must be published, not hidden.
    assert body["metabolite_model"]["n_targets_r_below_0.2"] >= 0


def test_runs_endpoint_strips_host_details(client):
    for run in client.get("/api/runs").json():
        assert "host" not in run


# ------------------------------------------------------------- validation ---

def test_rejects_abundance_out_of_range(client):
    assert client.post("/api/simulate", json={"adjustments": {"Blautia": 1.5}}).status_code == 422
    assert client.post("/api/simulate", json={"adjustments": {"Blautia": -0.1}}).status_code == 422


def test_rejects_unknown_field(client):
    assert client.post("/api/simulate", json={"unexpected": 1}).status_code == 422


def test_rejects_too_many_adjustments(client):
    payload = {f"Genus{i}": 0.01 for i in range(200)}
    assert client.post("/api/simulate", json={"adjustments": payload}).status_code == 422


def test_rejects_malformed_taxon_key(client):
    for key in ["<script>", "../../etc/passwd", "a" * 400, "drop\x00table"]:
        response = client.post("/api/simulate", json={"adjustments": {key: 0.01}})
        assert response.status_code == 422, f"accepted malformed key {key!r}"


def test_rejects_non_numeric_abundance(client):
    assert client.post("/api/simulate", json={"adjustments": {"Blautia": "lots"}}).status_code == 422


def test_rejects_malformed_metabolite_id(client):
    for bad in ["DROP-TABLE", "HMDBxxxx", "../secrets", "HMDB" + "9" * 40]:
        assert client.get(f"/api/metabolite/{bad}").status_code in (400, 404)


def test_unknown_example_id_degrades_with_a_note(client):
    """An unknown example must fall back and say so, not 500."""
    body = client.post(
        "/api/simulate", json={"adjustments": {}, "example_id": "nope:nothing"}
    ).json()
    assert body["metabolites"]
    assert any("Unknown example" in note for note in body["notes"])


def test_security_headers_present(client):
    headers = client.get("/api/health").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert "content-security-policy" in headers


# ------------------------------------------- security review regressions ---
# Each of these covers a specific finding from an independent security review.

def test_chunked_body_without_content_length_is_size_limited(client):
    """Content-Length alone was checkable; Transfer-Encoding: chunked skipped it.

    httpx sends a generator body as chunked with no Content-Length, which is
    exactly the bypass: the body would otherwise be buffered into memory in
    full before any limit applied.
    """
    def oversized():
        chunk = b"x" * 64_000
        for _ in range(12):  # ~768KB against a 256KB ceiling
            yield chunk

    response = client.post("/api/simulate", content=oversized(),
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 413


def test_normal_chunked_request_still_works(client):
    """The streaming limiter must replay the body, not consume it."""
    import json as _json
    payload = _json.dumps({"adjustments": {}}).encode()

    def body():
        yield payload

    response = client.post("/api/simulate", content=body(),
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 200
    assert response.json()["metabolites"]


def test_runs_do_not_leak_absolute_filesystem_paths(client):
    """Run records embed exception text and paths that identify the host."""
    import re as _re
    body = client.get("/api/runs").json()
    blob = str(body)
    assert not _re.search(r"[A-Za-z]:[\/]Users", blob), "leaked a Windows user path"
    assert "/home/" not in blob


def test_taxon_key_rejects_trailing_newline(client):
    # '$' would have allowed this; '\Z' does not.
    response = client.post("/api/simulate", json={"adjustments": {"Roseburia\n": 0.01}})
    assert response.status_code == 422


def test_metabolite_id_rejects_non_ascii_digits(client):
    # str.isdigit() accepts these; an explicit ASCII pattern does not.
    assert client.get("/api/metabolite/HMDB\u00b2\u00b2\u00b2\u00b2").status_code == 400


def test_corp_header_allows_cross_origin_frontend(client):
    # 'same-site' silently breaks a frontend on a different registrable domain.
    assert client.get("/api/health").headers["cross-origin-resource-policy"] == "cross-origin"
