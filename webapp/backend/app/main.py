"""7448 AI API -- serves metabolite predictions, outcome associations, and the
model-transparency data behind them.

Deliberate properties of this service:

  - **Stateless.** Nothing a visitor enters is written anywhere. There is no
    database, no session, no log of submitted compositions. For a tool that
    discusses health-adjacent data, not storing it is the strongest privacy
    guarantee available, and it costs nothing here because every response is a
    pure function of the request and a fixed model snapshot.

  - **Fixed model.** The served artifacts are a specific trained snapshot. The
    API does not learn from traffic, which is what makes the transparency view
    meaningful -- what a visitor inspects is what produced their answer.

  - **Honest by construction.** Every prediction ships with its held-out
    accuracy, and /api/model-info is generated from the actual experiment
    outputs, so the app cannot claim performance the backtests don't support.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .schemas import HealthResponse, SimulationRequest
from .security import RateLimitMiddleware, RequestSizeLimitMiddleware, SecurityHeadersMiddleware

settings = get_settings()
sys.path.insert(0, str(settings.models_dir.parent / "src"))

import simulate as engine  # noqa: E402
from runlog import read_runs  # noqa: E402

app = FastAPI(
    title="7448 AI API",
    description="Predicts gut metabolite profiles from microbiome composition, with sourced outcome associations.",
    version="0.2.0",
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None,
)

app.add_middleware(SecurityHeadersMiddleware, is_production=settings.is_production)
app.add_middleware(RateLimitMiddleware,
                   max_requests=settings.rate_limit_requests,
                   window_s=settings.rate_limit_window_s,
                   trust_proxy=settings.trust_proxy_headers)
app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_bytes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,  # no cookies or auth exist; sending them would be meaningless
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    max_age=600,
)


def _read_app_json(name: str):
    path = settings.app_data_dir / name
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"{name} not built -- run scripts/build_app_data.py")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        predictor = engine.load_predictor()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            n_metabolite_targets=predictor.n_targets,
            n_taxa_features=len(predictor.feature_cols),
        )
    except (FileNotFoundError, ValueError):
        return HealthResponse(status="degraded", model_loaded=False,
                              n_metabolite_targets=0, n_taxa_features=0)


@app.post("/api/simulate")
def run_simulation(request: SimulationRequest) -> dict:
    """Predict a metabolite profile and its literature-backed outcome associations.

    Returns absolute predicted values and the delta against the base profile.
    The delta is the more trustworthy number -- systematic model error largely
    cancels in a difference, while an absolute value carries it in full.
    """
    try:
        result = engine.simulate(
            adjustments=request.adjustments,
            example_id=request.example_id,
            featured_only=request.featured_only,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model artifacts missing: {e}") from e
    return result.to_dict()


@app.post("/api/outlook")
def run_outlook(request: SimulationRequest) -> dict:
    """What was actually measured later in real people whose profile started here.

    Not a forecast: no model of change is fitted. It reports the observed spread
    in matched subjects, with the sample size attached, because with 331
    longitudinal subjects a projected curve would imply precision the data
    cannot support.
    """
    try:
        return engine.outlook_for(
            adjustments=request.adjustments,
            example_id=request.example_id,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Longitudinal data missing: {e}") from e


@app.get("/api/taxa")
def list_taxa() -> list[dict]:
    """The adjustable taxa, with healthy-reference abundance as each one's default."""
    return engine.taxon_options()


@app.get("/api/examples")
def list_examples() -> list[dict]:
    """Real, already-published cohort samples a visitor can load as a starting point.

    Composition vectors are omitted here -- they're large, and the client only
    needs them by reference (POST /api/simulate with example_id).
    """
    return [
        {"id": e["id"], "cohort": e["cohort"], "condition": e["condition"], "role": e["role"]}
        for e in engine.load_examples().values()
    ]


@app.get("/api/model-info")
def model_info() -> dict:
    """Performance and provenance, generated from the actual experiment outputs."""
    summary = _read_app_json("model_summary.json")
    return summary


@app.get("/api/population-map")
def population_map() -> dict:
    """2D projection of all 2,900 real samples, for the population view.

    The PCA components are withheld: the client only plots points and receives
    its own projected position from /api/simulate, so shipping the loadings
    would be dead weight on every page load.
    """
    payload = _read_app_json("population_map.json")
    return {"points": payload["points"], "explained_variance": payload["explained_variance"]}


@app.get("/api/metabolite/{hmdb_id}")
def metabolite_detail(hmdb_id: str) -> dict:
    """Glossary entry and model accuracy for one metabolite."""
    # str.isdigit() accepts non-ASCII digit code points; be explicit.
    if not re.fullmatch(r"HMDB[0-9]{1,15}", hmdb_id):
        raise HTTPException(status_code=400, detail="Invalid HMDB identifier")
    glossary = engine.load_glossary().get(hmdb_id)
    names = engine.load_metabolite_names()
    if glossary is None and hmdb_id not in names:
        raise HTTPException(status_code=404, detail="Unknown metabolite")
    predictor = engine.load_predictor()
    idx = predictor.target_index.get(hmdb_id)
    accuracy = predictor.targets[idx] if idx is not None else None
    return {
        "hmdb_id": hmdb_id,
        "name": (glossary or {}).get("display_name") or names.get(hmdb_id, hmdb_id),
        "glossary": glossary,
        "accuracy": accuracy,
    }


# Windows drive paths, or POSIX absolute paths not preceded by ':' (so a URL's
# "https://..." isn't mistaken for one and mangled in the output).
ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|(?<!:)(?<![\w/])/)[^\s'\"]{3,}")


def _redact_paths(value):
    """Strip absolute filesystem paths out of anything headed for a response.

    Run records capture exception text and tracebacks, and a FileNotFoundError
    stringifies with the full attempted path -- which on a developer machine
    embeds the OS username. Removing the `host` key alone gives false comfort:
    `error`, `notes` and unrelativized `artifacts` are a richer identifying
    channel than the hostname that was deliberately dropped.
    """
    if isinstance(value, str):
        return ABSOLUTE_PATH.sub("<path>", value)
    if isinstance(value, list):
        return [_redact_paths(v) for v in value]
    if isinstance(value, dict):
        return {k: _redact_paths(v) for k, v in value.items()}
    return value


@app.get("/api/runs")
def runs(limit: int = 25) -> list[dict]:
    """The pipeline's own run history -- what was executed, on what, and what came out.

    Exposed because "show me what the model is actually doing" should be
    answerable from the product, not only from a terminal on the machine that
    trained it. Host details are dropped and absolute paths redacted.
    """
    limit = max(1, min(limit, 100))
    return [
        _redact_paths({k: v for k, v in run.items() if k != "host"})
        for run in read_runs(limit=limit)
    ]
