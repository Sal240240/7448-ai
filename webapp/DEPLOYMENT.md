# Running and deploying the webapp

The app is two services: a stateless FastAPI backend serving a fixed trained
model, and a static React frontend. Neither holds state, so deployment is
mostly a question of where to put two containers.

## Prerequisites

The backend serves artifacts that the pipeline produces. Generate them first
(from the repository root):

```
python scripts/fetch_borenstein.py
python scripts/standardize_taxonomy.py
python scripts/standardize_metabolites.py
python scripts/build_dataset.py
python scripts/make_splits.py
python scripts/train_elastic_net.py --min-train-measured 100
python scripts/train_production_model.py
python scripts/build_metabolite_reference.py
python scripts/fetch_kegg.py
python scripts/fetch_disbiome.py
python scripts/build_outcome_labels.py
python scripts/train_outcome_model.py
python scripts/build_trajectories.py
python scripts/build_app_data.py
```

`python scripts/status.py` prints what exists and what is still missing.

Required at runtime: `models/`, `webapp/data/`, `data/reference/`.
Optional: `data/processed/` (enables the Outlook page) and `experiments/`
(enables the run log on the Inside-the-model page). Both degrade to an honest
"not available" rather than failing.

## Local development

Two terminals, from the repository root:

```
python -m uvicorn app.main:app --reload --port 8000 --app-dir webapp/backend
```

```
cd webapp/frontend && npm install && npm run dev
```

Open http://localhost:5173. Vite proxies `/api` to port 8000, so development is
same-origin and CORS is not involved.

## Local production-shaped run

```
docker compose -f webapp/docker-compose.yml up --build
```

Open http://localhost:8080. nginx serves the built bundle and proxies `/api` to
the backend container.

## Deploying

Any host that runs two containers works — a single VM, Fly.io, Render, Cloud
Run, ECS. There is no database, no session store, no persistent volume, and no
secret to manage, so there is nothing to provision beyond the containers
themselves.

```
docker build -f webapp/backend/Dockerfile -t 7448-api .          # from repo root
docker build -f Dockerfile -t 7448-web webapp/frontend
```

### Configuration

Backend, all via environment variables:

| Variable | Default | Notes |
|---|---|---|
| `ENVIRONMENT` | `development` | `production` disables `/docs` and enables HSTS |
| `CORS_ORIGINS` | localhost dev ports | Comma-separated. Leave empty when the frontend is proxied same-origin. Never set to `*`. |
| `TRUST_PROXY_HEADERS` | off | Enable **only** behind a reverse proxy you control. It makes the rate limiter read `X-Forwarded-For`, which is caller-controlled and therefore forgeable without a proxy in front. |
| `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_S` | 120 / 60 | Per client address, per process. |
| `MAX_REQUEST_BYTES` | 262144 | Simulation payloads are a few hundred bytes. |
| `MODELS_DIR`, `APP_DATA_DIR`, `REFERENCE_DIR`, `EXPERIMENTS_DIR` | repo-relative | Set inside the image. |

Frontend: `VITE_API_BASE` is **baked in at build time** (Vite inlines env vars
into the bundle), so it is a `--build-arg`, not a runtime variable. Leave it
empty when nginx proxies `/api` on the same origin — which is preferable, since
it avoids cross-origin requests entirely.

### Before exposing it publicly

- **Terminate TLS.** The backend sets HSTS in production, which only means
  anything over HTTPS.
- **Set `CORS_ORIGINS` to the real origin**, or serve same-origin and leave it
  empty.
- **The rate limiter is per-process and in-memory.** Behind multiple replicas
  the effective limit multiplies by the replica count. For a read-only public
  endpoint that is usually acceptable; if not, put a limiter in the proxy or
  CDN layer, which is where this belongs anyway.
- **Nothing is logged about request bodies**, and that is deliberate — the
  service discusses health-adjacent data and stores none of it. Keep it that
  way if you add logging.

## What the containers deliberately do not contain

The backend image ships the trained model and reference tables, not the ~480MB
of raw cohort data or the training scripts. It serves a fixed snapshot and has
no reason to be able to retrain itself. Retraining is a deliberate, reviewable
step someone runs against the repository.
