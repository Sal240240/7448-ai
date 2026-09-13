# Backend image.
#
# Build context is the REPOSITORY ROOT, not this directory, because the service
# imports the shared inference modules from src/ and loads the trained model
# artifacts from models/ and webapp/data/. Build with:
#     docker build -f webapp/backend/Dockerfile -t 7448-api .
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so a code change doesn't invalidate the layer.
COPY webapp/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Only what the service actually needs at runtime. The raw cohort data
# (data/raw, ~480MB) and the training scripts are deliberately absent: this
# container serves a fixed model snapshot and has no reason to carry the
# material used to produce it.
COPY src/ ./src/
COPY models/ ./models/
COPY webapp/data/ ./webapp/data/
COPY data/reference/ ./data/reference/
COPY webapp/backend/app/ ./app/

ENV MODELS_DIR=/app/models \
    APP_DATA_DIR=/app/webapp/data \
    REFERENCE_DIR=/app/data/reference \
    EXPERIMENTS_DIR=/app/experiments \
    ENVIRONMENT=production

# The trajectory ("Outlook") feature reads the full processed feature table.
# It is optional: the endpoint reports its absence honestly rather than
# failing, so this stays out of the default image to keep it small. Mount
# data/processed/ as a volume, or uncomment, to enable it.
# COPY data/processed/ ./data/processed/

# Run unprivileged. Nothing here needs to write to disk at all.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
