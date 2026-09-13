"""Environment-driven configuration.

Everything deployment-specific is read from the environment with a
local-development default, so the same image runs locally and hosted without
code changes. Nothing here is a secret -- this service has no database, no
credentials and no user accounts, which is a deliberate design property rather
than an unfinished feature: a tool that discusses health data is far easier to
operate safely when it never stores any.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _env_list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


class Settings:
    def __init__(self) -> None:
        self.app_name = "7448 AI"
        self.environment = os.getenv("ENVIRONMENT", "development")

        # Vite's dev server ports. In production this MUST be set to the real
        # origin -- a wildcard here would let any site drive this API from a
        # visitor's browser.
        self.cors_origins = _env_list("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")

        self.models_dir = Path(os.getenv("MODELS_DIR", REPO_ROOT / "models"))
        self.app_data_dir = Path(os.getenv("APP_DATA_DIR", REPO_ROOT / "webapp" / "data"))
        self.reference_dir = Path(os.getenv("REFERENCE_DIR", REPO_ROOT / "data" / "reference"))
        self.experiments_dir = Path(os.getenv("EXPERIMENTS_DIR", REPO_ROOT / "experiments"))

        # Rate limiting: generous for a human clicking around, low enough that a
        # single client can't monopolise the process. Per-IP, in-process.
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "120"))
        self.rate_limit_window_s = int(os.getenv("RATE_LIMIT_WINDOW_S", "60"))

        # Request body ceiling. Simulation payloads are a few hundred bytes; this
        # is three orders of magnitude of headroom and still refuses anything
        # designed to exhaust memory.
        self.max_request_bytes = int(os.getenv("MAX_REQUEST_BYTES", str(256 * 1024)))

        # Enable ONLY when a reverse proxy you control sets X-Forwarded-For.
        # Without a proxy in front, this header is attacker-controlled and
        # enabling it makes the rate limiter trivially bypassable.
        self.trust_proxy_headers = os.getenv("TRUST_PROXY_HEADERS", "").lower() in {"1", "true", "yes"}

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
