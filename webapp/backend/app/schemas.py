"""Request/response schemas.

Validation here is the API's only trust boundary -- everything downstream
(numpy indexing, the sparse matmul, the literature lookup) assumes it is
receiving sane values. Bounds are therefore enforced strictly and explicitly
rather than left to coerce.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

# A taxon key is either a genus name or a full GTDB lineage. Both are drawn from
# a fixed server-side vocabulary, but the pattern rejects obvious junk before it
# reaches a dictionary lookup.
# Anchored with \Z rather than $: in Python, $ also matches immediately before
# a trailing newline, so "Roseburia\n" would pass a validator documented as
# rejecting anything outside the character class.
TAXON_KEY = re.compile(r"^[A-Za-z0-9_;\-\.\[\] ]{2,300}\Z")
EXAMPLE_ID = re.compile(r"^[A-Za-z0-9_\-:\.]{1,120}\Z")

MAX_ADJUSTMENTS = 64


class SimulationRequest(BaseModel):
    model_config = {"extra": "forbid"}

    adjustments: dict[str, float] = Field(
        default_factory=dict,
        description="Genus name -> relative abundance (0-1). Applied on top of the base profile.",
    )
    example_id: str | None = Field(
        default=None,
        description="Optional real cohort sample to use as the base profile instead of the healthy reference.",
    )
    featured_only: bool = Field(
        default=True,
        description="Return only the curated, explainable metabolite set rather than all 1,098.",
    )

    @field_validator("adjustments")
    @classmethod
    def validate_adjustments(cls, value: dict[str, float]) -> dict[str, float]:
        if len(value) > MAX_ADJUSTMENTS:
            raise ValueError(f"at most {MAX_ADJUSTMENTS} adjustments per request")
        for key, abundance in value.items():
            if not TAXON_KEY.match(key):
                raise ValueError(f"invalid taxon key: {key[:40]!r}")
            # Relative abundance is a proportion of the community. Values outside
            # [0, 1] aren't compositions, and the arcsine-sqrt transform would
            # silently clamp them -- better to reject than to quietly reinterpret.
            if not (0.0 <= abundance <= 1.0):
                raise ValueError(f"abundance for {key[:40]!r} must be between 0 and 1")
        return value

    @field_validator("example_id")
    @classmethod
    def validate_example_id(cls, value: str | None) -> str | None:
        if value is not None and not EXAMPLE_ID.match(value):
            raise ValueError("invalid example_id")
        return value


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_metabolite_targets: int
    n_taxa_features: int
