from __future__ import annotations

from pydantic import BaseModel, Field


class Correction(BaseModel):
    match_id: str
    dH: float = Field(ge=-0.15, le=0.15)   # hard cap per spec §10
    dD: float = Field(ge=-0.15, le=0.15)
    dA: float = Field(ge=-0.15, le=0.15)
    rationale: str = Field(default="", max_length=200)


class CorrectionBatch(BaseModel):
    corrections: list[Correction]
