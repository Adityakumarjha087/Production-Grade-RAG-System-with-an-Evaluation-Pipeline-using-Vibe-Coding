"""Request/response schemas for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """A user question."""

    query: str = Field(
        ..., min_length=3, max_length=2000, examples=["What is the T+1 settlement timeline?"]
    )

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if len(value.strip()) < 3:
            raise ValueError("query must contain at least 3 non-space characters")
        return value.strip()


class HealthResponse(BaseModel):
    """Liveness/readiness result."""

    status: str
    checks: dict[str, bool] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error payload (also used as the SSE ``error`` event)."""

    error: str
    message: str
    request_id: str = ""
