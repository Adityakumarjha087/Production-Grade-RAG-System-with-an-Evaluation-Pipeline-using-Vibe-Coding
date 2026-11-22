"""Input normalisation and limits of the query schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragsys.api.schemas import QueryRequest


def test_query_is_stripped() -> None:
    assert QueryRequest(query="  what is T+1?  ").query == "what is T+1?"


def test_query_length_limits() -> None:
    assert QueryRequest(query="x" * 2000)
    with pytest.raises(ValidationError):
        QueryRequest(query="x" * 2001)
    with pytest.raises(ValidationError):
        QueryRequest(query=" a ")  # under 3 characters once the padding is removed
