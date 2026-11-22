"""Request-id handling in the ASGI middleware."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ragsys.api.middleware import RequestContextMiddleware


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"ok": "yes"}

    return TestClient(app)


def test_valid_incoming_id_is_echoed(client: TestClient) -> None:
    resp = client.get("/ping", headers={"X-Request-ID": "trace-42"})
    assert resp.headers["x-request-id"] == "trace-42"


@pytest.mark.parametrize("bad", ["x" * 129, "   "])
def test_unusable_incoming_id_is_replaced(client: TestClient, bad: str) -> None:
    resp = client.get("/ping", headers={"X-Request-ID": bad})
    assert resp.status_code == 200
    new_id = resp.headers["x-request-id"]
    assert new_id != bad and len(new_id) == 32


def test_each_request_without_an_id_gets_a_distinct_one(client: TestClient) -> None:
    first = client.get("/ping").headers["x-request-id"]
    second = client.get("/ping").headers["x-request-id"]
    assert first != second
