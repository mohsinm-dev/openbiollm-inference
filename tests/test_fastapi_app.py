from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

import openbiollm_inference.serve_gpu.fastapi_app as app_mod


class _FakeResponse:
    def __init__(self, json_data: dict[str, Any]) -> None:
        self._json = json_data

    def raise_for_status(self) -> None:  # no-op for success
        return None

    def json(self) -> dict[str, Any]:
        return self._json


class _FakeClient:
    def __init__(self, timeout: float | int | None = None) -> None:  # signature-compatible
        self.timeout = timeout

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def post(self, url: str, headers: dict[str, str] | None = None, json: dict[str, Any] | None = None) -> _FakeResponse:  # noqa: A002
        data = {
            "choices": [
                {"message": {"role": "assistant", "content": "Hello test"}, "index": 0}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return _FakeResponse(data)


def test_health_ok() -> None:
    client = TestClient(app_mod.app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"


def test_generate_with_mocked_httpx() -> None:
    # Patch httpx.Client in the module to avoid network calls
    app_mod.httpx.Client = _FakeClient  # type: ignore[attr-defined]
    client = TestClient(app_mod.app)
    r = client.post("/generate", json={"input": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data["text"] == "Hello test"
    assert data["prompt_tokens"] == 10
    assert data["completion_tokens"] == 5

