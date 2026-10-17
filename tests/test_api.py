import pytest
from fastapi.testclient import TestClient

import app as app_module


@pytest.fixture
def client(toy_predictor):
    app_module.app.state.model_card = {"schema": 1, "books": 12, "model": {}, "evaluation": None}
    app_module.app.state.predictor = toy_predictor
    app_module.app.state.genres = sorted(set(toy_predictor.data["genre_clean"]))
    app_module.RATE_LIMITER.hits.clear()
    # raise_server_exceptions=False so we see what a real client sees on a 500
    return TestClient(app_module.app, raise_server_exceptions=False)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready(client):
    body = client.get("/ready").json()
    assert body["status"] == "ready" and body["books"] == 12


def test_recommend_ok(client):
    r = client.get("/v1/recommend", params={"title": "Dune", "n": 3})
    assert r.status_code == 200
    assert r.json()["count"] == 3


def test_legacy_path_still_works(client):
    assert client.get("/recommend", params={"title": "Dune"}).status_code == 200


def test_model_endpoint(client):
    r = client.get("/v1/model")
    assert r.status_code == 200 and r.json()["books"] == 12


def test_model_endpoint_404_without_card(client, monkeypatch):
    monkeypatch.setattr(app_module.app.state, "model_card", None)
    assert client.get("/v1/model").status_code == 404


@pytest.mark.parametrize("params", [
    {"title": "Dune", "n": 500},
    {"title": "Dune", "n": 0},
    {"title": "Dune", "min_rating": 6},
    {"title": "Dune", "language": "english123"},
    {},                                    # nothing to search on
])
def test_invalid_input_is_rejected(client, params):
    assert client.get("/v1/recommend", params=params).status_code == 422


def test_unknown_title_is_404(client):
    assert client.get("/v1/recommend", params={"title": "zzzzqq"}).status_code == 404


def test_not_ready_when_model_missing(client, monkeypatch):
    monkeypatch.setattr(app_module.app.state, "predictor", None)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.get("/v1/recommend", params={"title": "Dune"}).status_code == 503


def test_internal_errors_do_not_leak(client, toy_predictor, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(toy_predictor, "recommend", boom)
    r = client.get("/v1/recommend", params={"title": "Dune"})
    assert r.status_code == 500
    assert "secret" not in r.text
    assert "request_id" in r.json()


def test_request_id_is_echoed(client):
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"

def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(app_module.RATE_LIMITER, "limit", 2)
    assert [client.get("/v1/genres").status_code for _ in range(3)] == [200, 200, 429]
    assert client.get("/v1/genres").headers["Retry-After"]
    assert client.get("/health").status_code == 200