# ai-written
import pytest
from fastapi.testclient import TestClient

from core.domain import Domain


@pytest.fixture
def chats_db(tmp_path, monkeypatch):
    monkeypatch.setenv("STRAW_CHATS_DB", str(tmp_path / "chats.db"))


def client(token, monkeypatch):
    if token is None:
        monkeypatch.delenv("STRAW_TOKEN", raising=False)
    else:
        monkeypatch.setenv("STRAW_TOKEN", token)
    return TestClient(Domain("test", "seed").app)


def test_no_token_configured_rejects_missing_header(chats_db, monkeypatch):
    assert client(None, monkeypatch).get("/chats").status_code == 503


def test_no_token_configured_rejects_empty_bearer(chats_db, monkeypatch):
    r = client(None, monkeypatch).get("/chats", headers={"Authorization": "Bearer "})
    assert r.status_code == 503


def test_wrong_token_rejected(chats_db, monkeypatch):
    r = client("secret", monkeypatch).get("/chats", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_missing_header_rejected(chats_db, monkeypatch):
    assert client("secret", monkeypatch).get("/chats").status_code == 401


def test_right_token_accepted(chats_db, monkeypatch):
    r = client("secret", monkeypatch).get("/chats", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200
