"""Test fixtures: the API runs against the in-memory store and the
demo-only fallback embedder. No database, no Docker, no API keys."""

import os

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings import FallbackEmbedder
from app.main import app, get_embedder_dep, get_settings_dep, get_store
from app.store import InMemoryStore


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("STORE", "memory")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    store = InMemoryStore()
    settings = Settings(database_url="", gemini_api_key="", store="memory")
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_embedder_dep] = lambda: FallbackEmbedder(768)
    app.dependency_overrides[get_settings_dep] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
