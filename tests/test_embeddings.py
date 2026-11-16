import math

import pytest

from app.config import Settings
from app.embeddings import FallbackEmbedder, GeminiEmbedder, get_embedder, hash_embed


def test_fallback_embedder_is_deterministic():
    assert hash_embed("hello world") == hash_embed("hello world")
    assert hash_embed("hello world") != hash_embed("goodbye world")


def test_fallback_embedder_is_normalised():
    vector = hash_embed("some text about cats and dogs")
    norm = math.sqrt(sum(v * v for v in vector))
    assert norm == pytest.approx(1.0)


def test_fallback_embedder_dimension():
    assert len(hash_embed("hello", dim=768)) == 768
    assert len(FallbackEmbedder(dim=768).embed("hello")) == 768


def test_fallback_embedder_empty_text_is_zero_vector():
    assert hash_embed("") == [0.0] * 768


def test_get_embedder_selects_fallback_without_key():
    embedder = get_embedder(Settings(gemini_api_key=""))
    assert embedder.name == "fallback-demo-only"


def test_get_embedder_selects_gemini_with_key():
    embedder = get_embedder(
        Settings(gemini_api_key="test-key", gemini_embed_model="text-embedding-004")
    )
    assert embedder.name == "gemini"


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeHttpClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return FakeResponse(self.payload)


def test_gemini_embedder_request_shape():
    fake = FakeHttpClient({"embedding": {"values": [0.1, 0.2, 0.3]}})
    embedder = GeminiEmbedder("test-key", client=fake)
    values = embedder.embed("hello world")
    assert values == [0.1, 0.2, 0.3]
    call = fake.calls[0]
    assert "text-embedding-004:embedContent" in call["url"]
    assert call["headers"]["x-goog-api-key"] == "test-key"
    assert call["json"]["content"]["parts"][0]["text"] == "hello world"
