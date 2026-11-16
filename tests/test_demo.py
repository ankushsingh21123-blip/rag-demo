"""Tests for the live-preview page (GET /demo) and sample-doc seeding.

No database, no Docker, no API keys — the in-memory store and the
demo-only fallback embedder, as in the rest of the suite.
"""

from app.embeddings import FallbackEmbedder
from app.main import app
from app.samples import SAMPLE_DOCS, ensure_sample_docs
from app.store import InMemoryStore


def test_demo_page_returns_html_with_query_ui_markers(client):
    response = client.get("/demo")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    # Query UI markers.
    assert 'id="question"' in html
    assert 'id="ask-button"' in html
    # Ingest form markers.
    assert 'id="ingest-form"' in html
    assert 'id="doc-text"' in html
    # The banner must state plainly that this is a demo.
    assert "Preview demo" in html
    assert "demo-only fallback embedder" in html
    assert "sample documents" in html
    # No external resources: the page must be self-contained.
    assert "http://" not in html
    assert "https://" not in html
    assert "<script src=" not in html
    assert '<link rel="stylesheet"' not in html


def test_root_redirects_to_demo(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/demo"


def test_seeding_is_idempotent_across_queries(client):
    client.post("/query", json={"question": "Where is the Eiffel Tower?"})
    first = client.get("/documents").json()
    client.post("/query", json={"question": "What is photosynthesis?"})
    client.post("/query", json={"question": "How far away is the Moon?"})
    second = client.get("/documents").json()
    assert len(first) == len(second) == len(SAMPLE_DOCS)
    assert [d["doc_id"] for d in first] == [d["doc_id"] for d in second]


def test_ensure_sample_docs_is_idempotent_directly():
    store = InMemoryStore()
    embedder = FallbackEmbedder(768)
    assert ensure_sample_docs(store, embedder) == len(SAMPLE_DOCS)
    assert ensure_sample_docs(store, embedder) == 0
    assert ensure_sample_docs(store, embedder) == 0
    assert len(store.list_documents()) == len(SAMPLE_DOCS)


def test_ensure_sample_docs_skips_nonempty_store():
    store = InMemoryStore()
    embedder = FallbackEmbedder(768)
    store.add_document("Mine", "mine.txt", [("hello world", embedder.embed("hello"))])
    assert ensure_sample_docs(store, embedder) == 0
    assert len(store.list_documents()) == 1


def test_startup_seeding_populates_lifespan_store(client):
    # The TestClient context manager ran the app lifespan, which seeds
    # the lifespan store (separate from the dependency-overridden store
    # used by the endpoints in these tests).
    docs = app.state.store.list_documents()
    assert len(docs) == len(SAMPLE_DOCS)
    assert docs[0].source == "sample-eiffel-tower.txt"
