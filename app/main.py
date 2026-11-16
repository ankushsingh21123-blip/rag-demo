"""RAG demo API: FastAPI + pgvector + Gemini, with $0 fallback paths.

Endpoints:
  GET  /health      — status, active store, active embedding provider
  POST /documents   — ingest a text document (chunked + embedded + stored)
  GET  /documents   — list ingested documents
  POST /query       — top-k retrieval + grounded answer with citations
  GET  /demo        — self-contained live-preview page (no build step)
  GET  /            — redirect to /demo

Sample documents are seeded on startup and lazily before a query when
the store is empty (see samples.py), so the preview works instantly
with zero setup and zero keys. Seeding is idempotent.

When GEMINI_API_KEY is unset, embeddings and answers use the
clearly-labelled demo-only fallback paths (see embeddings.py and
answering.py). Set STORE=memory to run with no database at all.
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from .answering import GeminiAnswerer, extractive_answer
from .chunking import chunk_text
from .config import Settings, get_settings
from .demo import DEMO_HTML
from .embeddings import get_embedder
from .samples import ensure_sample_docs
from .store import InMemoryStore, PgVectorStore, RetrievedChunk


class DocumentIn(BaseModel):
    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)


class DocumentOut(BaseModel):
    doc_id: int
    title: str
    source: str
    chunk_count: int


class QueryIn(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class Citation(BaseModel):
    rank: int
    doc_id: int
    chunk_index: int
    title: str
    source: str
    text: str
    score: float


class QueryOut(BaseModel):
    answer: str
    citations: list[Citation]
    answer_provider: str
    embedding_provider: str


def build_store(settings: Settings):
    if settings.store == "memory":
        return InMemoryStore()
    return PgVectorStore(settings.database_url, settings.embedding_dim)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    store = build_store(settings)
    store.init_schema()
    embedder = get_embedder(settings)
    ensure_sample_docs(store, embedder)
    app.state.store = store
    app.state.settings = settings
    app.state.embedder = embedder
    yield
    if hasattr(store, "close"):
        store.close()


app = FastAPI(title="rag-demo", lifespan=lifespan)


def _ensure_state():
    """Initialise app state on first use when the lifespan did not run.

    Some serverless runtimes do not deliver lifespan events; without this
    fallback the dependency getters below would fail on a cold start.
    When the lifespan did run, this is a no-op.
    """
    if getattr(app.state, "store", None) is None:
        settings = get_settings()
        store = build_store(settings)
        store.init_schema()
        embedder = get_embedder(settings)
        ensure_sample_docs(store, embedder)
        app.state.store = store
        app.state.settings = settings
        app.state.embedder = embedder


def get_store():
    _ensure_state()
    return app.state.store


def get_embedder_dep():
    _ensure_state()
    return app.state.embedder


def get_settings_dep():
    _ensure_state()
    return app.state.settings


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/demo", status_code=307)


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo_page():
    return DEMO_HTML


@app.get("/health")
def health(
    store=Depends(get_store),
    embedder=Depends(get_embedder_dep),
):
    return {
        "status": "ok",
        "store": store.name,
        "embedding_provider": embedder.name,
    }


@app.post("/documents", response_model=DocumentOut, status_code=201)
def ingest_document(
    doc: DocumentIn,
    store=Depends(get_store),
    embedder=Depends(get_embedder_dep),
):
    chunks = chunk_text(doc.text)
    if not chunks:
        raise HTTPException(status_code=422, detail="document text is empty")
    embedded = [(chunk, embedder.embed(chunk)) for chunk in chunks]
    record = store.add_document(doc.title, doc.source, embedded)
    return DocumentOut(
        doc_id=record.doc_id,
        title=record.title,
        source=record.source,
        chunk_count=record.chunk_count,
    )


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(store=Depends(get_store)):
    return [
        DocumentOut(
            doc_id=r.doc_id,
            title=r.title,
            source=r.source,
            chunk_count=r.chunk_count,
        )
        for r in store.list_documents()
    ]


@app.post("/query", response_model=QueryOut)
def query(
    query_in: QueryIn,
    store=Depends(get_store),
    embedder=Depends(get_embedder_dep),
    settings: Settings = Depends(get_settings_dep),
):
    # Lazy seeding: the preview works instantly on an empty store, and
    # seeding is idempotent, so this is a no-op once documents exist.
    ensure_sample_docs(store, embedder)
    query_embedding = embedder.embed(query_in.question)
    retrieved: list[RetrievedChunk] = store.search(query_embedding, query_in.top_k)
    citations = [
        Citation(
            rank=i,
            doc_id=c.doc_id,
            chunk_index=c.chunk_index,
            title=c.title,
            source=c.source,
            text=c.text,
            score=c.score,
        )
        for i, c in enumerate(retrieved, start=1)
    ]
    if settings.gemini_api_key:
        answer = GeminiAnswerer(
            settings.gemini_api_key, settings.gemini_model
        ).answer(query_in.question, retrieved)
        answer_provider = "gemini"
    else:
        answer = extractive_answer(retrieved)
        answer_provider = "extractive-fallback-demo-only"
    return QueryOut(
        answer=answer,
        citations=citations,
        answer_provider=answer_provider,
        embedding_provider=embedder.name,
    )
