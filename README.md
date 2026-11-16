# rag-demo

A small, honest RAG (retrieval-augmented generation) demo: ingest text
documents, retrieve the most relevant chunks, and get a grounded answer
with citations. Built as a proof project in **Python** with **FastAPI**,
**pgvector**, **Supabase-compatible Postgres**, **RAG** retrieval, the
**Gemini API**, **pytest**, and **Docker**.

## What is real, and what is not

**Real, working code:**

- FastAPI API: `POST /documents`, `GET /documents`, `POST /query`,
  `GET /health`.
- Word-window chunking with overlap (`app/chunking.py`).
- Cosine-similarity retrieval over pgvector (`app/store.py`), using the
  pgvector `<=>` operator and a `vector_cosine_ops` index.
- Gemini embeddings (`text-embedding-004`) and grounded answer generation
  via the Gemini API, when `GEMINI_API_KEY` is set.
- A pytest suite (36 tests) covering chunking, retrieval ranking,
  citation grounding, API validation, the pgvector SQL construction, and
  the fallback paths.

**Demo-only fallbacks (labelled as such in code, API responses, and here):**

- When no `GEMINI_API_KEY` is set, embeddings use a deterministic
  hash-based embedder and answers are extractive quotes from the
  retrieved chunks. These paths exist so the app and tests run with $0
  and no keys. The hash embedder captures token overlap only — it has no
  semantic understanding and is **not** production embeddings.
- `STORE=memory` runs an in-memory store with no database. The pgvector
  store is the real store; the in-memory store exists for tests and
  keyless local runs.

**Not verified:** this demo has not been run against a live Postgres +
pgvector database or a live Gemini API call from this machine (no Docker
or Postgres was available where it was built, and no API key was used).
The pgvector SQL is covered by tests that record the SQL against a fake
connection, and the Gemini request shapes are covered by tests with a
fake HTTP client. Run `docker compose up` and set `GEMINI_API_KEY` to
verify the live paths yourself.

**Not production:** no auth, no rate limiting, no multi-tenancy, no
streaming. This is a demo, not a production system.

## Quick start ($0, no keys, no database)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
STORE=memory uvicorn app.main:app --port 8000
```

```bash
curl localhost:8000/health
curl -X POST localhost:8000/documents \
  -H 'Content-Type: application/json' \
  -d '{"title":"Paris guide","source":"guide.txt",
       "text":"The Eiffel Tower is in Paris. It was completed in 1889."}'
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Where is the Eiffel Tower?"}'
```

## Run with Postgres + pgvector (Docker)

```bash
docker compose up --build
```

This starts the `pgvector/pgvector:pg16` image and the API. The API
creates the `vector` extension, the `documents` and `chunks` tables, and
the cosine index on startup.

## Run against Supabase Postgres

Supabase Postgres ships pgvector. Create a project in the Supabase
dashboard (not done by this repo), then point `DATABASE_URL` at it:

```bash
DATABASE_URL=postgresql://postgres:<password>@<host>:5432/postgres \
  uvicorn app.main:app --port 8000
```

The same schema code runs unchanged — the only difference is the
connection string.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATABASE_URL` | _(empty)_ | Postgres connection string (local pgvector container or Supabase). Required unless `STORE=memory`. |
| `GEMINI_API_KEY` | _(empty)_ | Gemini API key. Empty = demo-only fallback paths. Never commit a key; see `.env.example`. |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Model for answer generation. Set to a model available on your key. |
| `GEMINI_EMBED_MODEL` | `text-embedding-004` | Embedding model (768 dimensions). |
| `STORE` | `postgres` | `postgres` or `memory`. |
| `EMBEDDING_DIM` | `768` | Vector dimension; must match the embedding model. |

Copy `.env.example` to `.env` for local values. `.env` is git-ignored.

## API

### `GET /health`

```json
{"status": "ok", "store": "postgres", "embedding_provider": "gemini"}
```

`embedding_provider` is `gemini` when a key is set, otherwise
`fallback-demo-only`.

### `POST /documents` → 201

```json
{"title": "Paris guide", "source": "guide.txt", "text": "..."}
```

Response: `{"doc_id": 1, "title": "Paris guide", "source": "guide.txt", "chunk_count": 3}`

### `GET /documents`

Lists ingested documents with chunk counts.

### `POST /query`

```json
{"question": "Where is the Eiffel Tower?", "top_k": 3}
```

Response:

```json
{
  "answer": "The Eiffel Tower is in Paris. [1]",
  "citations": [
    {"rank": 1, "doc_id": 1, "chunk_index": 0, "title": "Paris guide",
     "source": "guide.txt", "text": "The Eiffel Tower is in Paris...",
     "score": 0.46}
  ],
  "answer_provider": "gemini",
  "embedding_provider": "gemini"
}
```

Every answer carries citations to the retrieved chunks, with the
similarity score, so grounding can be checked. With no key set,
`answer_provider` is `extractive-fallback-demo-only`.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

36 tests, all passing, with no API keys, no Docker, and no database:

- Chunking: splitting, overlap, coverage, empty input, invalid params.
- Embeddings: fallback determinism, normalisation, dimension, provider
  selection, Gemini request shape (fake HTTP client).
- API: health, ingest/list, grounded answers with citations, retrieval
  ranking, empty-store honesty, `top_k`, validation errors.
- Store: cosine similarity, in-memory ranking, pgvector SQL construction
  (extension, `vector(768)` column, `<=>` cosine operator, `::vector`
  casts) against a fake connection.
- Answering: extractive citations, empty-context honesty, Gemini prompt
  grounding (fake HTTP client, no live call).

Verified 16 Nov 2026: **36 passed** on Python 3.12, plus a live
`uvicorn` smoke run (ingest → query → list) against the in-memory store
and fallback paths.

## Project layout

```
app/
  main.py        FastAPI app and endpoints
  chunking.py    Word-window chunking with overlap
  embeddings.py  Gemini embedder + demo-only fallback embedder
  store.py       PgVectorStore + InMemoryStore, cosine retrieval
  answering.py   Gemini grounded answers + extractive fallback
  config.py      Environment-only configuration
tests/           pytest suite (36 tests)
docker-compose.yml   pgvector/pgvector:pg16 + API service
Dockerfile
```

## Licence

MIT — see [LICENSE](LICENSE).
