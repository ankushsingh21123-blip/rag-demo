# Semantic Support Atlas

AI-assisted synthetic portfolio implementation. This is not an employer assessment, a production service or a claim of unaided authorship.

The current deployment entrypoint is `app.semantic:app`. It combines a browser-local semantic embedding model, a Python/FastAPI backend, a LangChain retrieval/citation chain and a real PostgreSQL/pgvector database. It searches eight fixed fictional support passages and returns exact quotations with source titles and cosine similarity.

## What is actually implemented

- Transformers.js runs quantised `Xenova/all-MiniLM-L6-v2` in the browser. It downloads about 23 MB of model weights plus runtime on first use, then produces normalised 384-dimensional vectors locally.
- `POST /query` validates the question and vector, then runs a LangChain `RunnableLambda` retrieval and citation pipeline.
- A narrow Supabase RPC executes pgvector cosine-distance search on the real database. The corpus is in its own private schema with RLS enabled and no anonymous direct-table grants. The RPC is read-only, bounded to five results and a one-second database statement timeout.
- `GET /health` reads the live database. There is no in-memory or hash-embedding substitution in this entrypoint. Database failure returns 503.
- A 0.30 cosine-similarity threshold suppresses weak matches. It is a demo heuristic, not a calibrated answer-confidence score.
- All output uses exact source quotations and safe text rendering. No generative LLM is connected. There is no OpenAI/Anthropic/Gemini API call in the hosted semantic entrypoint.

The publishable database configuration in `app/semantic.py` is intentionally public. It is not a database password or service-role key. No private credentials belong in this repository or deployment.

## Verification, 22 November 2026

Local FastAPI server against the actual remote pgvector database:

- 62 pytest tests passed, including 20 new semantic-mode tests and 42 legacy tests.
- Browser-local ONNX inference, LangChain and real pgvector retrieval worked end to end. Webhook retries, forgotten credentials and concurrent-editor paraphrases returned the expected source passages.
- An unrelated capital-city question returned no sufficiently close passage.
- No page errors, no horizontal overflow at 320/390/768/1280 pixels, safe handling of HTML-like input. Desktop and mobile screenshots inspected.
- Exact-vector database retrieval scored above 0.999. Direct anonymous table access was denied. Invalid vectors, oversized bodies, missing routes and unavailable storage were checked.

These are local-application plus live-database checks. A public-host check is a separate step and must not be inferred from them. See `HOSTED_VERIFY.md` when available for the hosted URL and acceptance result.

## Run

```sh
python -m pip install -r requirements.txt
python -m uvicorn app.semantic:app --host 127.0.0.1 --port 8000
python -m pip install pytest
python -m pytest -q
```

The default public corpus is read-only. To use another isolated database, apply `schema.semantic.sql`, seed the fictional passages from `seed-data.json` with an authorised database administrator, then set `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY`. Do not apply this demo's schema to a product database. The eight stored vectors were computed using the same quantised MiniLM model and mean pooling/normalisation as the browser.

## Limits

Synthetic data only. Do not enter private or customer data. Questions and vectors reach this demo's backend but are not saved by the app. Hosting providers may keep operational logs. The UI downloads model/runtime assets from Hugging Face and jsDelivr.

There is no document upload, authentication, billing or write endpoint in semantic mode. Eight documents do not establish large-scale retrieval quality, production security or general RAG competence. No LLM generation, reranking, streaming or evaluation benchmark is claimed. Public traffic can exhaust free-tier usage; hosts can pause. No paid plan, card, trial, upgrade or uptime guarantee is part of this deployment.

The original Gemini/hash/memory demonstration remains in `app.main`, with its original claims and limitations preserved in `LEGACY_README.md`. It is not the semantic deployment entrypoint. Legacy Render/Docker configs still target that original mode unless changed explicitly.

## Credits and licences

Repository code: MIT. Transformers.js and MiniLM model: Apache-2.0. LangChain and FastAPI: MIT. PostgreSQL and pgvector: PostgreSQL licence. Existing source history is preserved. Hosted Vercel and Supabase are free-tier services, not a claim that the hosting services themselves are open source.
