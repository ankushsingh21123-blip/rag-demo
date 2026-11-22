# Hosted verification: Semantic Support Atlas

Verified 22 November 2026 against https://semantic-support-atlas.vercel.app/semantic in an anonymous browser.

## Production checks

- A browser-local quantised all-MiniLM-L6-v2 embedding reached the Python/LangChain chain and the live PostgreSQL pgvector store. The first webhook-retry paraphrase returned the expected source passage.
- Credential-recovery and concurrent-editor paraphrases returned the expected source documents.
- An unrelated geography question returned no sufficiently close passage.
- No horizontal overflow at 320, 390, 768 and 1280 CSS pixels.
- HTML-like question text rendered as text, not executable elements.
- No browser errors were captured during these checks.
- Desktop and mobile screenshots were visually inspected for readable text, complete results and layout.

The repository test run passed 62 tests across legacy and semantic modes, including 20 tests added for semantic mode. The hosted checks above were separate browser acceptance checks, not part of that unit-test count.

## What this does and does not prove

This AI-assisted portfolio demonstrates browser-local sentence embeddings, a LangChain retrieval chain, a real remote pgvector store and source quotations. It does not use a generative LLM, perform real webhook automation, establish independent coding ability or demonstrate production customer usage. The webhook passage is fictional handbook content, not an implemented webhook endpoint.

The corpus has eight fictional passages. The 0.30 similarity threshold is a demo heuristic, not calibrated confidence. Query text and vectors reach the backend when submitted. No private or customer data should be entered. Free-tier hosting can pause or reach limits, so this dated result is not an uptime promise.
