"""Vercel serverless entrypoint: exposes the FastAPI app as an ASGI app.

Defaults to the in-memory store when neither STORE nor DATABASE_URL is
configured, so a fresh deploy works with zero env vars (demo-only
fallback paths). To use a real Postgres/pgvector database instead, set
DATABASE_URL (and STORE=postgres) in the Vercel dashboard — the
``setdefault`` logic below leaves an explicitly configured store alone.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not os.environ.get("STORE") and not os.environ.get("DATABASE_URL"):
    os.environ["STORE"] = "memory"

from app.main import app  # noqa: E402

__all__ = ["app"]
