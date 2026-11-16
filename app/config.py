"""Configuration from environment variables only. No secrets are hardcoded.

``get_settings()`` reads the environment fresh on every call so tests and
local runs can change configuration without re-importing the module.
"""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Postgres connection string. Works against a local pgvector container
    # (see docker-compose.yml) or a Supabase Postgres project, which ships
    # pgvector. No Supabase project is created by this repo.
    database_url: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_embed_model: str = "text-embedding-004"
    # "postgres" (default) or "memory" (in-memory store, no database needed).
    store: str = "postgres"
    # text-embedding-004 produces 768-dimensional vectors. The demo-only
    # fallback embedder uses the same dimension so both paths share one schema.
    embedding_dim: int = 768


def get_settings() -> Settings:
    return Settings(
        database_url=os.environ.get("DATABASE_URL", ""),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        gemini_embed_model=os.environ.get(
            "GEMINI_EMBED_MODEL", "text-embedding-004"
        ),
        store=os.environ.get("STORE", "postgres"),
        embedding_dim=int(os.environ.get("EMBEDDING_DIM", "768")),
    )
