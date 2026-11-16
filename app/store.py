"""Vector storage.

``PgVectorStore`` is the real store: Postgres + pgvector, addressed by
``DATABASE_URL``. The same connection string works against a Supabase
Postgres project (pgvector is available there). ``InMemoryStore`` is a
dependency-injected stand-in used by the test suite and by ``STORE=memory``
local runs with no database; it implements the same interface with
in-process cosine similarity.
"""

import math
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    doc_id: int
    chunk_index: int
    title: str
    source: str
    text: str
    score: float


@dataclass
class DocumentRecord:
    doc_id: int
    title: str
    source: str
    chunk_count: int


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryStore:
    """In-memory store for tests and keyless local runs."""

    name = "memory"

    def __init__(self):
        self._documents: list[DocumentRecord] = []
        self._chunks: list[dict] = []
        self._next_doc_id = 1

    def init_schema(self) -> None:
        return None

    def add_document(
        self, title: str, source: str, chunks: list[tuple[str, list[float]]]
    ) -> DocumentRecord:
        doc_id = self._next_doc_id
        self._next_doc_id += 1
        for index, (text, embedding) in enumerate(chunks):
            self._chunks.append(
                {
                    "doc_id": doc_id,
                    "chunk_index": index,
                    "title": title,
                    "source": source,
                    "text": text,
                    "embedding": embedding,
                }
            )
        record = DocumentRecord(doc_id, title, source, len(chunks))
        self._documents.append(record)
        return record

    def list_documents(self) -> list[DocumentRecord]:
        return list(self._documents)

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        scored = [
            (cosine_similarity(query_embedding, c["embedding"]), c)
            for c in self._chunks
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(
                doc_id=c["doc_id"],
                chunk_index=c["chunk_index"],
                title=c["title"],
                source=c["source"],
                text=c["text"],
                score=score,
            )
            for score, c in scored[:top_k]
        ]


class PgVectorStore:
    """Postgres + pgvector store. Requires the ``pgvector`` extension.

    Works with a local pgvector container (docker-compose.yml) or with a
    Supabase Postgres project over its connection string.
    """

    name = "postgres"

    def __init__(self, database_url: str, embedding_dim: int = 768):
        if not database_url:
            raise ValueError("DATABASE_URL is required for the postgres store")
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "psycopg is required for the postgres store; "
                "install requirements.txt"
            ) from exc
        self._psycopg = psycopg
        self._database_url = database_url
        self._dim = embedding_dim
        self._conn = None

    def _connect(self):
        if self._conn is None or self._conn.closed:
            self._conn = self._psycopg.connect(self._database_url)
        return self._conn

    @staticmethod
    def _vector_literal(embedding: list[float]) -> str:
        return "[" + ",".join(repr(float(v)) for v in embedding) + "]"

    def init_schema(self) -> None:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    doc_id INTEGER NOT NULL REFERENCES documents(id),
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector({self._dim}) NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks USING ivfflat (embedding vector_cosine_ops)
                """
            )
        conn.commit()

    def add_document(
        self, title: str, source: str, chunks: list[tuple[str, list[float]]]
    ) -> DocumentRecord:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (title, source) VALUES (%s, %s) "
                "RETURNING id",
                (title, source),
            )
            doc_id = cur.fetchone()[0]
            for index, (text, embedding) in enumerate(chunks):
                cur.execute(
                    "INSERT INTO chunks (doc_id, chunk_index, text, embedding) "
                    "VALUES (%s, %s, %s, %s::vector)",
                    (doc_id, index, text, self._vector_literal(embedding)),
                )
        conn.commit()
        return DocumentRecord(doc_id, title, source, len(chunks))

    def list_documents(self) -> list[DocumentRecord]:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.id, d.title, d.source, COUNT(c.id)
                FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id
                GROUP BY d.id, d.title, d.source ORDER BY d.id
                """
            )
            return [DocumentRecord(*row) for row in cur.fetchall()]

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.doc_id, c.chunk_index, d.title, d.source, c.text,
                       1 - (c.embedding <=> %s::vector) AS score
                FROM chunks c JOIN documents d ON d.id = c.doc_id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    self._vector_literal(query_embedding),
                    self._vector_literal(query_embedding),
                    top_k,
                ),
            )
            return [RetrievedChunk(*row) for row in cur.fetchall()]

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None
