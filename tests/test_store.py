"""Store tests.

The in-memory store is exercised directly. The pgvector store is tested
with a fake psycopg module that records the SQL it would execute — no
live Postgres exists on this machine, so these tests verify the SQL
construction (extension, vector column, cosine operator, casts) without
claiming a live database run.
"""

import sys
import types

from app.store import InMemoryStore, PgVectorStore, cosine_similarity


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_in_memory_store_search_ranks_by_similarity():
    store = InMemoryStore()
    store.add_document("A", "a.txt", [("cats", [1.0, 0.0])])
    store.add_document("B", "b.txt", [("dogs", [0.0, 1.0])])
    results = store.search([1.0, 0.0], top_k=2)
    assert results[0].source == "a.txt"
    assert results[0].score == 1.0
    assert results[1].score == 0.0


def test_in_memory_store_lists_documents():
    store = InMemoryStore()
    store.add_document("A", "a.txt", [("x", [1.0]), ("y", [0.5])])
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0].chunk_count == 2


class FakeCursor:
    def __init__(self):
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return (7,)

    def fetchall(self):
        return []


class FakeConnection:
    def __init__(self):
        self.cursor_obj = FakeCursor()
        self.closed = False
        self.commits = 0

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True


def _fake_psycopg(monkeypatch):
    conn = FakeConnection()
    module = types.SimpleNamespace(connect=lambda url: conn)
    monkeypatch.setitem(sys.modules, "psycopg", module)
    return conn


def test_pgvector_store_schema_sql(monkeypatch):
    conn = _fake_psycopg(monkeypatch)
    store = PgVectorStore("postgresql://user:pass@localhost:5432/ragdemo")
    store.init_schema()
    sql = " ".join(call[0] for call in conn.cursor_obj.executed)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "vector(768)" in sql
    assert "vector_cosine_ops" in sql


def test_pgvector_store_insert_sql(monkeypatch):
    conn = _fake_psycopg(monkeypatch)
    store = PgVectorStore("postgresql://user:pass@localhost:5432/ragdemo")
    record = store.add_document("T", "s.txt", [("chunk", [0.1, 0.2])])
    assert record.doc_id == 7
    assert record.chunk_count == 1
    sql = " ".join(call[0] for call in conn.cursor_obj.executed)
    assert "INSERT INTO documents" in sql
    assert "INSERT INTO chunks" in sql
    assert "%s::vector" in sql


def test_pgvector_store_search_uses_cosine_operator(monkeypatch):
    conn = _fake_psycopg(monkeypatch)
    store = PgVectorStore("postgresql://user:pass@localhost:5432/ragdemo")
    store.search([0.1, 0.2], top_k=3)
    sql, params = conn.cursor_obj.executed[0]
    assert "<=>" in sql
    assert params[2] == 3
    assert params[0].startswith("[0.1,0.2]")


def test_pgvector_store_requires_database_url():
    try:
        PgVectorStore("")
    except ValueError as exc:
        assert "DATABASE_URL" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
