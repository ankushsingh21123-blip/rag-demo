"""End-to-end API tests against the in-memory store and the demo-only
fallback embedder. No database, no network, no API keys."""


def test_health_reports_store_and_fallback_provider(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["store"] == "memory"
    assert body["embedding_provider"] == "fallback-demo-only"


def test_ingest_and_list_documents(client):
    response = client.post(
        "/documents",
        json={
            "title": "Paris guide",
            "source": "guide.txt",
            "text": "The Eiffel Tower is in Paris. It was completed in 1889.",
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["chunk_count"] == 1
    assert body["title"] == "Paris guide"

    listing = client.get("/documents")
    assert listing.status_code == 200
    docs = listing.json()
    assert len(docs) == 1
    assert docs[0]["source"] == "guide.txt"


def test_ingest_long_document_produces_multiple_chunks(client):
    text = " ".join(f"sentence number {i} about Paris." for i in range(80))
    response = client.post(
        "/documents",
        json={"title": "Long", "source": "long.txt", "text": text},
    )
    assert response.status_code == 201
    assert response.json()["chunk_count"] > 1


def test_query_answer_is_grounded_in_citations(client):
    client.post(
        "/documents",
        json={
            "title": "Paris guide",
            "source": "guide.txt",
            "text": "The Eiffel Tower is in Paris. It was completed in 1889.",
        },
    )
    response = client.post("/query", json={"question": "Where is the Eiffel Tower?"})
    assert response.status_code == 200
    body = response.json()
    # The answer must quote the retrieved chunk and carry a citation marker.
    assert "Paris" in body["answer"]
    assert "[1]" in body["answer"]
    assert body["answer_provider"] == "extractive-fallback-demo-only"
    assert body["embedding_provider"] == "fallback-demo-only"
    assert len(body["citations"]) == 1
    citation = body["citations"][0]
    assert citation["source"] == "guide.txt"
    assert "Paris" in citation["text"]
    assert citation["rank"] == 1
    assert citation["score"] > 0


def test_retrieval_ranking_prefers_matching_document(client):
    client.post(
        "/documents",
        json={
            "title": "Dogs",
            "source": "dogs.txt",
            "text": "Dogs are loyal animals. Dogs bark and fetch sticks.",
        },
    )
    client.post(
        "/documents",
        json={
            "title": "Cats",
            "source": "cats.txt",
            "text": "Cats are small furry animals. Cats sleep all day and chase mice.",
        },
    )
    response = client.post("/query", json={"question": "cats"})
    citations = response.json()["citations"]
    assert citations[0]["source"] == "cats.txt"
    assert citations[0]["score"] > citations[1]["score"]


def test_query_with_no_documents_is_honest(client):
    response = client.post("/query", json={"question": "anything?"})
    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == []
    assert "No documents" in body["answer"]


def test_top_k_is_respected(client):
    for i in range(5):
        client.post(
            "/documents",
            json={
                "title": f"Doc {i}",
                "source": f"doc{i}.txt",
                "text": f"Document {i} mentions Paris and travel.",
            },
        )
    response = client.post(
        "/query", json={"question": "Paris travel", "top_k": 2}
    )
    assert len(response.json()["citations"]) == 2


def test_ingest_rejects_empty_text(client):
    response = client.post(
        "/documents",
        json={"title": "Empty", "source": "empty.txt", "text": ""},
    )
    assert response.status_code == 422


def test_query_rejects_empty_question(client):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_rejects_top_k_out_of_range(client):
    response = client.post("/query", json={"question": "x", "top_k": 0})
    assert response.status_code == 422
    response = client.post("/query", json={"question": "x", "top_k": 11})
    assert response.status_code == 422
