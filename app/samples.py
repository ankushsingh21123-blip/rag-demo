"""Pre-loaded sample documents for the live preview (``GET /demo``).

Generic public facts only — nothing about any person, client, or venture.
Seeding is idempotent: it only runs when the store is empty, so repeated
calls, restarts, and concurrent first queries never duplicate documents.
"""

import threading

from .chunking import chunk_text

SAMPLE_DOCS = [
    {
        "title": "Eiffel Tower",
        "source": "sample-eiffel-tower.txt",
        "text": (
            "The Eiffel Tower is in Paris, France. "
            "It was completed in 1889 for the World's Fair. "
            "The tower is 330 metres tall including its antennas."
        ),
    },
    {
        "title": "FastAPI",
        "source": "sample-fastapi.txt",
        "text": (
            "FastAPI is a modern web framework for building APIs with Python. "
            "It is based on standard Python type hints and uses Starlette "
            "and Pydantic. FastAPI automatically generates interactive API "
            "documentation."
        ),
    },
    {
        "title": "The Moon",
        "source": "sample-moon.txt",
        "text": (
            "The Moon is Earth's only natural satellite. "
            "It orbits Earth at an average distance of about 384,400 "
            "kilometres. The first crewed Moon landing was Apollo 11 in 1969."
        ),
    },
    {
        "title": "Photosynthesis",
        "source": "sample-photosynthesis.txt",
        "text": (
            "Photosynthesis is the process by which green plants make food "
            "using sunlight, water, and carbon dioxide. "
            "It produces oxygen as a byproduct. "
            "Photosynthesis takes place in chloroplasts inside plant cells."
        ),
    },
    {
        "title": "Great Wall of China",
        "source": "sample-great-wall.txt",
        "text": (
            "The Great Wall of China is a series of fortifications built "
            "across northern China. Its construction began more than 2,000 "
            "years ago. The wall was built to protect Chinese states from "
            "invasions from the north."
        ),
    },
]

_seed_lock = threading.Lock()


def ensure_sample_docs(store, embedder) -> int:
    """Ingest the sample documents if the store is empty.

    Returns the number of documents ingested (0 when the store already
    contains documents). Safe to call on every query: the empty check
    runs under a lock, so concurrent first queries cannot double-seed.
    """
    with _seed_lock:
        if store.list_documents():
            return 0
        count = 0
        for doc in SAMPLE_DOCS:
            chunks = chunk_text(doc["text"])
            embedded = [(chunk, embedder.embed(chunk)) for chunk in chunks]
            store.add_document(doc["title"], doc["source"], embedded)
            count += 1
        return count
