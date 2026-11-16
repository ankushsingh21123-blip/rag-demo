"""Answer generation: Gemini grounded generation, or a clearly-labelled
extractive fallback.

``extractive_answer`` is DEMO ONLY: it quotes the retrieved chunks
directly and cites them, so the API works with $0 and no API keys. It
does no language-model reasoning and must never be presented as an LLM
answer. When ``GEMINI_API_KEY`` is set, ``GeminiAnswerer`` generates an
answer grounded in the retrieved chunks with citation markers.
"""

import re

import httpx

from .store import RetrievedChunk


def _first_sentence(text: str) -> str:
    match = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return match[0].strip() if match else text.strip()


def extractive_answer(chunks: list[RetrievedChunk]) -> str:
    """DEMO ONLY extractive answer. Quotes retrieved chunks with citations."""
    if not chunks:
        return "No documents have been ingested yet, so there is nothing to answer from."
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"{_first_sentence(chunk.text)} [{i}]")
    return " ".join(parts)


class GeminiAnswerer:
    """Grounded answer generation via the Gemini API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        client: httpx.Client | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self._client = client

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No documents have been ingested yet, so there is nothing to answer from."
        context = "\n\n".join(
            f"[{i}] (source: {c.source}) {c.text}"
            for i, c in enumerate(chunks, start=1)
        )
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the context does not contain the answer, say so. "
            "Cite the context with markers like [1], [2].\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"x-goog-api-key": self.api_key}
        if self._client is not None:
            response = self._client.post(url, json=payload, headers=headers)
        else:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
