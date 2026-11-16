"""Answer-generation tests. The Gemini path is tested with a fake HTTP
client that records the request — no live API call is ever made."""

from app.answering import GeminiAnswerer, extractive_answer
from app.store import RetrievedChunk


def _chunk(text, source="guide.txt"):
    return RetrievedChunk(
        doc_id=1, chunk_index=0, title="Guide", source=source, text=text, score=0.9
    )


def test_extractive_answer_quotes_chunks_with_citations():
    chunks = [
        _chunk("The Eiffel Tower is in Paris. It was completed in 1889."),
        _chunk("Paris is the capital of France.", source="facts.txt"),
    ]
    answer = extractive_answer(chunks)
    assert "The Eiffel Tower is in Paris. [1]" in answer
    assert "Paris is the capital of France. [2]" in answer


def test_extractive_answer_with_no_chunks_is_honest():
    assert "No documents" in extractive_answer([])


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeHttpClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return FakeResponse(self.payload)


def test_gemini_answerer_grounds_prompt_in_context():
    canned = {
        "candidates": [
            {"content": {"parts": [{"text": "The Eiffel Tower is in Paris. [1]"}]}}
        ]
    }
    fake = FakeHttpClient(canned)
    answerer = GeminiAnswerer("test-key", model="gemini-2.0-flash", client=fake)
    answer = answerer.answer(
        "Where is the Eiffel Tower?",
        [_chunk("The Eiffel Tower is in Paris.")],
    )
    assert answer == "The Eiffel Tower is in Paris. [1]"
    call = fake.calls[0]
    assert "gemini-2.0-flash:generateContent" in call["url"]
    assert call["headers"]["x-goog-api-key"] == "test-key"
    prompt = call["json"]["contents"][0]["parts"][0]["text"]
    assert "The Eiffel Tower is in Paris." in prompt
    assert "ONLY the context" in prompt
    assert "Where is the Eiffel Tower?" in prompt


def test_gemini_answerer_with_no_chunks_does_not_call_api():
    fake = FakeHttpClient({})
    answerer = GeminiAnswerer("test-key", client=fake)
    answer = answerer.answer("anything?", [])
    assert "No documents" in answer
    assert fake.calls == []
