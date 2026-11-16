import pytest

from app.chunking import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_long_text_splits_within_max_chars():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, max_chars=100, overlap_chars=10)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_chunks_cover_all_words_in_order():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, max_chars=120, overlap_chars=20)
    seen = []
    for chunk in chunks:
        for word in chunk.split(" "):
            if word not in seen:
                seen.append(word)
    assert seen == [f"word{i}" for i in range(100)]


def test_overlap_repeats_trailing_words():
    text = " ".join(f"w{i}" for i in range(60))
    chunks = chunk_text(text, max_chars=50, overlap_chars=15)
    assert len(chunks) > 1
    first_words = set(chunks[0].split(" "))
    second_words = set(chunks[1].split(" "))
    assert first_words & second_words


def test_empty_and_whitespace_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        chunk_text("hello", max_chars=0)
    with pytest.raises(ValueError):
        chunk_text("hello", max_chars=100, overlap_chars=100)
    with pytest.raises(ValueError):
        chunk_text("hello", max_chars=100, overlap_chars=-1)
