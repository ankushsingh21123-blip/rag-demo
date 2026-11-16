"""Deterministic word-window chunking with character overlap."""


def chunk_text(text: str, max_chars: int = 800, overlap_chars: int = 100) -> list[str]:
    """Split text into chunks of at most ``max_chars`` characters.

    Chunking is word-based so words are never split mid-token, and each
    chunk after the first re-includes up to ``overlap_chars`` of trailing
    words from the previous chunk so context survives chunk boundaries.
    Empty or whitespace-only input returns an empty list.
    """
    text = " ".join(text.split())
    if not text:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be >= 0 and < max_chars")

    words = text.split(" ")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            chunks.append(" ".join(current))
            # Carry trailing words totalling at most overlap_chars forward.
            keep: list[str] = []
            keep_len = 0
            for prev in reversed(current):
                if keep and keep_len + len(prev) + 1 > overlap_chars:
                    break
                keep.append(prev)
                keep_len += len(prev) + 1
            current = list(reversed(keep))
            current_len = len(" ".join(current))
            extra = len(word) + (1 if current else 0)
        current.append(word)
        current_len += extra

    if current:
        chunks.append(" ".join(current))
    return chunks
