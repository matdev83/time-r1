from __future__ import annotations

BAD_WORDS = {"bad", "toxic"}


def is_toxic(text: str) -> bool:
    lowered = text.lower()
    return any(w in lowered for w in BAD_WORDS)
