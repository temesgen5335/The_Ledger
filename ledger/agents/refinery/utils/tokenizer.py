"""Token counting for LDU size and max_tokens enforcement."""

import re


def count_tokens(text: str) -> int:
    """
    Estimate token count from text (cheap heuristic).
    Uses word count * 1.3 as approximation for English.
    """
    if not text or not text.strip():
        return 0
    words = len(re.findall(r"\S+", text))
    if words > 0:
        return max(1, int(words * 1.3))
    return max(1, len(text) // 4)
