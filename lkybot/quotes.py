"""Substring-based quote checker.

Pure functions: no network, no database. A quote is flagged when it has at
least four words and does not appear in the retrieved context after case,
whitespace, and curly-quote normalization.
"""
from __future__ import annotations

import re

QUOTE_RE = re.compile(r'"([^"]+)"')
MIN_QUOTE_WORDS = 4
REPLACEMENT = "[quote removed: not found in sources]"

_CURLY = {
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
}


def _plain_quotes(text: str) -> str:
    for curly, plain in _CURLY.items():
        text = text.replace(curly, plain)
    return text


def normalize(text: str) -> str:
    text = _plain_quotes(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_quotes(text: str) -> list[str]:
    return [match.group(1) for match in QUOTE_RE.finditer(_plain_quotes(text))]


def check_quotes(answer: str, context_chunks: list[str]) -> tuple[str, list[str]]:
    """Return (cleaned_answer, flagged_quotes).

    Quoted text of four or more words that is absent from the retrieved context
    is replaced with a visible marker so the reader knows it was removed.
    """
    haystack = normalize("\n".join(context_chunks))
    flagged: list[str] = []

    def replace(match: re.Match) -> str:
        quote = match.group(1)
        normalized_quote = normalize(quote)
        if len(normalized_quote.split()) < MIN_QUOTE_WORDS:
            return match.group(0)
        if normalized_quote in haystack:
            return match.group(0)
        flagged.append(quote)
        return REPLACEMENT

    cleaned = QUOTE_RE.sub(replace, _plain_quotes(answer))
    return cleaned, flagged
