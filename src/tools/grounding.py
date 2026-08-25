"""Lightweight grounding check.

Guardrail-in-depth on top of the system-prompt instructions: after an
agent produces an answer, check how much of it is lexically supported
by the chunks it actually retrieved. This is a cheap word-overlap
heuristic, not a second LLM call, so it runs on every turn without
adding real latency or cost.

It is intentionally conservative-in-the-other-direction -- it flags
*possibly* ungrounded sentences for the UI to surface, it does not
block or rewrite the answer. A false positive (e.g. a sentence that is
just a transition like "Here's your budget breakdown:") is cheap; a
silent hallucination is not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "but", "of", "to",
    "in", "on", "for", "with", "at", "by", "from", "this", "that", "it", "as",
    "be", "your", "you", "can", "will", "if", "not", "so", "also", "there",
    "here", "these", "those", "their", "its", "per", "day", "up",
}
MIN_SENTENCE_WORDS = 5
OVERLAP_THRESHOLD = 0.25


def _tokenize(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _split_sentences(text: str) -> list[str]:
    # Skip markdown structural lines (headers, bullets, table rows) --
    # they're formatting, not claims, and would just add noise.
    candidates = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [c.strip() for c in candidates if c.strip() and not c.strip().startswith(("#", "|", "-"))]


@dataclass
class GroundingResult:
    checked_sentences: int
    flagged_sentences: list[str]

    @property
    def has_flags(self) -> bool:
        return bool(self.flagged_sentences)


def check_grounding(answer: str, retrieved_chunks: list[str]) -> GroundingResult:
    """Flag sentences in `answer` with low word-overlap against retrieved context."""
    context_words = set()
    for chunk in retrieved_chunks:
        context_words |= _tokenize(chunk)

    flagged = []
    checked = 0
    for sentence in _split_sentences(answer):
        words = _tokenize(sentence)
        if len(words) < MIN_SENTENCE_WORDS:
            continue
        checked += 1
        overlap = len(words & context_words) / len(words)
        if overlap < OVERLAP_THRESHOLD:
            flagged.append(sentence)

    return GroundingResult(checked_sentences=checked, flagged_sentences=flagged)
