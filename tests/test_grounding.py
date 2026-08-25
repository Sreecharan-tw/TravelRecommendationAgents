"""Tests for the lexical-overlap grounding check guardrail."""
from __future__ import annotations

from src.tools.grounding import check_grounding


def test_grounded_answer_is_not_flagged():
    chunks = [
        "Gulmarg has India's only ski resort with one of the highest gondolas in the world.",
    ]
    answer = "Gulmarg is home to India's only ski resort and a very high gondola."
    result = check_grounding(answer, chunks)
    assert result.flagged_sentences == []


def test_ungrounded_sentence_is_flagged():
    chunks = [
        "Gulmarg has India's only ski resort.",
    ]
    answer = "The Eiffel Tower in Paris is a must-see attraction with a rooftop restaurant."
    result = check_grounding(answer, chunks)
    assert len(result.flagged_sentences) == 1


def test_short_sentences_are_not_checked():
    result = check_grounding("Yes. Sure. OK.", ["some unrelated context"])
    assert result.checked_sentences == 0
    assert result.flagged_sentences == []
