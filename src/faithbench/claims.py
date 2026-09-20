"""
Splits a generated answer into individual claims to check against evidence.

One claim per sentence: simple, explainable, and easy to see exactly why
each "claim" was extracted. A real claim can span multiple sentences, or one
sentence can bundle several claims -- sentence-level segmentation is a
reasonable, inspectable default rather than a claim of perfect accuracy.
"""
import re
from typing import List

from faithbench.types import Claim

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MIN_WORDS = 3  # skip fragments like "Yes." or "Okay." -- not really checkable claims


def extract_claims(answer: str) -> List[Claim]:
    if not answer or not answer.strip():
        return []

    sentences = _SENTENCE_SPLIT_RE.split(answer.strip())
    claims: List[Claim] = []
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence or len(sentence.split()) < _MIN_WORDS:
            continue
        claims.append(Claim(id=f"claim_{i}", text=sentence))
    return claims
