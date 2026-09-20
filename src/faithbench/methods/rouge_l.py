"""
ROUGE-L: lexical overlap (longest common subsequence F1) between each claim
and its best-matching evidence chunk. The cheap, dependency-light default --
no model download, no API call, no GPU. Good baseline; not a substitute for
a semantic method when wording differs a lot between claim and evidence.
"""
from typing import List

from faithbench.types import Claim, EvidenceChunk, MethodScore

try:
    from rouge_score import rouge_scorer
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

name = "rouge_l"
_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True) if _AVAILABLE else None


def score_claims(claims: List[Claim], evidence: List[EvidenceChunk]) -> List[MethodScore]:
    if not _AVAILABLE:
        return [
            MethodScore(
                method=name, score=None,
                error="rouge-score is not installed. Install with: pip install faithbench[rouge]",
            )
            for _ in claims
        ]
    if not evidence:
        return [MethodScore(method=name, score=0.0, detail="No evidence provided.") for _ in claims]

    results = []
    for claim in claims:
        best_f1 = 0.0
        best_chunk_id = None
        for chunk in evidence:
            f1 = _scorer.score(chunk.text, claim.text)["rougeL"].fmeasure
            if f1 > best_f1:
                best_f1 = f1
                best_chunk_id = chunk.id
        results.append(MethodScore(method=name, score=round(best_f1, 4), detail=f"best match: {best_chunk_id}"))
    return results
