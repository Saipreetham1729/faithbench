"""Core data types. Deliberately plain -- dataclasses and lists/dicts, no
pydantic or other framework dependency, so this stays easy to drop into any
codebase regardless of what else it uses.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class EvidenceChunk:
    """One piece of retrieved evidence a claim can be checked against."""

    text: str
    id: Optional[str] = None
    page_number: Optional[int] = None

    def __post_init__(self):
        if self.id is None:
            self.id = f"chunk_{abs(hash(self.text)) % 100000}"


@dataclass
class Claim:
    id: str
    text: str


@dataclass
class MethodScore:
    """One method's verdict on one claim. score is None (never 0.0) when the
    method couldn't run -- e.g. no LLM callable was provided, or a network
    call failed. Callers should treat None as "not evaluated", not as zero.
    """

    method: str
    score: Optional[float]
    detail: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ClaimResult:
    claim: Claim
    scores: Dict[str, MethodScore] = field(default_factory=dict)

    @property
    def aggregate_score(self) -> Optional[float]:
        """Mean of every method that actually produced a score."""
        values = [s.score for s in self.scores.values() if s.score is not None]
        return round(sum(values) / len(values), 4) if values else None

    def status(self, supported_threshold: float = 0.6, partial_threshold: float = 0.3) -> str:
        """A simple supported/partially_supported/unsupported/unevaluated
        label from the aggregate score. Purely a convenience -- the per-method
        scores in `.scores` are the actual data; this is one opinionated view
        of them.
        """
        agg = self.aggregate_score
        if agg is None:
            return "unevaluated"
        if agg >= supported_threshold:
            return "supported"
        if agg >= partial_threshold:
            return "partially_supported"
        return "unsupported"


@dataclass
class CheckResult:
    answer: str
    claims: List[ClaimResult]

    @property
    def overall_score(self) -> Optional[float]:
        values = [c.aggregate_score for c in self.claims if c.aggregate_score is not None]
        return round(sum(values) / len(values), 4) if values else None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "overall_score": self.overall_score,
            "claims": [
                {
                    "claim_id": c.claim.id,
                    "text": c.claim.text,
                    "aggregate_score": c.aggregate_score,
                    "status": c.status(),
                    "scores": {
                        name: {"score": s.score, "detail": s.detail, "error": s.error}
                        for name, s in c.scores.items()
                    },
                }
                for c in self.claims
            ],
        }


EvidenceInput = Union[str, EvidenceChunk]


def normalize_evidence(context: List[EvidenceInput]) -> List[EvidenceChunk]:
    """Accepts plain strings or EvidenceChunk objects -- lets the simple
    case (`context=["chunk1", "chunk2"]`) stay simple, without losing the
    richer form for callers who want chunk IDs/page numbers preserved.
    """
    return [c if isinstance(c, EvidenceChunk) else EvidenceChunk(text=c) for c in context]
