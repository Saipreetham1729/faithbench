"""
Every method implements one function: score_claims(claims, evidence) ->
list of MethodScore, one per claim, same order as `claims` in. Batched
rather than one-claim-at-a-time so LLM-based methods can score every claim
in a single call instead of N separate ones.

Keeping this a plain protocol (not a class hierarchy) makes it trivial to
register a custom method alongside the built-in ones -- any object with a
`name` attribute and a matching `score_claims` callable works.
"""
from typing import List, Protocol

from faithbench.types import Claim, EvidenceChunk, MethodScore


class ScoringMethod(Protocol):
    name: str

    def score_claims(self, claims: List[Claim], evidence: List[EvidenceChunk]) -> List[MethodScore]:
        ...
