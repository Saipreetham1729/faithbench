"""
RAGAS-style faithfulness: a strictly binary yes/no inference check per
claim, following RAGAS's own methodology, rather than the 3-way verdict
llm_judge uses. Same (system, user) -> str llm_fn interface as llm_judge --
see that module's docstring for how to wire up any LLM provider.
"""
import json
from typing import Callable, List

from faithbench.types import Claim, EvidenceChunk, MethodScore

name = "ragas_faithfulness"

_SYSTEM_PROMPT = (
    "You are evaluating whether statements can be directly inferred from a given context, "
    "following the RAGAS faithfulness methodology. You will be given a list of numbered "
    "statements and a block of context. For EACH statement, answer STRICTLY 'yes' or 'no': "
    "'yes' only if the statement can be directly inferred from the context alone, with no "
    "outside knowledge or assumptions. There is no partial credit. Respond with ONLY a JSON "
    'array, one object per statement: [{"claim_id": "...", "verdict": "yes"|"no", "reason": "..."}]. '
    "No text outside the JSON array."
)


def _build_user_prompt(claims: List[Claim], evidence: List[EvidenceChunk]) -> str:
    statements_block = "\n".join(f"{c.id}: {c.text}" for c in claims)
    context_block = "\n\n".join(chunk.text for chunk in evidence)
    return f"Context:\n{context_block}\n\nStatements:\n{statements_block}\n\nRespond with the JSON array now."


def _parse_response(raw_response: str):
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        parsed = json.loads(cleaned)
        return {item["claim_id"]: item for item in parsed}
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def make_scorer(llm_fn: Callable[[str, str], str]):
    """llm_fn(system_prompt, user_prompt) -> str."""

    class _RagasFaithfulness:
        name = "ragas_faithfulness"

        def score_claims(self, claims: List[Claim], evidence: List[EvidenceChunk]) -> List[MethodScore]:
            if not claims:
                return []
            if not evidence:
                return [MethodScore(method=name, score=0.0, detail="no") for _ in claims]

            try:
                raw = llm_fn(_SYSTEM_PROMPT, _build_user_prompt(claims, evidence))
            except Exception as exc:  # noqa: BLE001
                return [MethodScore(method=name, score=None, error=str(exc)) for _ in claims]

            by_id = _parse_response(raw)
            if by_id is None:
                return [
                    MethodScore(method=name, score=None, error="RAGAS faithfulness response could not be parsed as JSON.")
                    for _ in claims
                ]

            results = []
            for claim in claims:
                item = by_id.get(claim.id)
                if item is None:
                    results.append(MethodScore(method=name, score=None, error="Missing from RAGAS response."))
                    continue
                verdict = item.get("verdict")
                score = 1.0 if verdict == "yes" else 0.0 if verdict == "no" else None
                results.append(
                    MethodScore(
                        method=name, score=score, detail=item.get("reason"),
                        error=None if score is not None else f"Unrecognized verdict: {verdict!r}",
                    )
                )
            return results

    return _RagasFaithfulness()
