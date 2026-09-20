"""
LLM-judge: asks a model, in one call, to verdict every claim as supported /
partially_supported / unsupported against the evidence.

This method needs an LLM, but faithbench doesn't ship an API client for one
-- you supply a plain function instead:

    def my_llm(system_prompt: str, user_prompt: str) -> str:
        response = your_client.chat.completions.create(...)
        return response.choices[0].message.content

    from faithbench.methods import llm_judge
    checker = FaithfulnessChecker(methods=[llm_judge.make_scorer(my_llm)])

This keeps the library provider-agnostic: OpenAI, Anthropic, a local model,
anything that can be wrapped in a (system, user) -> str function works.

Per the "never fabricate results" principle: if the model's response can't
be parsed as the expected JSON, every claim gets score=None with an error
explaining why -- never a guessed number.
"""
import json
from typing import Callable, List

from faithbench.types import Claim, EvidenceChunk, MethodScore

name = "llm_judge"

_VERDICT_SCORES = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0}

_SYSTEM_PROMPT = (
    "You are a strict fact-checking assistant. You will be given a list of numbered claims "
    "and a block of evidence text. For EACH claim, decide whether it is 'supported', "
    "'partially_supported', or 'unsupported' by the evidence -- based only on the evidence, "
    "not outside knowledge. Respond with ONLY a JSON array, one object per claim, in this "
    'exact format: [{"claim_id": "...", "verdict": "...", "explanation": "..."}]. '
    "No text outside the JSON array."
)


def _build_user_prompt(claims: List[Claim], evidence: List[EvidenceChunk]) -> str:
    claims_block = "\n".join(f"{c.id}: {c.text}" for c in claims)
    evidence_block = "\n\n".join(chunk.text for chunk in evidence)
    return f"Evidence:\n{evidence_block}\n\nClaims:\n{claims_block}\n\nRespond with the JSON array now."


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
    """llm_fn(system_prompt, user_prompt) -> str. See module docstring."""

    class _LlmJudge:
        name = "llm_judge"

        def score_claims(self, claims: List[Claim], evidence: List[EvidenceChunk]) -> List[MethodScore]:
            if not claims:
                return []
            if not evidence:
                return [
                    MethodScore(method=name, score=0.0, detail="unsupported", error=None)
                    for _ in claims
                ]

            try:
                raw = llm_fn(_SYSTEM_PROMPT, _build_user_prompt(claims, evidence))
            except Exception as exc:  # noqa: BLE001 -- the user's llm_fn can fail in any way (network, auth, etc.)
                return [MethodScore(method=name, score=None, error=str(exc)) for _ in claims]

            by_id = _parse_response(raw)
            if by_id is None:
                return [
                    MethodScore(method=name, score=None, error="LLM judge response could not be parsed as JSON.")
                    for _ in claims
                ]

            results = []
            for claim in claims:
                item = by_id.get(claim.id)
                if item is None:
                    results.append(MethodScore(method=name, score=None, error="Missing from LLM judge response."))
                    continue
                verdict = item.get("verdict")
                results.append(
                    MethodScore(
                        method=name, score=_VERDICT_SCORES.get(verdict),
                        detail=item.get("explanation"),
                        error=None if verdict in _VERDICT_SCORES else f"Unrecognized verdict: {verdict!r}",
                    )
                )
            return results

    return _LlmJudge()
