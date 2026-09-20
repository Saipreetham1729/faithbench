"""
FaithfulnessChecker: extracts claims from an answer and scores each one with
every registered method.

    from faithbench import FaithfulnessChecker

    checker = FaithfulnessChecker(methods=["rouge_l"])  # zero extra deps
    result = checker.check(answer="...", context=["chunk 1 text", "chunk 2 text"])
    result.overall_score
    result.claims[0].status()

`methods` accepts:
  - "rouge_l" or "nli" (strings) -- the built-in defaults, by name
  - a method object/module built via a factory, e.g. nli.make_scorer("other/model"),
    llm_judge.make_scorer(my_llm_fn), self_consistency.make_scorer(my_generate_fn, question=...)
  - any custom object with a `.name` and a matching `.score_claims(claims, evidence)`

A method that fails to run (missing dependency, network error, bad API
response) degrades to score=None with an explanation for every claim it
touched -- other methods keep running. Nothing is ever silently replaced
with a guessed number.
"""
from typing import List, Union

from faithbench.claims import extract_claims
from faithbench.methods import nli, rouge_l
from faithbench.methods.base import ScoringMethod
from faithbench.types import CheckResult, ClaimResult, EvidenceInput, MethodScore, normalize_evidence

_BUILTIN_METHODS = {
    "rouge_l": rouge_l,
    "nli": nli.default,
    "nli_entailment": nli.default,
}

MethodInput = Union[str, ScoringMethod]


class FaithfulnessChecker:
    def __init__(self, methods: List[MethodInput] = ("rouge_l",)):
        self._methods: List[ScoringMethod] = [self._resolve(m) for m in methods]

    @staticmethod
    def _resolve(method: MethodInput) -> ScoringMethod:
        if isinstance(method, str):
            if method not in _BUILTIN_METHODS:
                raise ValueError(
                    f"Unknown method name {method!r}. Built-in names are "
                    f"{sorted(_BUILTIN_METHODS)}, or pass a method object directly "
                    f"(e.g. from faithbench.methods.llm_judge.make_scorer(...))."
                )
            return _BUILTIN_METHODS[method]
        return method

    def check(self, answer: str, context: List[EvidenceInput]) -> CheckResult:
        claims = extract_claims(answer)
        evidence = normalize_evidence(context)

        # claim_id -> {method_name -> MethodScore}
        per_claim_scores = {claim.id: {} for claim in claims}
        for method in self._methods:
            scores = method.score_claims(claims, evidence)
            for claim, score in zip(claims, scores):
                per_claim_scores[claim.id][method.name] = score

        claim_results = [ClaimResult(claim=c, scores=per_claim_scores[c.id]) for c in claims]
        return CheckResult(answer=answer, claims=claim_results)
