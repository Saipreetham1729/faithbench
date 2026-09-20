"""
Self-consistency: re-answers the same question N times independently (at a
higher temperature), then scores each original claim by its average lexical
overlap against those independent samples. Needs no evidence at all --
grounded in the model's own sampling variance instead, which is why it's
worth comparing against the evidence-based methods rather than folded in
with them.

Unlike the other LLM-based methods, self-consistency needs a `generate_fn`
that produces one text completion for one prompt, plus the original
`question` -- both bound at construction time, since a self-consistency
scorer is inherently tied to the question that produced the answer:

    def my_generate(prompt: str) -> str:
        return your_client.complete(prompt)

    from faithbench.methods import self_consistency
    sc = self_consistency.make_scorer(my_generate, question="How does RAG reduce hallucination?")
    checker = FaithfulnessChecker(methods=["rouge_l", sc])

The `evidence` argument to score_claims is used only to build the prompt
context for re-generation (its chunk texts are joined) -- construct a fresh
scorer per question rather than reusing one across different questions.
"""
from typing import Callable, List

from faithbench.methods import rouge_l as _rouge_l
from faithbench.types import Claim, EvidenceChunk, MethodScore

name = "self_consistency"


def make_scorer(
    generate_fn: Callable[[str], str],
    question: str,
    n_samples: int = 3,
):
    """generate_fn(prompt) -> str. See module docstring."""

    class _SelfConsistency:
        name = "self_consistency"

        def score_claims(self, claims: List[Claim], evidence: List[EvidenceChunk]) -> List[MethodScore]:
            if not claims:
                return []

            context = "\n\n".join(chunk.text for chunk in evidence)
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

            samples = []
            errors = []
            for i in range(n_samples):
                try:
                    samples.append(generate_fn(prompt))
                except Exception as exc:  # noqa: BLE001 -- one failed sample shouldn't sink the whole metric
                    errors.append(str(exc))

            if not samples:
                err = f"All {n_samples} samples failed: {'; '.join(errors)}" if errors else "No samples produced."
                return [MethodScore(method=name, score=None, error=err) for _ in claims]

            # Average each claim's ROUGE-L overlap against every independent sample.
            totals = {claim.id: 0.0 for claim in claims}
            for i, sample_text in enumerate(samples):
                pseudo_evidence = [EvidenceChunk(id=f"sample_{i}", text=sample_text)]
                for claim, scored in zip(claims, _rouge_l.score_claims(claims, pseudo_evidence)):
                    totals[claim.id] += scored.score or 0.0

            return [
                MethodScore(
                    method=name, score=round(totals[claim.id] / len(samples), 4),
                    detail=f"{len(samples)}/{n_samples} samples used",
                )
                for claim in claims
            ]

    return _SelfConsistency()
