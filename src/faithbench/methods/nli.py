"""
NLI-based entailment: a DeBERTa cross-encoder scores whether each evidence
chunk (premise) entails each claim (hypothesis), keeping the best match per
claim.

Requires the `nli` extra: pip install faithbench[nli]
Downloads a ~140MB model from Hugging Face on first use.
"""
from functools import lru_cache
from typing import List

from faithbench.types import Claim, EvidenceChunk, MethodScore

try:
    import numpy as np
    from sentence_transformers import CrossEncoder
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

name = "nli_entailment"

# Label order for cross-encoder/nli-deberta-v3-xsmall (the default model):
# [contradiction, entailment, neutral]. Pass a different model_name if you
# use a different NLI cross-encoder with a different label order.
_ENTAILMENT_INDEX = 1
_DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-xsmall"


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    return CrossEncoder(model_name)


def _softmax(logits):
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


def make_scorer(model_name: str = _DEFAULT_MODEL):
    """Returns a score_claims function bound to a specific NLI model, so you
    can swap models without touching the rest of your code:
    `checker = FaithfulnessChecker(methods=[nli.make_scorer("other/model")])`
    """

    class _NliMethod:
        name = "nli_entailment"

        def score_claims(self, claims: List[Claim], evidence: List[EvidenceChunk]) -> List[MethodScore]:
            if not _AVAILABLE:
                return [
                    MethodScore(
                        method=name, score=None,
                        error="sentence-transformers is not installed. Install with: pip install faithbench[nli]",
                    )
                    for _ in claims
                ]
            if not evidence:
                return [MethodScore(method=name, score=0.0, detail="No evidence provided.") for _ in claims]

            try:
                model = _get_model(model_name)
                results = []
                for claim in claims:
                    pairs = [(chunk.text, claim.text) for chunk in evidence]
                    logits = model.predict(pairs)
                    probs = [float(_softmax(np.array(l))[_ENTAILMENT_INDEX]) for l in logits]
                    best_idx = max(range(len(probs)), key=lambda i: probs[i])
                    results.append(
                        MethodScore(
                            method=name, score=round(probs[best_idx], 4),
                            detail=f"best match: {evidence[best_idx].id}",
                        )
                    )
                return results
            except Exception as exc:  # noqa: BLE001 -- model download/inference can fail many ways
                return [MethodScore(method=name, score=None, error=str(exc)) for _ in claims]

    return _NliMethod()


default = make_scorer()
