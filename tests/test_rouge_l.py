from faithbench.claims import extract_claims
from faithbench.methods import rouge_l
from faithbench.types import EvidenceChunk


def test_exact_match_scores_high():
    claims = extract_claims("The sky is blue during the day.")
    evidence = [EvidenceChunk(text="The sky is blue during the day.", id="e1")]
    result = rouge_l.score_claims(claims, evidence)
    assert result[0].score > 0.95


def test_unrelated_text_scores_low():
    claims = extract_claims("The sky is blue during the day.")
    evidence = [EvidenceChunk(text="Bananas are a good source of potassium.", id="e1")]
    result = rouge_l.score_claims(claims, evidence)
    assert result[0].score < 0.3


def test_no_evidence_scores_zero():
    claims = extract_claims("A claim with no evidence to check.")
    result = rouge_l.score_claims(claims, [])
    assert result[0].score == 0.0


def test_keeps_best_matching_chunk():
    claims = extract_claims("The sky is blue during the day.")
    evidence = [
        EvidenceChunk(text="Bananas are yellow.", id="unrelated"),
        EvidenceChunk(text="The sky is blue during the day.", id="exact_match"),
    ]
    result = rouge_l.score_claims(claims, evidence)
    assert result[0].detail == "best match: exact_match"
