import json

from faithbench import FaithfulnessChecker
from faithbench.methods import llm_judge, ragas, self_consistency
from faithbench.types import EvidenceChunk


def test_check_returns_one_claim_result_per_sentence():
    checker = FaithfulnessChecker(methods=["rouge_l"])
    result = checker.check(
        answer="RAG grounds answers in evidence. Hallucination is still possible.",
        context=["RAG grounds generated answers in retrieved evidence."],
    )
    assert len(result.claims) == 2
    assert all("rouge_l" in c.scores for c in result.claims)


def test_accepts_plain_strings_or_evidence_chunk_objects():
    checker = FaithfulnessChecker(methods=["rouge_l"])
    r1 = checker.check(answer="A short test claim here.", context=["A short test claim here."])
    r2 = checker.check(
        answer="A short test claim here.",
        context=[EvidenceChunk(text="A short test claim here.", id="c1", page_number=3)],
    )
    assert r1.claims[0].aggregate_score == r2.claims[0].aggregate_score


def test_no_evidence_scores_zero_not_none():
    checker = FaithfulnessChecker(methods=["rouge_l"])
    result = checker.check(answer="A claim with no evidence at all.", context=[])
    assert result.claims[0].scores["rouge_l"].score == 0.0


def test_empty_answer_returns_no_claims():
    checker = FaithfulnessChecker(methods=["rouge_l"])
    result = checker.check(answer="", context=["some evidence"])
    assert result.claims == []
    assert result.overall_score is None


def test_unknown_method_name_raises_with_helpful_message():
    try:
        FaithfulnessChecker(methods=["not_a_real_method"])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "not_a_real_method" in str(exc)
        assert "rouge_l" in str(exc)


def test_multiple_methods_all_appear_in_scores():
    def fake_llm(system, user):
        return json.dumps([{"claim_id": "claim_0", "verdict": "supported", "explanation": "ok"}])

    checker = FaithfulnessChecker(methods=["rouge_l", llm_judge.make_scorer(fake_llm)])
    result = checker.check(answer="One single claim here.", context=["One single claim here."])
    assert set(result.claims[0].scores.keys()) == {"rouge_l", "llm_judge"}


def test_status_thresholds():
    checker = FaithfulnessChecker(methods=["rouge_l"])
    result = checker.check(
        answer="This exact sentence appears in the evidence verbatim.",
        context=["This exact sentence appears in the evidence verbatim."],
    )
    assert result.claims[0].status() == "supported"


def test_to_dict_is_json_serializable():
    checker = FaithfulnessChecker(methods=["rouge_l"])
    result = checker.check(answer="A claim to serialize.", context=["A claim to serialize."])
    json.dumps(result.to_dict())  # raises if not serializable


def test_llm_judge_unparseable_response_gives_none_not_a_guess():
    def broken_llm(system, user):
        return "not json at all"

    checker = FaithfulnessChecker(methods=[llm_judge.make_scorer(broken_llm)])
    result = checker.check(answer="A claim here to check.", context=["some evidence"])
    assert result.claims[0].scores["llm_judge"].score is None
    assert result.claims[0].scores["llm_judge"].error is not None


def test_llm_judge_fn_exception_is_caught_not_raised():
    def failing_llm(system, user):
        raise RuntimeError("network down")

    checker = FaithfulnessChecker(methods=[llm_judge.make_scorer(failing_llm)])
    result = checker.check(answer="A claim here to check.", context=["some evidence"])
    assert result.claims[0].scores["llm_judge"].score is None
    assert "network down" in result.claims[0].scores["llm_judge"].error


def test_ragas_binary_verdicts():
    def fake_llm(system, user):
        return json.dumps([{"claim_id": "claim_0", "verdict": "yes", "reason": "ok"}])

    checker = FaithfulnessChecker(methods=[ragas.make_scorer(fake_llm)])
    result = checker.check(answer="A single supported claim.", context=["evidence"])
    assert result.claims[0].scores["ragas_faithfulness"].score == 1.0


def test_self_consistency_scores_against_independent_samples():
    def fake_generate(prompt):
        return "The sky is blue during the day."

    sc = self_consistency.make_scorer(fake_generate, question="What color is the sky?", n_samples=3)
    checker = FaithfulnessChecker(methods=[sc])
    result = checker.check(answer="The sky is blue during the day.", context=["irrelevant"])
    assert result.claims[0].scores["self_consistency"].score > 0.9


def test_self_consistency_all_samples_failing_gives_none():
    def failing_generate(prompt):
        raise RuntimeError("rate limited")

    sc = self_consistency.make_scorer(failing_generate, question="Q?", n_samples=2)
    checker = FaithfulnessChecker(methods=[sc])
    result = checker.check(answer="A claim here to check.", context=["evidence"])
    assert result.claims[0].scores["self_consistency"].score is None
