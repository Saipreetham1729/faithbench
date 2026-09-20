from faithbench.claims import extract_claims


def test_splits_on_sentence_boundaries():
    claims = extract_claims("First sentence here. Second sentence here.")
    assert len(claims) == 2
    assert claims[0].text == "First sentence here."


def test_skips_short_fragments():
    claims = extract_claims("Yes. This is a longer real sentence here.")
    assert len(claims) == 1


def test_empty_input_returns_empty_list():
    assert extract_claims("") == []
    assert extract_claims("   ") == []


def test_claim_ids_are_sequential():
    claims = extract_claims("One sentence here. Another sentence here. A third one here.")
    assert [c.id for c in claims] == ["claim_0", "claim_1", "claim_2"]
