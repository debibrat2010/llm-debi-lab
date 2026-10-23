from lesson04.rag_eval import keyword_hit


def test_keyword_hit_true() -> None:
    answer = "Carry a water bottle and drink regularly."
    assert keyword_hit(answer, ["water bottle", "dehydration"]) is True


def test_keyword_hit_false() -> None:
    answer = "This sentence has no relevant phrase."
    assert keyword_hit(answer, ["hydration", "urine"]) is False
