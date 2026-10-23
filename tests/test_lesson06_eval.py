from lesson06.generation_eval import keyword_hit


def test_keyword_hit_any_match() -> None:
    assert keyword_hit("Use fixed prompts and compare runs.", ["fixed", "missing"]) is True


def test_keyword_hit_empty_keywords_always_true() -> None:
    assert keyword_hit("anything", []) is True
