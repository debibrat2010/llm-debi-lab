from lesson02.prompt_fundamentals import build_prompts


def test_build_prompts_has_weak_and_strong() -> None:
    prompts = build_prompts("tokenization")
    assert set(prompts.keys()) == {"weak", "strong"}
    assert prompts["weak"] == "Write about tokenization."
    assert "Exactly 3 bullet points" in prompts["strong"]
