from lesson03.inference_patterns import tokens_per_second


def test_tokens_per_second_basic() -> None:
    assert tokens_per_second(10, 2.0) == 5.0


def test_tokens_per_second_zero_elapsed() -> None:
    assert tokens_per_second(10, 0.0) == 0.0
