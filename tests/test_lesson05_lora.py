import pytest

pytest.importorskip("peft")

from lesson05.lora_finetune import format_example


def test_format_example_shape() -> None:
    text = format_example("What is LoRA?", "A parameter-efficient tuning method.")
    assert text.startswith("Instruction:")
    assert "\nResponse:" in text
