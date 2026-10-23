import pytest

pytest.importorskip("peft")

from llm_debi_lab.generation import format_prompt_for_causal, move_model_to_device
from lesson05.lora_finetune import (
    encode_supervised_example,
    format_train_pair,
    mask_prompt_labels,
)


class _ChatTok:
    chat_template = "chat"
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        user = messages[0]["content"]
        prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        if add_generation_prompt:
            return prompt
        assistant = messages[1]["content"]
        return f"{prompt}{assistant}<|im_end|>\n"

    def __call__(self, text, truncation=True, max_length=256, **kwargs):
        ids = [ord(ch) for ch in text][:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class _RawTok:
    chat_template = None

    def __call__(self, text, truncation=True, max_length=256, **kwargs):
        ids = [ord(ch) for ch in text][:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class _MpsFallback:
    def __init__(self) -> None:
        self.device = None

    def to(self, device: str):
        if device == "mps":
            raise RuntimeError("mps placement failed")
        self.device = device
        return self


def test_train_pair_uses_the_inference_chat_prompt() -> None:
    tok = _ChatTok()
    instruction = "What is LoRA?"
    prompt, full = format_train_pair(tok, instruction, "A small adapter.")
    assert prompt == format_prompt_for_causal(tok, instruction)
    assert full.startswith(prompt)
    assert "A small adapter." in full


def test_supervised_labels_ignore_the_prompt() -> None:
    tok = _ChatTok()
    instruction = "What is LoRA?"
    response = "A small adapter."
    prompt = format_prompt_for_causal(tok, instruction)
    item = encode_supervised_example(tok, instruction, response)
    prompt_len = len(tok(prompt)["input_ids"])
    assert item["input_ids"][:prompt_len] == tok(prompt)["input_ids"]
    assert item["labels"][:prompt_len] == [-100] * prompt_len
    tail = "".join(chr(i) for i in item["input_ids"][prompt_len:])
    assert response in tail
    assert any(token != -100 for token in item["labels"][prompt_len:])


def test_supervised_example_without_template_continues_the_raw_prompt() -> None:
    tok = _RawTok()
    item = encode_supervised_example(tok, "Hi", "There")
    assert "".join(chr(i) for i in item["input_ids"]) == "HiThere"
    assert item["labels"][:2] == [-100, -100]
    assert item["labels"][2:] == [ord(ch) for ch in "There"]


def test_mask_prompt_labels_keeps_response_after_a_boundary_mismatch() -> None:
    assert mask_prompt_labels([1, 2, 9], [1, 2, 3, 4]) == [-100, -100, 3, 4]


def test_mask_prompt_labels_masks_everything_when_response_was_truncated() -> None:
    assert mask_prompt_labels([1, 2, 3, 4], [1, 2]) == [-100, -100]


def test_move_model_to_device_falls_back_from_mps() -> None:
    model, resolved = move_model_to_device(_MpsFallback(), "mps")
    assert resolved == "cpu"
    assert model.device == "cpu"
