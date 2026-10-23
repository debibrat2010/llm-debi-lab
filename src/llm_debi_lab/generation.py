from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token


def format_prompt_for_causal(tokenizer: AutoTokenizer, user_text: str) -> str:
    """Use chat template when present (instruct models); otherwise raw text."""
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_text


def generate_continuation_detailed(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    user_prompt: str,
    device: str,
    *,
    max_new_tokens: int = 128,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.12,
) -> tuple[str, int, float]:
    ensure_pad_token(tokenizer)
    text = format_prompt_for_causal(tokenizer, user_prompt)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = int(inputs["input_ids"].shape[1])

    gen_kwargs: dict = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = max(float(temperature), 1e-5)
        gen_kwargs["top_p"] = float(top_p)

    if repetition_penalty and repetition_penalty > 1.0:
        gen_kwargs["repetition_penalty"] = float(repetition_penalty)

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)
    elapsed = time.perf_counter() - t0

    new_len = int(output_ids.shape[1]) - input_len
    new_ids = output_ids[0, input_len:]
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return decoded, new_len, elapsed


def generate_continuation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    user_prompt: str,
    device: str,
    *,
    max_new_tokens: int = 128,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.12,
) -> str:
    text, _, _ = generate_continuation_detailed(
        model,
        tokenizer,
        user_prompt,
        device,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return text


def load_causal_lm(
    model_name: str, device: str
) -> tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ensure_pad_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    try:
        model = model.to(device)
        resolved = device
    except Exception:
        if device == "mps":
            model = model.to("cpu")
            resolved = "cpu"
        else:
            raise
    model.eval()
    return model, tokenizer, resolved
