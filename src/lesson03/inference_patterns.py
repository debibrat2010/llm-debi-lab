"""Lesson 3: sampling vs greedy, repetition control, and simple throughput timing."""

from llm_debi_lab.device import pick_device
from llm_debi_lab.generation import generate_continuation_detailed, load_causal_lm
from llm_debi_lab.settings import DEFAULT_MODEL_NAME


def tokens_per_second(new_tokens: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 0.0
    return new_tokens / elapsed_s


def main() -> None:
    requested = pick_device()
    model, tokenizer, device = load_causal_lm(DEFAULT_MODEL_NAME, requested)
    print(f"Device: {device} | Model: {DEFAULT_MODEL_NAME}\n")

    prompt = "List three practical tips for running LLMs locally on a laptop."

    print("=== Greedy decoding (do_sample=False) ===")
    text_g, n_g, t_g = generate_continuation_detailed(
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=64,
        do_sample=False,
        repetition_penalty=1.0,
    )
    print(text_g)
    print(
        f"(new_tokens={n_g}, time={t_g:.3f}s, ~{tokens_per_second(n_g, t_g):.1f} tok/s)\n"
    )

    print("=== Sampling (temperature=0.8) ===")
    text_s, n_s, t_s = generate_continuation_detailed(
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.08,
    )
    print(text_s)
    print(
        f"(new_tokens={n_s}, time={t_s:.3f}s, ~{tokens_per_second(n_s, t_s):.1f} tok/s)\n"
    )

    print("=== Same sampling run again (stochastic: expect variation) ===")
    text_s2, n_s2, t_s2 = generate_continuation_detailed(
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.08,
    )
    print(text_s2)
    print(
        f"(new_tokens={n_s2}, time={t_s2:.3f}s, ~{tokens_per_second(n_s2, t_s2):.1f} tok/s)\n"
    )

    print(
        "Takeaway: greedy is repeatable; sampling explores. "
        "Throughput (tok/s) is a coarse local benchmark—use the same settings when comparing."
    )


if __name__ == "__main__":
    main()
