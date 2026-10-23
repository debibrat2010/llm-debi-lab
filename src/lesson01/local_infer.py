from llm_debi_lab.device import pick_device
from llm_debi_lab.generation import (
    format_prompt_for_causal,
    generate_continuation,
    load_causal_lm,
)
from llm_debi_lab.settings import DEFAULT_MODEL_NAME


def main() -> None:
    requested = pick_device()
    print(f"Requested device: {requested}")

    model, tokenizer, device = load_causal_lm(DEFAULT_MODEL_NAME, requested)
    print(f"Using device: {device}")

    user_prompt = "Explain what a token is in one short paragraph:"
    formatted = format_prompt_for_causal(tokenizer, user_prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    print("\nToken IDs for formatted prompt (first 40):")
    ids = inputs["input_ids"][0].tolist()
    print(ids[:40], "..." if len(ids) > 40 else "")

    output_text = generate_continuation(
        model,
        tokenizer,
        user_prompt,
        device,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    print("\nGenerated continuation (new tokens only):\n")
    print(output_text)


if __name__ == "__main__":
    main()
