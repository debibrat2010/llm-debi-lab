from llm_debi_lab.device import pick_device
from llm_debi_lab.generation import generate_continuation, load_causal_lm
from llm_debi_lab.settings import DEFAULT_MODEL_NAME


def build_prompts(topic: str) -> dict[str, str]:
    weak_prompt = f"Write about {topic}."
    strong_prompt = (
        "You are a concise ML tutor.\n"
        f"Task: Explain {topic} for a beginner.\n"
        "Requirements:\n"
        "- Exactly 3 bullet points\n"
        "- Include one real-world example\n"
        "- Keep total response under 90 words\n"
        "Output format:\n"
        "- Bullet list only"
    )
    return {"weak": weak_prompt, "strong": strong_prompt}


def main() -> None:
    requested = pick_device()
    print(f"Requested device: {requested}")

    model, tokenizer, device = load_causal_lm(DEFAULT_MODEL_NAME, requested)
    print(f"Using device: {device}")
    print(f"Using model: {DEFAULT_MODEL_NAME}")

    topic = "prompt engineering for local LLMs"
    prompts = build_prompts(topic)

    print("\n=== Prompt A (weak) ===\n")
    print(prompts["weak"])
    weak_output = generate_continuation(
        model,
        tokenizer,
        prompts["weak"],
        device,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.6,
        top_p=0.92,
        repetition_penalty=1.15,
    )
    print("\n=== Output A (new tokens only) ===\n")
    print(weak_output)

    print("\n=== Prompt B (strong/structured) ===\n")
    print(prompts["strong"])
    strong_output = generate_continuation(
        model,
        tokenizer,
        prompts["strong"],
        device,
        max_new_tokens=160,
        do_sample=True,
        temperature=0.35,
        top_p=0.9,
        repetition_penalty=1.18,
    )
    print("\n=== Output B (new tokens only) ===\n")
    print(strong_output)

    print(
        "\nObservation: instruct-tuned models + structured prompts usually follow "
        "format constraints better than tiny base models like distilgpt2."
    )


if __name__ == "__main__":
    main()
