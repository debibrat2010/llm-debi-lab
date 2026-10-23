"""Compare base model output against trained LoRA adapter output."""

from __future__ import annotations

from pathlib import Path

from peft import PeftModel

from llm_debi_lab.device import pick_device
from llm_debi_lab.generation import generate_continuation, load_causal_lm
from llm_debi_lab.settings import DEFAULT_MODEL_NAME

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = ROOT / "data" / "finetune" / "artifacts" / "lesson05-lora"


def main() -> None:
    requested = pick_device()
    print(f"Requested device: {requested}")
    print(f"Base model: {DEFAULT_MODEL_NAME}")
    print(f"Adapter dir: {ADAPTER_DIR}")

    prompt = (
        "Explain tokenization for a beginner in 2 concise sentences. "
        "Keep wording simple."
    )

    model, tokenizer, device = load_causal_lm(DEFAULT_MODEL_NAME, requested)
    print(f"Using device: {device}")
    base_text = generate_continuation(
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=120,
        do_sample=False,
        temperature=0.2,
    )
    print("\n=== Base output ===\n")
    print(base_text)

    if not ADAPTER_DIR.exists():
        print("\nLoRA adapter not found. Run `python -m lesson05.lora_finetune` first.")
        return

    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()
    lora_text = generate_continuation(
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=120,
        do_sample=False,
        temperature=0.2,
    )
    print("\n=== LoRA output ===\n")
    print(lora_text)


if __name__ == "__main__":
    main()
