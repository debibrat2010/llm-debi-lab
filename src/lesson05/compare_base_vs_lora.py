"""Compare base model output against trained LoRA adapter output."""

from __future__ import annotations

from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_debi_lab.device import pick_device
from llm_debi_lab.generation import generate_continuation
from llm_debi_lab.settings import DEFAULT_MODEL_NAME

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = ROOT / "data" / "finetune" / "artifacts" / "lesson05-lora"


def main() -> None:
    device = pick_device()
    print(f"Device: {device}")
    print(f"Base model: {DEFAULT_MODEL_NAME}")
    print(f"Adapter dir: {ADAPTER_DIR}")

    prompt = (
        "Explain tokenization for a beginner in 2 concise sentences. "
        "Keep wording simple."
    )

    base_tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME).to(device)
    base_text = generate_continuation(
        base_model,
        base_tok,
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

    lora_tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    lora_base = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME).to(device)
    lora_model = PeftModel.from_pretrained(lora_base, str(ADAPTER_DIR))
    lora_text = generate_continuation(
        lora_model,
        lora_tok,
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
