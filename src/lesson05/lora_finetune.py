"""Lesson 5: tiny LoRA fine-tuning on local instruction data."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from llm_debi_lab.device import pick_device
from llm_debi_lab.generation import (
    ensure_pad_token,
    format_prompt_for_causal,
    move_model_to_device,
)
from llm_debi_lab.settings import DEFAULT_MODEL_NAME

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "finetune" / "lora_tiny_train.jsonl"
OUTPUT_DIR = ROOT / "data" / "finetune" / "artifacts" / "lesson05-lora"


def format_train_pair(tokenizer: AutoTokenizer, instruction: str, response: str) -> tuple[str, str]:
    """Return (inference prompt, full training text).

    The prompt is exactly what `format_prompt_for_causal` sends at generation
    time, so the adapter is trained on the same token prefix used in serving.
    """
    prompt = format_prompt_for_causal(tokenizer, instruction)
    if getattr(tokenizer, "chat_template", None):
        full = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        if isinstance(full, str) and full.startswith(prompt):
            return prompt, full
        eos = tokenizer.eos_token or ""
        return prompt, f"{prompt}{response}{eos}"
    return prompt, f"{prompt}{response}"


def mask_prompt_labels(prompt_ids: list[int], input_ids: list[int]) -> list[int]:
    """Ignore prompt tokens in the loss. Keep only the response tokens."""
    labels = list(input_ids)
    if not prompt_ids or not labels:
        return labels
    if len(labels) >= len(prompt_ids) and labels[: len(prompt_ids)] == list(prompt_ids):
        mask_len = len(prompt_ids)
    else:
        mask_len = 0
        for prompt_id, input_id in zip(prompt_ids, labels):
            if prompt_id != input_id:
                break
            mask_len += 1
    if mask_len >= len(labels):
        return [-100] * len(labels)
    labels[:mask_len] = [-100] * mask_len
    return labels


def encode_supervised_example(
    tokenizer: AutoTokenizer,
    instruction: str,
    response: str,
    max_length: int = 256,
) -> dict[str, list[int]]:
    prompt, full = format_train_pair(tokenizer, instruction, response)
    prompt_ids = list(
        tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
    )
    full_enc = tokenizer(full, truncation=True, max_length=max_length)
    input_ids = list(full_enc["input_ids"])
    attention_mask = list(full_enc["attention_mask"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": mask_prompt_labels(prompt_ids, input_ids),
    }


def load_train_records(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Missing training file: {path}")
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rows.append(
            {
                "instruction": str(obj["instruction"]),
                "response": str(obj["response"]),
            }
        )
    if not rows:
        raise RuntimeError(f"No rows found in: {path}")
    return rows


def tokenize_dataset(ds: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        encoded = [
            encode_supervised_example(tokenizer, instruction, response)
            for instruction, response in zip(batch["instruction"], batch["response"])
        ]
        return {
            "input_ids": [row["input_ids"] for row in encoded],
            "attention_mask": [row["attention_mask"] for row in encoded],
            "labels": [row["labels"] for row in encoded],
        }

    return ds.map(_tokenize, batched=True, remove_columns=ds.column_names)


def main() -> None:
    requested = pick_device()
    print(f"Requested device: {requested}")
    print(f"Base model: {DEFAULT_MODEL_NAME}")

    rows = load_train_records(TRAIN_PATH)
    print(f"Training rows: {len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    ensure_pad_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME)
    model, resolved = move_model_to_device(model, requested)
    print(f"Using device: {resolved}")
    template = "chat template" if getattr(tokenizer, "chat_template", None) else "raw prompt"
    print(f"Train/infer format: {template}")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = Dataset.from_list(rows)
    tokenized = tokenize_dataset(ds, tokenizer)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,
        num_train_epochs=8,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=20,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        fp16=False,
    )
    # Seq2Seq collator keeps the -100 prompt mask. The language-modeling
    # collator would replace labels with a copy of input_ids.
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nSaved LoRA adapter and tokenizer to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
