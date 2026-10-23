"""Lesson 5: tiny LoRA fine-tuning on local instruction data."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from llm_debi_lab.device import pick_device
from llm_debi_lab.generation import ensure_pad_token
from llm_debi_lab.settings import DEFAULT_MODEL_NAME

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "finetune" / "lora_tiny_train.jsonl"
OUTPUT_DIR = ROOT / "data" / "finetune" / "artifacts" / "lesson05-lora"


def format_example(instruction: str, response: str) -> str:
    return f"Instruction: {instruction}\nResponse: {response}"


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
                "text": format_example(str(obj["instruction"]), str(obj["response"])),
            }
        )
    if not rows:
        raise RuntimeError(f"No rows found in: {path}")
    return rows


def tokenize_dataset(ds: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        tok = tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
            padding=False,
        )
        tok["labels"] = [ids[:] for ids in tok["input_ids"]]
        return tok

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
    model = model.to(requested)

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
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
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
