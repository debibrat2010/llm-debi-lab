# Lesson 5: LoRA Fine-Tuning

## Why LoRA

- Full fine-tuning updates all model weights (expensive in memory and time).
- **LoRA** freezes base weights and trains small adapter matrices.
- You get task/style adaptation with much lower compute cost.

## Mental Model

- Base model = broad language ability.
- LoRA adapter = lightweight "specialization patch".
- At inference time, base + adapter are combined.

## Practical Workflow

1. Prepare small instruction dataset (`data/finetune/lora_tiny_train.jsonl`).
2. Train adapter (`python -m lesson05.lora_finetune`).
3. Compare base vs adapter outputs (`python -m lesson05.compare_base_vs_lora`).
4. In the local app, enable **use LoRA adapter** and run the same prompt as without it (`Generate` or `RAG Answer`). Check JSON field `lora_active` when debug is on.
5. Evaluate on fixed prompts and keep notes on regressions.

Adapter path: `data/finetune/artifacts/lesson05-lora` or override with env `LLM_DEBI_LORA_PATH`.

## Important Caveats (Mac local training)

- Tiny datasets overfit quickly; useful for learning workflow, not production quality.
- Use deterministic comparison prompts (`do_sample=False`) when measuring gains.
- If `mps` OOM occurs, reduce max length, epochs, or batch size.
- Keep the base model small until workflow is stable.
