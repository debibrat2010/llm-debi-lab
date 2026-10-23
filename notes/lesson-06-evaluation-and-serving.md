# Lesson 6: Evaluation and Serving

## Why this lesson exists

- **Prompting** and **RAG** change behavior, but you need a **repeatable way to know if you improved**.
- **Serving** (FastAPI + `uvicorn`) turns your model into a small local product you can call from a browser or scripts.

## Evaluation basics

- **Fixed prompt set:** same questions every run so you can compare before/after.
- **Cheap checks:** keyword presence, max length, forbidden phrases, or simple structure.
- **Regressions:** a change that helps one task can hurt another; log both.

## Scripts in this repo

- `python -m lesson06.generation_eval` — runs `data/eval/generation_eval_set.jsonl` on **base** and **LoRA** (if present), writes CSV/JSONL under `data/eval/reports/`.
- `python -m lesson06.local_api_smoke` — with `uvicorn` running, checks `/api/health` and one `/api/generate` call (stdlib only).

## Serving checklist

- Start server: `python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8765`
- Check: `GET /api/health` (model, device, RAG, LoRA status)
- Version your eval sets and report files when you change models or adapters.

## What to learn next

- Add your own **golden** prompts to `generation_eval_set.jsonl`.
- Treat RAG eval (Lesson 4.2) and generation eval (Lesson 6) as two layers: retrieval quality vs answer quality.
