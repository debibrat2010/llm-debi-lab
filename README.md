# llm-debi-lab

Hands-on local LLM learning lab for Apple Silicon macOS.

## Lesson Path

- Lesson 1: What is an LLM? (theory + token inspection)
- Lesson 2: Prompting fundamentals
- Lesson 3: Local inference patterns
- Lesson 4: RAG basics
- Lesson 5: LoRA fine-tuning
- Lesson 6: Evaluation and serving

## Quick Start

1. Create and activate environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   This installs the project in editable mode (`-e .`) so imports like `lesson01` and `llm_debi_lab` work from anywhere in the venv.

2. Run lessons (from repo root, venv active):

   ```bash
   python -m lesson01.local_infer
   python -m lesson02.prompt_fundamentals
   python -m lesson03.inference_patterns
   python -m lesson04.rag_basics
   python -m lesson04.rag_eval
   python -m lesson05.lora_finetune
   python -m lesson05.compare_base_vs_lora
   python -m lesson06.generation_eval
   python -m lesson06.local_api_smoke   # requires uvicorn already running
   ```

3. Tests:

   ```bash
   pytest -q
   ```

4. Local web playground (loads generation + RAG at server startup):

   ```bash
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8765
   ```

   After Lesson 5 training, check **`GET /api/health`** for `lora_loaded`. In the UI, use **use LoRA adapter** to compare base vs LoRA on the same prompt.

   Open `http://127.0.0.1:8765` in a browser.
   - Generate API: `POST /api/generate`
   - RAG API: `POST /api/rag` with JSON body
     `{ "prompt": "...", "top_k": 3, "max_new_tokens": 128, "temperature": 0.7, "top_p": 0.9, "do_sample": true }`

## Model selection

- Default model is **`HuggingFaceTB/SmolLM2-360M-Instruct`** (small, instruct-tuned, good on Apple Silicon).
- Override with env: `export LLM_DEBI_MODEL="your/model-id"` then run lessons or `uvicorn` again.

## Project Structure

- `src/` packages: `lesson01` … `lesson06`, shared `llm_debi_lab/`
- `app/` FastAPI localhost playground
- `data/kb/` local text files used for Lesson 4.1 RAG app endpoint
- `data/eval/` eval sets (RAG + generation) and generated reports
- `data/finetune/` Lesson 5 tiny LoRA dataset + adapter artifacts
- `notes/` notebook-ready theory points
- `tests/` lightweight checks
- `pyproject.toml` project metadata and dependencies

## Git Setup (your personal repo)

```bash
git init
git add .
git commit -m "Initialize llm-debi-lab with lesson 1"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Notes for MacBook M5 (no NVIDIA)

- `mps` (Apple Metal) is used automatically when available.
- If moving weights to `mps` fails, `load_causal_lm` falls back to CPU.
- Start with small models to keep memory usage smooth.
