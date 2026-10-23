"""Minimal local web UI + JSON APIs for generation and RAG."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from llm_debi_lab.device import pick_device
from llm_debi_lab.generation import (
    generate_continuation_detailed,
    load_causal_lm,
)
from llm_debi_lab.rag import (
    EMBED_MODEL_NAME,
    build_rag_prompt,
    chunk_text,
    load_kb_texts,
    retrieve_top_k,
)
from llm_debi_lab.settings import DEFAULT_MODEL_NAME, LLM_DEBI_LORA_PATH

_state: dict[str, Any] = {}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = PROJECT_ROOT / "data" / "kb"
DEFAULT_LORA_ADAPTER_DIR = PROJECT_ROOT / "data" / "finetune" / "artifacts" / "lesson05-lora"


@asynccontextmanager
async def lifespan(app: FastAPI):
    requested = pick_device()
    try:
        model_base, tokenizer, device = load_causal_lm(DEFAULT_MODEL_NAME, requested)
    except Exception as exc:  # pragma: no cover - requires HF + weights
        _state["error"] = str(exc)
        yield
        return
    _state["model_base"] = model_base
    _state["tokenizer"] = tokenizer
    _state["device"] = device
    _state["model_name"] = DEFAULT_MODEL_NAME
    _state["model_lora"] = None
    _state["lora_status"] = "none"
    adapter_dir = Path(LLM_DEBI_LORA_PATH) if LLM_DEBI_LORA_PATH else DEFAULT_LORA_ADAPTER_DIR
    try:
        from peft import PeftModel  # noqa: PLC0415

        if adapter_dir.is_dir() and any(adapter_dir.iterdir()):
            base_for_lora = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME).to(
                device
            )
            _state["model_lora"] = PeftModel.from_pretrained(
                base_for_lora, str(adapter_dir)
            )
            _state["model_lora"].eval()
            _state["lora_status"] = f"ok:{adapter_dir}"
        else:
            _state["lora_status"] = f"missing:{adapter_dir}"
    except ImportError:
        _state["lora_status"] = "peft-not-installed"
    except Exception as exc:  # pragma: no cover
        _state["lora_status"] = f"error:{exc}"
    _state["rag_error"] = ""
    try:
        kb_docs = load_kb_texts(KB_DIR)
        chunks: list[str] = []
        for doc in kb_docs:
            chunks.extend(chunk_text(doc, chunk_size_words=65, overlap_words=15))
        if not chunks:
            raise RuntimeError(f"No .txt KB docs found in {KB_DIR}")
        emb_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
        emb_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device)
        emb_model.eval()
        _state["emb_tokenizer"] = emb_tokenizer
        _state["emb_model"] = emb_model
        _state["kb_chunks"] = chunks
    except Exception as exc:  # pragma: no cover - external model download/device issues
        _state["rag_error"] = str(exc)
    yield
    _state.clear()


app = FastAPI(title="llm-debi-lab", lifespan=lifespan)

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>llm-debi-lab</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 52rem; margin: 2rem auto; padding: 0 1rem; }
    label { display: block; margin-top: 0.75rem; font-weight: 600; }
    textarea { width: 100%; min-height: 6rem; }
    button { margin-top: 1rem; padding: 0.5rem 1rem; }
    pre { white-space: pre-wrap; background: #111; color: #eee; padding: 1rem; border-radius: 8px; }
    .muted { color: #666; font-size: 0.9rem; }
  </style>
</head>
<body>
  <h1>llm-debi-lab</h1>
  <p class="muted">Local-only playground. First load downloads the model from Hugging Face.</p>
  <label for="prompt">Prompt</label>
  <textarea id="prompt">Give one sentence of advice for learning LLMs locally.</textarea>
  <label for="max_new_tokens">max_new_tokens</label>
  <input id="max_new_tokens" type="number" value="128" min="1" max="512" />
  <label for="temperature">temperature</label>
  <input id="temperature" type="number" value="0.7" step="0.05" min="0" max="2" />
  <label for="top_p">top_p</label>
  <input id="top_p" type="number" value="0.9" step="0.05" min="0" max="1" />
  <label><input id="do_sample" type="checkbox" checked /> do_sample</label>
  <label for="top_k">RAG top_k</label>
  <input id="top_k" type="number" value="3" min="1" max="8" />
  <label><input id="show_debug" type="checkbox" /> show debug json</label>
  <label><input id="use_lora" type="checkbox" /> use LoRA adapter (Lesson 5)</label>
  <div>
    <button id="go" type="button">Generate</button>
    <button id="go-rag" type="button">RAG Answer</button>
  </div>
  <h2>Output</h2>
  <pre id="out">(not run yet)</pre>
  <script>
    async function callApi(path, body) {
      const res = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) return { error: data };
      return { data };
    }

    const out = document.getElementById("out");
    function baseBody() {
      return {
        prompt: document.getElementById("prompt").value,
        max_new_tokens: Number(document.getElementById("max_new_tokens").value),
        temperature: Number(document.getElementById("temperature").value),
        top_p: Number(document.getElementById("top_p").value),
        do_sample: document.getElementById("do_sample").checked,
        use_lora: document.getElementById("use_lora").checked,
      };
    }

    document.getElementById("go").onclick = async () => {
      out.textContent = "Running generate...";
      const result = await callApi("/api/generate", baseBody());
      const showDebug = document.getElementById("show_debug").checked;
      if (showDebug || result.error) {
        out.textContent = JSON.stringify(result.error || result.data, null, 2);
      } else {
        out.textContent = result.data.text;
      }
    };

    document.getElementById("go-rag").onclick = async () => {
      out.textContent = "Running RAG...";
      const body = {
        ...baseBody(),
        top_k: Number(document.getElementById("top_k").value),
      };
      const result = await callApi("/api/rag", body);
      const showDebug = document.getElementById("show_debug").checked;
      if (showDebug || result.error) {
        out.textContent = JSON.stringify(result.error || result.data, null, 2);
      } else {
        out.textContent = result.data.text;
      }
    };
  </script>
</body>
</html>
"""


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    max_new_tokens: int = Field(128, ge=1, le=512)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    do_sample: bool = True
    use_lora: bool = False


class GenerateResponse(BaseModel):
    text: str
    new_tokens: int
    elapsed_s: float
    device: str
    model_name: str
    lora_active: bool = False


class RagRequest(GenerateRequest):
    max_new_tokens: int = Field(180, ge=1, le=512)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    do_sample: bool = False
    top_k: int = Field(3, ge=1, le=8)


class RetrievedChunk(BaseModel):
    score: float
    text: str


class RagResponse(GenerateResponse):
    retrieved_chunks: list[RetrievedChunk]


def _pick_generation_model(use_lora: bool):
    if use_lora:
        m = _state.get("model_lora")
        if m is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "LoRA adapter not loaded. Run `python -m lesson05.lora_finetune` "
                    f"(status: {_state.get('lora_status', 'unknown')})"
                ),
            )
        return m
    return _state["model_base"]


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML


@app.get("/api/health")
def health() -> dict[str, Any]:
    if _state.get("error"):
        return {"status": "error", "detail": _state["error"]}
    if "model_base" not in _state:
        return {"status": "loading"}
    response: dict[str, Any] = {
        "status": "ok",
        "device": _state["device"],
        "model": _state["model_name"],
        "lora_status": _state.get("lora_status", ""),
        "lora_loaded": _state.get("model_lora") is not None,
    }
    if _state.get("rag_error"):
        response["rag_status"] = "error"
    else:
        response["rag_status"] = "ok"
    return response


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if err := _state.get("error"):
        raise HTTPException(status_code=503, detail=err)
    tokenizer = _state.get("tokenizer")
    device = _state.get("device")
    if tokenizer is None or device is None or "model_base" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    model = _pick_generation_model(req.use_lora)
    text, n_new, elapsed = generate_continuation_detailed(
        model,
        tokenizer,
        req.prompt,
        device,
        max_new_tokens=req.max_new_tokens,
        do_sample=req.do_sample,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    return GenerateResponse(
        text=text,
        new_tokens=n_new,
        elapsed_s=elapsed,
        device=device,
        model_name=str(_state.get("model_name", DEFAULT_MODEL_NAME)),
        lora_active=bool(req.use_lora and _state.get("model_lora") is not None),
    )


@app.post("/api/rag", response_model=RagResponse)
def rag_answer(req: RagRequest) -> RagResponse:
    if err := _state.get("error"):
        raise HTTPException(status_code=503, detail=err)
    if rag_err := _state.get("rag_error"):
        raise HTTPException(status_code=503, detail=f"RAG not available: {rag_err}")

    tokenizer = _state.get("tokenizer")
    emb_model = _state.get("emb_model")
    emb_tokenizer = _state.get("emb_tokenizer")
    chunks = _state.get("kb_chunks")
    device = _state.get("device")
    if (
        tokenizer is None
        or emb_model is None
        or emb_tokenizer is None
        or not chunks
        or device is None
        or "model_base" not in _state
    ):
        raise HTTPException(status_code=503, detail="RAG resources not loaded yet.")

    model = _pick_generation_model(req.use_lora)
    matches = retrieve_top_k(
        req.prompt, chunks, emb_tokenizer, emb_model, device=device, k=req.top_k
    )
    rag_prompt = build_rag_prompt(req.prompt, matches)
    text, n_new, elapsed = generate_continuation_detailed(
        model,
        tokenizer,
        rag_prompt,
        device,
        max_new_tokens=req.max_new_tokens,
        do_sample=req.do_sample,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    return RagResponse(
        text=text,
        new_tokens=n_new,
        elapsed_s=elapsed,
        device=device,
        model_name=str(_state.get("model_name", DEFAULT_MODEL_NAME)),
        lora_active=bool(req.use_lora and _state.get("model_lora") is not None),
        retrieved_chunks=[
            RetrievedChunk(score=round(m.score, 4), text=m.text) for m in matches
        ],
    )
