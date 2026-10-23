"""Lesson 4.2: simple local RAG evaluation runner."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

from llm_debi_lab.device import pick_device
from llm_debi_lab.generation import generate_continuation_detailed, load_causal_lm
from llm_debi_lab.rag import (
    EMBED_MODEL_NAME,
    build_rag_prompt,
    chunk_text,
    load_kb_texts,
    retrieve_top_k,
)
from llm_debi_lab.settings import DEFAULT_MODEL_NAME

ROOT = Path(__file__).resolve().parents[2]
KB_DIR = ROOT / "data" / "kb"
EVAL_SET_PATH = ROOT / "data" / "eval" / "rag_eval_set.jsonl"
REPORT_DIR = ROOT / "data" / "eval" / "reports"


@dataclass(frozen=True)
class EvalRow:
    eval_id: str
    question: str
    top1_score: float
    retrieval_ok: bool
    answer_has_keyword: bool
    answer_preview: str
    latency_s: float
    new_tokens: int


def load_eval_set(path: Path) -> list[dict]:
    if not path.exists():
        raise RuntimeError(f"Eval set not found: {path}")
    rows: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rows.append(json.loads(raw))
    if not rows:
        raise RuntimeError(f"Eval set is empty: {path}")
    return rows


def keyword_hit(answer: str, keywords: list[str]) -> bool:
    answer_l = answer.lower()
    return any(k.lower() in answer_l for k in keywords)


def write_reports(rows: list[EvalRow], report_base: Path) -> None:
    report_base.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = report_base.with_suffix(".jsonl")
    csv_path = report_base.with_suffix(".csv")

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.__dict__, ensure_ascii=True) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].__dict__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    print(f"\nSaved reports:\n- {jsonl_path}\n- {csv_path}")


def main() -> None:
    device = pick_device()
    print(f"Requested device: {device}")

    kb_docs = load_kb_texts(KB_DIR)
    if not kb_docs:
        raise RuntimeError(f"No KB docs found in: {KB_DIR}")
    chunks: list[str] = []
    for doc in kb_docs:
        chunks.extend(chunk_text(doc, chunk_size_words=65, overlap_words=15))
    print(f"KB chunks: {len(chunks)}")

    eval_rows = load_eval_set(EVAL_SET_PATH)
    print(f"Eval queries: {len(eval_rows)}")

    emb_tok = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    emb_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device)
    emb_model.eval()
    gen_model, gen_tok, gen_device = load_causal_lm(DEFAULT_MODEL_NAME, device)

    outputs: list[EvalRow] = []
    for row in eval_rows:
        qid = str(row["id"])
        question = str(row["question"])
        keywords = list(row.get("must_contain_any", []))
        min_top1 = float(row.get("min_top1_score", 0.2))

        matches = retrieve_top_k(question, chunks, emb_tok, emb_model, device, k=3)
        top1 = matches[0].score if matches else 0.0
        retrieval_ok = top1 >= min_top1

        rag_prompt = build_rag_prompt(question, matches)
        answer, n_new, elapsed = generate_continuation_detailed(
            gen_model,
            gen_tok,
            rag_prompt,
            gen_device,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.2,
            top_p=0.9,
        )
        has_kw = keyword_hit(answer, keywords) if keywords else True

        outputs.append(
            EvalRow(
                eval_id=qid,
                question=question,
                top1_score=round(top1, 4),
                retrieval_ok=retrieval_ok,
                answer_has_keyword=has_kw,
                answer_preview=answer[:180],
                latency_s=round(elapsed, 3),
                new_tokens=n_new,
            )
        )
        print(
            f"- {qid}: top1={top1:.3f} "
            f"retrieval_ok={retrieval_ok} answer_has_keyword={has_kw}"
        )

    retrieval_pass_rate = sum(1 for r in outputs if r.retrieval_ok) / len(outputs)
    answer_pass_rate = sum(1 for r in outputs if r.answer_has_keyword) / len(outputs)
    print(
        f"\nSummary: retrieval_pass_rate={retrieval_pass_rate:.0%}, "
        f"answer_keyword_pass_rate={answer_pass_rate:.0%}"
    )

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_base = REPORT_DIR / f"rag_eval_{ts}"
    write_reports(outputs, report_base)


if __name__ == "__main__":
    main()
