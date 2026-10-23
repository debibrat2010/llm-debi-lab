"""Lesson 4: local RAG basics (chunk -> embed -> retrieve -> generate)."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoModel, AutoTokenizer

from llm_debi_lab.device import pick_device
from llm_debi_lab.generation import generate_continuation, load_causal_lm
from llm_debi_lab.rag import (
    EMBED_MODEL_NAME,
    build_rag_prompt,
    chunk_text,
    load_kb_texts,
    retrieve_top_k,
)
from llm_debi_lab.settings import DEFAULT_MODEL_NAME

KB_DIR = Path(__file__).resolve().parents[2] / "data" / "kb"


def main() -> None:
    device = pick_device()
    print(f"Requested device: {device}")

    # 1) Build local chunks from small .txt files in data/kb
    kb_docs = load_kb_texts(KB_DIR)
    if not kb_docs:
        raise RuntimeError(f"No KB files found in {KB_DIR}")
    chunks: list[str] = []
    for doc in kb_docs:
        chunks.extend(chunk_text(doc, chunk_size_words=55, overlap_words=12))
    print(f"Created {len(chunks)} chunks.")

    # 2) Load local embedding model and retrieve top chunks
    emb_tok = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    emb_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device)
    emb_model.eval()
    question = "Why can RAG still hallucinate even when retrieval is used?"
    matches = retrieve_top_k(question, chunks, emb_tok, emb_model, device, k=3)

    print("\nTop retrieved chunks:")
    for i, m in enumerate(matches, start=1):
        print(f"{i}. score={m.score:.3f} | {m.text[:140]}...")

    # 3) Build grounded prompt and generate answer with local instruct model
    rag_prompt = build_rag_prompt(question, matches)
    gen_model, gen_tok, gen_device = load_causal_lm(DEFAULT_MODEL_NAME, device)
    answer = generate_continuation(
        gen_model,
        gen_tok,
        rag_prompt,
        gen_device,
        max_new_tokens=180,
        do_sample=True,
        temperature=0.35,
        top_p=0.9,
        repetition_penalty=1.12,
    )

    print("\nRAG answer:\n")
    print(answer)


if __name__ == "__main__":
    main()
