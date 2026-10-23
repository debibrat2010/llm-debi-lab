from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class ChunkMatch:
    text: str
    score: float


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size_words: int = 60, overlap_words: int = 15) -> list[str]:
    words = normalize_text(text).split(" ")
    if not words or chunk_size_words <= 0:
        return []
    step = max(1, chunk_size_words - max(0, overlap_words))
    chunks: list[str] = []
    for i in range(0, len(words), step):
        piece = words[i : i + chunk_size_words]
        if not piece:
            continue
        chunks.append(" ".join(piece))
        if i + chunk_size_words >= len(words):
            break
    return chunks


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
) -> torch.Tensor:
    if not texts:
        return torch.empty((0, 0))
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        outputs = model(**tokens)
    embeddings = mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
    return F.normalize(embeddings, p=2, dim=1)


def retrieve_top_k(
    query: str,
    chunks: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    k: int = 3,
) -> list[ChunkMatch]:
    if not chunks:
        return []
    doc_emb = embed_texts(chunks, tokenizer, model, device)
    query_emb = embed_texts([query], tokenizer, model, device)
    if doc_emb.numel() == 0 or query_emb.numel() == 0:
        return []
    scores = torch.matmul(doc_emb, query_emb[0])
    top_k = min(k, len(chunks))
    vals, idxs = torch.topk(scores, k=top_k)
    return [
        ChunkMatch(text=chunks[int(i)], score=float(v))
        for v, i in zip(vals.tolist(), idxs.tolist())
    ]


def build_rag_prompt(question: str, matches: list[ChunkMatch]) -> str:
    context_lines = []
    for i, m in enumerate(matches, start=1):
        context_lines.append(f"[Chunk {i}] {m.text}")
    context = "\n".join(context_lines) if context_lines else "No context retrieved."
    return (
        "You are a helpful local RAG assistant.\n"
        "Answer ONLY using the retrieved context. If context is insufficient, say so clearly.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Return exactly one plain paragraph (80-120 words). No bullets, no headings, no metadata."
    )


def load_kb_texts(kb_dir: Path) -> list[str]:
    if not kb_dir.exists():
        return []
    texts: list[str] = []
    for path in sorted(kb_dir.glob("*.txt")):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            texts.append(content)
    return texts

