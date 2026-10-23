# Lesson 4: RAG Basics

## Core Idea

- **RAG = Retrieval + Generation**. Instead of asking the LLM to answer from parametric memory alone, we retrieve relevant text and pass it into the prompt.
- This improves grounding and freshness, especially for private/local docs.

## Pipeline Mental Model

1. **Chunk** documents into manageable text segments.
2. **Embed** each chunk into a vector.
3. **Retrieve** top-k similar chunks for a query.
4. **Generate** an answer constrained by retrieved context.

## Important Practical Notes

- **Chunk size tradeoff:** big chunks add noise; tiny chunks lose context.
- **Overlap matters:** overlap between neighboring chunks reduces boundary loss.
- **Retrieval quality is a separate problem:** weak retrieval => weak answer, even with a good LLM.
- **Similarity metric:** cosine similarity on normalized embeddings is a common baseline.
- **RAG reduces but does not eliminate hallucinations.**

## Beginner Evaluation Checklist

- [ ] Log retrieved chunks with scores for each query.
- [ ] Manually verify whether top-k chunks actually answer the question.
- [ ] Compare "LLM only" vs "RAG prompt" for the same questions.
- [ ] Track failure type: retrieval miss vs generation mistake.

## Lesson 4.1 (App Integration)

- Put local documents in `data/kb/*.txt`.
- `POST /api/rag` retrieves top-k chunks and then generates a grounded answer.
- Always inspect `retrieved_chunks` in the response before trusting the final text.

## Lesson 4.2 (RAG Evaluation Runner)

- Script: `python -m lesson04.rag_eval`
- Eval set: `data/eval/rag_eval_set.jsonl`
- Reports: `data/eval/reports/rag_eval_<timestamp>.jsonl|csv`

Track at least these metrics per query:

- **Top-1 retrieval score** (is retrieval likely relevant?)
- **Retrieval pass/fail** against per-query threshold
- **Answer keyword hit** (quick semantic sanity check)
- **Latency and output length** (`elapsed_s`, `new_tokens`)
