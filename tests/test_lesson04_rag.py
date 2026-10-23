from llm_debi_lab.rag import build_rag_prompt, chunk_text, normalize_text


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("a   b\n\nc\t d") == "a b c d"


def test_chunk_text_has_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(20))
    chunks = chunk_text(text, chunk_size_words=8, overlap_words=2)
    assert len(chunks) >= 3
    first_words = chunks[0].split(" ")
    second_words = chunks[1].split(" ")
    # Last two words of chunk 1 should reappear at start of chunk 2
    assert first_words[-2:] == second_words[:2]


def test_build_rag_prompt_includes_question() -> None:
    prompt = build_rag_prompt("What is RAG?", [])
    assert "Question:\nWhat is RAG?" in prompt
    assert "No context retrieved." in prompt
