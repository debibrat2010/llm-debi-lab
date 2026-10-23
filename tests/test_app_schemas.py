import pytest

pytest.importorskip("fastapi")

from app.main import GenerateRequest, RagRequest  # noqa: E402


def test_generate_request_accepts_prompt() -> None:
    req = GenerateRequest(prompt="hello")
    assert req.prompt == "hello"
    assert req.max_new_tokens == 128
    assert req.use_lora is False


def test_rag_request_defaults() -> None:
    req = RagRequest(prompt="what is rag?")
    assert req.top_k == 3
    assert req.do_sample is False
    assert req.temperature == 0.2
