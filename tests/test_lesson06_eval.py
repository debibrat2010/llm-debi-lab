from contextlib import contextmanager

from llm_debi_lab.generation import adapter_scope
from lesson06.generation_eval import keyword_hit


def test_keyword_hit_any_match() -> None:
    assert keyword_hit("Use fixed prompts and compare runs.", ["fixed", "missing"]) is True


def test_keyword_hit_empty_keywords_always_true() -> None:
    assert keyword_hit("anything", []) is True


class _Toggle:
    def __init__(self) -> None:
        self.enabled = True

    @contextmanager
    def disable_adapter(self):
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True


def test_adapter_scope_toggles_a_shared_model() -> None:
    model = _Toggle()
    with adapter_scope(model, enabled=False):
        assert model.enabled is False
    assert model.enabled is True
    with adapter_scope(model, enabled=True):
        assert model.enabled is True


def test_adapter_scope_leaves_a_plain_model_alone() -> None:
    with adapter_scope(object(), enabled=False):
        pass
