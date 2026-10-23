from llm_debi_lab.device import pick_device


def test_pick_device_is_valid() -> None:
    assert pick_device() in {"mps", "cpu"}
