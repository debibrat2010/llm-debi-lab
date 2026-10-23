import torch


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
