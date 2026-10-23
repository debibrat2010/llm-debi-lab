import os

# Small instruct-tuned model; good on Apple Silicon with MPS. Override via env.
DEFAULT_MODEL_NAME = os.environ.get(
    "LLM_DEBI_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct"
)

# Optional LoRA adapter directory (Lesson 5). Empty = use default path next to repo data/.
LLM_DEBI_LORA_PATH = os.environ.get("LLM_DEBI_LORA_PATH", "").strip()
