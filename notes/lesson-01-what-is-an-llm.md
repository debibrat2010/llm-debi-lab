# Lesson 1: What is an LLM?

## Notebook Theory Points

- **Definition:** A large neural network trained mostly to predict the next token (a piece of text, often a subword) given previous tokens. Long outputs are produced by repeated prediction.
- **Token:** Not always a full word; tokenization can split words into subwords, punctuation, and symbols.
- **Context window:** Maximum tokens the model can consider at once (input + generated output).
- **No built-in truth:** LLMs generate plausible language, not guaranteed facts.
- **Temperature:** Higher values increase randomness; lower values improve consistency.
- **Local inference:** Running models on your own machine gives privacy and control, but requires hardware-aware model selection.
- **Apple Silicon tip:** Prefer small models first and use `mps` acceleration where supported.

## Default model

Lessons use **`HuggingFaceTB/SmolLM2-360M-Instruct`** by default (see `LLM_DEBI_MODEL`). Scripts print **only newly generated tokens**, not the full prompt echo.

## Practical Checklist

- [ ] Understand tokenization by inspecting token IDs.
- [ ] Run one local generation script.
- [ ] Compare outputs with low vs high temperature.
