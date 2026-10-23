# Lesson 2: Prompting Fundamentals

## Theory Notes (Short + Practical)

- **Prompting is interface design:** You are specifying task requirements for a probabilistic text engine.
- **Weak prompt:** Broad request with little structure; output quality varies more.
- **Strong prompt:** Includes role, task, constraints, and output format; reduces ambiguity.
- **Why structure helps:** The model predicts next tokens; clearer constraints narrow the likely token paths.
- **Temperature effect:** Lower values improve repeatability. Use low temperature for factual or format-sensitive tasks.
- **Evaluation mindset:** Judge by usefulness and adherence to requirements, not by style alone.

## Prompt Template You Can Reuse

1. Role: "You are a ___"
2. Task: "Do ___ for ___ audience"
3. Constraints: length, tone, inclusions/exclusions
4. Output format: bullets, JSON, markdown, etc.

## Model note

Lessons default to **`HuggingFaceTB/SmolLM2-360M-Instruct`** (instruct-tuned). Set `LLM_DEBI_MODEL` to swap models. Base models like `distilgpt2` often ignore bullet/length instructions.

## Practical Checklist

- [ ] Compare weak vs strong prompts on same topic.
- [ ] Hold temperature constant while changing only prompt quality.
- [ ] Run each prompt at least 3 times and note consistency.
- [ ] Save one "best prompt" and one "bad prompt" with observations.
