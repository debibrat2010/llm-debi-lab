# Lesson 3: Local Inference Patterns

## Theory Notes

- **Greedy decoding:** Always picks the highest-probability next token (`do_sample=False`). Outputs are usually stable but can be repetitive or bland.
- **Sampling:** Draws from the token distribution (`do_sample=True`). `temperature` scales randomness; `top_p` (nucleus) limits the candidate tail.
- **Repetition control:** `repetition_penalty > 1.0` discourages reusing tokens; helpful for bullet lists and chatty loops.
- **Stochastic runs:** With sampling, two runs with the same prompt can differ unless you fix a random seed.
- **Throughput:** `new_tokens / wall_time` is a simple local metric. Compare runs with the same model, device, and decoding settings.

## Practical Checklist

- [ ] Run greedy vs sampling on the same prompt and compare stability.
- [ ] Run sampling twice and note output variation.
- [ ] Record approximate tok/s on your Mac (`mps` vs `cpu` if you test both).
- [ ] Try raising/lowering `repetition_penalty` when the model loops.
