"""Lesson 6: repeatable generation evaluation (base vs optional LoRA)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from transformers import AutoModelForCausalLM

from llm_debi_lab.device import pick_device
from llm_debi_lab.generation import generate_continuation_detailed, load_causal_lm
from llm_debi_lab.settings import DEFAULT_MODEL_NAME

ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = ROOT / "data" / "eval" / "generation_eval_set.jsonl"
REPORT_DIR = ROOT / "data" / "eval" / "reports"
LORA_ADAPTER_DIR = ROOT / "data" / "finetune" / "artifacts" / "lesson05-lora"


def keyword_hit(answer: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    lower = answer.lower()
    return any(k.lower() in lower for k in keywords)


def load_eval_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise RuntimeError(f"Eval set not found: {path}")
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"Eval set empty: {path}")
    return rows


def try_load_lora(device: str):
    try:
        from peft import PeftModel  # noqa: PLC0415
    except ImportError:
        return None, "peft-not-installed"
    if not LORA_ADAPTER_DIR.is_dir() or not any(LORA_ADAPTER_DIR.iterdir()):
        return None, f"missing:{LORA_ADAPTER_DIR}"
    try:
        inner = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_NAME).to(device)
        wrapped = PeftModel.from_pretrained(inner, str(LORA_ADAPTER_DIR))
        wrapped.eval()
        return wrapped, "ok"
    except Exception as exc:  # pragma: no cover
        return None, str(exc)


@dataclass(frozen=True)
class GenerationEvalRow:
    eval_id: str
    prompt: str
    base_preview: str
    base_new_tokens: int
    base_elapsed_s: float
    base_keyword_ok: bool
    lora_preview: str
    lora_new_tokens: int
    lora_elapsed_s: float
    lora_keyword_ok: bool
    lora_note: str


def write_reports(rows: list[GenerationEvalRow], base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = base_path.with_suffix(".jsonl")
    csv_path = base_path.with_suffix(".csv")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.__dict__, ensure_ascii=True) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as wf:
        w = csv.DictWriter(wf, fieldnames=list(rows[0].__dict__.keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row.__dict__)
    print(f"\nReports:\n- {jsonl_path}\n- {csv_path}")


def main() -> None:
    device_req = pick_device()
    print(f"Device (requested): {device_req}")
    print(f"Base model: {DEFAULT_MODEL_NAME}")

    base_model, tokenizer, device = load_causal_lm(DEFAULT_MODEL_NAME, device_req)
    lora_model, lora_note = try_load_lora(device)
    if lora_model is None:
        print(f"LoRA: skipped ({lora_note})")
    else:
        print(f"LoRA: loaded ({lora_note})")

    cases = load_eval_rows(EVAL_PATH)
    print(f"Cases: {len(cases)}")

    out_rows: list[GenerationEvalRow] = []
    gen_kw = dict(do_sample=False, temperature=0.2, top_p=0.9)

    for row in cases:
        eid = str(row["id"])
        prompt = str(row["prompt"])
        keys = list(row.get("must_contain_any", []))

        b_txt, b_nt, b_el = generate_continuation_detailed(
            base_model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=160,
            **gen_kw,
        )
        b_ok = keyword_hit(b_txt, keys)

        if lora_model is not None:
            l_txt, l_nt, l_el = generate_continuation_detailed(
                lora_model,
                tokenizer,
                prompt,
                device,
                max_new_tokens=160,
                **gen_kw,
            )
            l_ok = keyword_hit(l_txt, keys)
            l_note = "ok"
        else:
            l_txt, l_nt, l_el, l_ok = "", -1, 0.0, False
            l_note = lora_note or "no-lora"

        out_rows.append(
            GenerationEvalRow(
                eval_id=eid,
                prompt=prompt,
                base_preview=b_txt[:220],
                base_new_tokens=b_nt,
                base_elapsed_s=round(b_el, 4),
                base_keyword_ok=b_ok,
                lora_preview=l_txt[:220],
                lora_new_tokens=l_nt,
                lora_elapsed_s=round(l_el, 4),
                lora_keyword_ok=l_ok,
                lora_note=l_note,
            )
        )
        print(
            f"- {eid}: base_kw={b_ok} "
            + (f"lora_kw={l_ok}" if lora_model else "lora=skipped")
        )

    base_kw_rate = sum(1 for r in out_rows if r.base_keyword_ok) / len(out_rows)
    lora_runs = [r for r in out_rows if r.lora_new_tokens >= 0]
    lora_kw_rate = (
        sum(1 for r in lora_runs if r.lora_keyword_ok) / len(lora_runs)
        if lora_runs
        else 0.0
    )
    print(
        f"\nSummary: base_keyword_pass={base_kw_rate:.0%}  "
        f"lora_keyword_pass={lora_kw_rate:.0%} (n_lora={len(lora_runs)})"
    )

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    write_reports(out_rows, REPORT_DIR / f"generation_eval_{ts}")


if __name__ == "__main__":
    main()
