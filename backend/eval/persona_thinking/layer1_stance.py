# -*- coding: utf-8 -*-
"""Layer-1a: held-out stance prediction (the no-judge main evidence).

Each item presents a concrete situation from a held-out essay and four candidate
judgments; exactly one is what the author actually argues (ground truth = the
author's own text). Every condition gets the SAME multiple-choice user prompt;
only the system prompt (persona condition) differs. We measure ACCURACY against
the author's real stance — no LLM judge, fully reproducible.

The discriminating signal is the `distinctive` subset: items where the NEUTRAL
baseline is wrong (the author's judgment diverges from the generic/common-sense
reading). full / thinking_only beating neutral *there* is the hard evidence that
the agent reproduces the author's specific thinking, not textbook recall.

Output feeds the double-dissociation table (T axis) and the stats layer.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from backend.eval.persona_thinking.common import call_model, parse_json_loose
from backend.eval.persona_thinking.conditions import (
    CONDITIONS,
    CONTROL_CONDITIONS,
    system_prompt_for,
)

HERE = Path(__file__).resolve().parent
STANCE_DIR = HERE / "stance_items"
RESULTS_DIR = HERE / "results"
DISPLAY = {"lu-xun": "鲁迅", "zhang-ailing": "张爱玲"}

ALL_CONDITIONS = list(CONDITIONS) + list(CONTROL_CONDITIONS)

_ANSWER_SYS = (
    "你将看到一个具体问题情境和四个候选判断（A/B/C/D）。"
    "请依据你的立场与思维方式，选出最贴近的那一个判断。"
    "只输出 JSON：{\"choice\": \"A\" 或 \"B\" 或 \"C\" 或 \"D\", \"reason\": \"一句话依据\"}。"
)


def load_items(persona: str) -> list[dict]:
    """Prefer the human-reviewed final set; fall back to raw candidates."""
    final = STANCE_DIR / f"{persona}.jsonl"
    cand = STANCE_DIR / f"{persona}.candidates.jsonl"
    path = final if final.exists() else cand
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _ask(persona: str, condition: str, item: dict, run_idx: int, use_cache: bool) -> str | None:
    author = DISPLAY[persona]
    opts = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(item["options"]))
    user = (
        f"【问题情境】{item['topic']}\n\n"
        f"【在这一问题上，{author}的判断最接近以下哪一项？】\n{opts}\n\n"
        "请选择，并只输出 JSON。"
    )
    # the persona system prompt + a short answer-format instruction
    system = system_prompt_for(persona, condition) + "\n\n" + _ANSWER_SYS
    raw = call_model(
        endpoint_key=persona,                      # all conditions run on this persona's endpoint
        system_prompt=system,
        user_prompt=user,
        temperature=0.7,
        max_tokens=200,
        json_object=True,
        use_cache=use_cache,
        tag=f"stance_ans::{persona}::{condition}::{item['id']}::r{run_idx}",
    )
    try:
        data = parse_json_loose(raw)
        ch = str(data.get("choice", "")).strip().upper()
        m = re.search(r"[ABCD]", ch)
        return m.group(0) if m else None
    except Exception:
        return None


def evaluate(persona: str, *, runs: int, use_cache: bool, max_items: int | None) -> dict[str, Any]:
    items = load_items(persona)
    if max_items:
        items = items[:max_items]
    per_item: list[dict] = []
    for item in items:
        correct_letter = chr(65 + item["correct_index"])
        cond_res: dict[str, Any] = {}
        for cond in ALL_CONDITIONS:
            choices = [_ask(persona, cond, item, r, use_cache) for r in range(runs)]
            valid = [c for c in choices if c]
            n_correct = sum(1 for c in valid if c == correct_letter)
            majority = Counter(valid).most_common(1)[0][0] if valid else None
            cond_res[cond] = {
                "choices": choices,
                "n_valid": len(valid),
                "n_correct": n_correct,
                "acc": round(n_correct / len(valid), 3) if valid else None,
                "majority": majority,
                "majority_correct": majority == correct_letter,
            }
        per_item.append(
            {
                "id": item["id"],
                "source_title": item.get("source_title", ""),
                "move": item.get("move", ""),
                "correct": correct_letter,
                "evidence_verified": item.get("evidence_verified", None),
                "conditions": cond_res,
                # distinctive = the generic baseline misses it
                "distinctive": cond_res["neutral"]["majority_correct"] is False,
            }
        )
    return _aggregate(persona, per_item, runs)


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def _aggregate(persona: str, per_item: list[dict], runs: int) -> dict[str, Any]:
    distinct = [p for p in per_item if p["distinctive"]]

    def acc_over(subset: list[dict], cond: str, mode: str) -> float | None:
        if mode == "soft":  # mean per-item accuracy across runs
            return _mean([p["conditions"][cond]["acc"] for p in subset])
        return _mean([1.0 if p["conditions"][cond]["majority_correct"] else 0.0 for p in subset])

    summary = {}
    for cond in ALL_CONDITIONS:
        summary[cond] = {
            "acc_all_soft": acc_over(per_item, cond, "soft"),
            "acc_all_majority": acc_over(per_item, cond, "majority"),
            "acc_distinctive_majority": acc_over(distinct, cond, "majority"),
        }
    return {
        "persona": persona,
        "display_name": DISPLAY[persona],
        "n_items": len(per_item),
        "n_distinctive": len(distinct),
        "runs": runs,
        "summary": summary,
        "per_item": per_item,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", nargs="+", default=["lu-xun", "zhang-ailing"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for persona in args.personas:
        rep = evaluate(persona, runs=args.runs, use_cache=not args.no_cache, max_items=args.max_items)
        out = RESULTS_DIR / f"layer1_stance__{persona}.json"
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        s = rep["summary"]
        print(f"\n[{rep['display_name']}] n={rep['n_items']} (distinctive={rep['n_distinctive']}), runs={rep['runs']}")
        print(f"  {'cond':14s} {'acc_all':>8s} {'acc_distinct':>13s}")
        for c in ALL_CONDITIONS:
            print(f"  {c:14s} {str(s[c]['acc_all_majority']):>8s} {str(s[c]['acc_distinctive_majority']):>13s}")
        print(f"  -> wrote {out.name}")


if __name__ == "__main__":
    main()
