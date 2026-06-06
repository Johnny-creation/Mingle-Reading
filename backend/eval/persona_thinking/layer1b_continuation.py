# -*- coding: utf-8 -*-
"""Layer-1b: reasoning-continuation scored against the author's real next move.

Each condition continues the author's SETUP; a reference-based judge compares the
continuation to the author's GOLD next move (its real verbatim continuation). This
tests PRODUCING the author's reasoning, not recognising a stance — so it resists
the base-model recognition ceiling that limits Layer-1a.

Two judges, both reference-based (the judge always sees the gold reference, which
the literature shows is far more reliable than reference-free judging):
  - DeepSeek, in-pipeline (automatic). NOTE: same family as the generator, so it
    carries a self-preference risk — reported, not relied upon alone.
  - Claude, via judge_io file handoff (a DIFFERENT family = the cross-family
    control). Run with --export to emit the blind task file; after the
    cross-family judge fills in results, run with --import-claude to merge.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.eval.persona_thinking.common import call_model, parse_json_loose
from backend.eval.persona_thinking import judge_io
from backend.eval.persona_thinking.conditions import (
    CONDITIONS,
    CONTROL_CONDITIONS,
    system_prompt_for,
)

HERE = Path(__file__).resolve().parent
ITEMS_DIR = HERE / "continuation_items"
RESULTS_DIR = HERE / "results"
ALL_CONDITIONS = list(CONDITIONS) + list(CONTROL_CONDITIONS)

_CONT_INSTR = (
    "\n\n你正在续写一段分析。下面给你这段分析的开头（铺垫部分）。"
    "请顺着它，写出你接下来真正的推理与判断，2~4 句；不要复述开头，直接往下推进。"
)
_JUDGE_SYS = (
    "你是严格的推理评审。给你一段分析的【开头】(原作者的铺垫)、【参照】(原作者接下来的真实推理与判断)、"
    "以及一段【候选】续写。只评判推理动作与判断是否与【参照】一致，**不看文采、不看文风**。"
    "评分：2 = 核心推理动作与结论和参照一致（措辞可不同）；1 = 方向相关但偏浅或只对了一半；"
    "0 = 不同的论点 / 跑题 / 泛泛而谈。只输出 JSON：{\"score\": 0|1|2, \"note\": \"一句依据\"}。"
)


def load_items(persona: str) -> list[dict]:
    path = ITEMS_DIR / f"{persona}.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def continue_text(persona: str, condition: str, item: dict, use_cache: bool) -> str:
    system = system_prompt_for(persona, condition) + _CONT_INSTR
    return call_model(
        endpoint_key=persona,
        system_prompt=system,
        user_prompt=f"【分析开头】\n{item['setup']}",
        temperature=0.7,
        max_tokens=400,
        use_cache=use_cache,
        tag=f"cont_gen::{persona}::{condition}::{item['id']}",
    ).strip()


def judge_deepseek(item: dict, candidate: str, use_cache: bool, cond: str) -> int | None:
    user = (
        f"【开头】\n{item['setup']}\n\n【参照（原作者真实的下一步）】\n{item['gold']}\n\n"
        f"【候选】\n{candidate}\n\n请打分，只输出 JSON。"
    )
    raw = call_model(
        endpoint_key="neutral",
        system_prompt=_JUDGE_SYS,
        user_prompt=user,
        temperature=0.0,
        max_tokens=150,
        json_object=True,
        use_cache=use_cache,
        tag=f"cont_judge_ds::{item['id']}::{cond}",
    )
    try:
        sc = int(round(float(parse_json_loose(raw).get("score"))))
        return sc if sc in (0, 1, 2) else None
    except Exception:
        return None


def run(persona: str, *, use_cache: bool, max_items: int | None, export: bool, import_claude: bool) -> dict[str, Any]:
    items = load_items(persona)
    if max_items:
        items = items[:max_items]
    claude_scores = {}
    if import_claude:
        merged = judge_io.import_results(f"layer1b_{persona}")
        claude_scores = merged

    per_item, export_rows = [], []
    for item in items:
        cond_res = {}
        for cond in ALL_CONDITIONS:
            cand = continue_text(persona, cond, item, use_cache)
            ds = judge_deepseek(item, cand, use_cache, cond)
            key = f"{persona}::{item['id']}::{cond}"
            cl = claude_scores.get(key, {}).get("score") if import_claude else None
            cond_res[cond] = {"candidate": cand, "score_deepseek": ds, "score_claude": cl}
            if export:
                export_rows.append({
                    "persona": persona, "item_id": item["id"], "condition": cond,
                    "setup": item["setup"], "reference": item["gold"], "candidate": cand,
                })
        per_item.append({"id": item["id"], "move": item.get("move", ""), "conditions": cond_res})

    if export:
        tasks_path, key_path = judge_io.export_tasks(f"layer1b_{persona}", export_rows)
        print(f"[{persona}] exported {len(export_rows)} blind judge tasks -> {tasks_path.name}")

    return _aggregate(persona, per_item)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def _aggregate(persona: str, per_item: list[dict]) -> dict[str, Any]:
    summary = {}
    for c in ALL_CONDITIONS:
        ds = [p["conditions"][c]["score_deepseek"] for p in per_item]
        cl = [p["conditions"][c]["score_claude"] for p in per_item]
        summary[c] = {
            "mean_deepseek": _mean(ds),
            "mean_claude": _mean(cl),
            "scores_deepseek": ds,
            "scores_claude": cl,
        }
    return {"persona": persona, "n_items": len(per_item), "summary": summary, "per_item": per_item}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", nargs="+", default=["lu-xun", "zhang-ailing"])
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--export", action="store_true", help="emit blind Claude judge task files")
    ap.add_argument("--import-claude", action="store_true", help="merge Claude judge results back in")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for persona in args.personas:
        rep = run(persona, use_cache=not args.no_cache, max_items=args.max_items,
                  export=args.export, import_claude=args.import_claude)
        out = RESULTS_DIR / f"layer1b__{persona}.json"
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        s = rep["summary"]
        print(f"\n[{persona}] n={rep['n_items']}  continuation score (0-2):")
        print(f"  {'cond':14s} {'DeepSeek':>9s} {'Claude':>8s}")
        for c in ALL_CONDITIONS:
            print(f"  {c:14s} {str(s[c]['mean_deepseek']):>9s} {str(s[c]['mean_claude']):>8s}")
        print(f"  -> wrote {out.name}")


if __name__ == "__main__":
    main()
