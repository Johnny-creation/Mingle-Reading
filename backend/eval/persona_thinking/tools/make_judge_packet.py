# -*- coding: utf-8 -*-
"""Build a BLIND judging packet from the latest full result, for a cross-family
judge (Claude, acting in-conversation) to score independently.

Outputs (in results/):
  - claude_judge_packet.json : shuffled items {item_id, persona, persona_name,
        scenario, stripped_text} + each persona's rubric + corpus anchors.
        Contains NO condition label, so the judge cannot infer full/style/neutral.
  - claude_judge_mapping.json : item_id -> {persona, probe_id, condition} and
        the DeepSeek per-dim scores for the same item (for later agreement calc).
        The judge must NOT read this until after scoring.

Run from Mingle-Reading-main/:
  python backend/eval/persona_thinking/tools/make_judge_packet.py
"""
from __future__ import annotations

import glob
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve()
PKG = HERE.parents[1]
RESULTS = PKG / "results"
ANCHORS = PKG / "corpus_anchors"
RUBRICS = PKG / "rubrics"


def latest_full_result() -> Path:
    files = sorted(RESULTS.glob("result_*.json"))
    # pick the one with the most probes (full run), else newest
    best, best_n = None, -1
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        n = sum(p["n_probes"] for p in d["personas"])
        if n >= best_n:
            best, best_n = f, n
    return best


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def main() -> None:
    src = latest_full_result()
    data = json.loads(src.read_text(encoding="utf-8"))
    rng = random.Random(12345)

    items = []
    mapping = {}
    rubrics = {}
    anchors = {}
    for pr in data["personas"]:
        persona = pr["persona"]
        rubrics[persona] = json.loads((RUBRICS / f"{persona}.json").read_text(encoding="utf-8"))
        ap = ANCHORS / f"{persona}.json"
        anchors[persona] = json.loads(ap.read_text(encoding="utf-8")) if ap.exists() else {"anchors": []}
        dim_ids = [d["id"] for d in rubrics[persona]["dimensions"]]
        for p in pr["per_probe"]:
            for cond, cd in p["conditions"].items():
                iid = f"IT{len(items):03d}"
                items.append(
                    {
                        "item_id": iid,
                        "persona": persona,
                        "persona_name": pr["display_name"],
                        "scenario": p["scenario"],
                        "stripped_text": cd["stripped"],
                    }
                )
                ds_scores = {
                    d: mean([run[d]["score"] for run in cd["rubric_runs"] if run[d]["score"] is not None])
                    for d in dim_ids
                }
                mapping[iid] = {
                    "persona": persona,
                    "probe_id": p["probe_id"],
                    "condition": cond,
                    "deepseek_dim_scores": ds_scores,
                }
    rng.shuffle(items)

    packet = {
        "source_result": src.name,
        "instructions": (
            "你是跨族裁判。对每个 item，依据其 persona 的 rubric 维度和该作家真实原文锚点，"
            "对‘去风格后的文本’在每个思维维度打 1-5 整数分。只看思维方式（认知动作/价值透镜/框定），"
            "不看文采。输出 {item_id: {维度ID: 分数}}。"
        ),
        "rubrics": {
            k: {
                "display_name": v["display_name"],
                "scale": v["scale"],
                "dimensions": [
                    {"id": d["id"], "name": d["name"], "description": d["description"],
                     "anchor_high": d["anchor_high"], "anchor_low": d["anchor_low"]}
                    for d in v["dimensions"]
                ],
            }
            for k, v in rubrics.items()
        },
        "anchors": {
            k: [{"label": a["label"], "moves": a.get("moves", []), "excerpt": a["excerpt"][:900]}
                for a in v.get("anchors", []) if a.get("found") and a.get("excerpt")]
            for k, v in anchors.items()
        },
        "items": items,
    }
    (RESULTS / "claude_judge_packet.json").write_text(
        json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "claude_judge_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"source: {src.name}")
    print(f"items: {len(items)} (personas={[pr['persona'] for pr in data['personas']]})")
    print("wrote results/claude_judge_packet.json + claude_judge_mapping.json")


if __name__ == "__main__":
    main()
