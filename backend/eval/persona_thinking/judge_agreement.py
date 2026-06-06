# -*- coding: utf-8 -*-
"""Cross-family judge agreement on the Layer-1b continuation scores (N judges).

The in-pipeline judge is DeepSeek (same family as the generator), so its scores
carry a self-preference risk. To control for that, one or more cross-family judges
score a BLIND, condition-shuffled random subset of the same continuations (see
judge_io.py / judge_handoff/). This module joins every available judge's scores
per (persona, item, condition) and reports, for each judge and each pair of judges:

  * coverage (how many tasks each judge covered)
  * each judge's per-condition mean ON ITS OWN covered items (so the condition
    ORDERING can be compared like-for-like)
  * pairwise exact-agreement, within-1 agreement, and Pearson r on the SHARED
    items between the two judges

Judges (auto-detected; select a subset with --judges):
  deepseek : in-pipeline, full coverage — results/layer1b__<persona>.json
  claude   : cross-family — judge_handoff/layer1b_<persona>.judge_results.jsonl
  codex    : cross-family — judge_handoff/layer1b_<persona>.judge_results_codex.jsonl
  (any other NAME maps to judge_handoff/layer1b_<persona>.judge_results_<NAME>.jsonl)

If a cross-family judge (a) agrees with DeepSeek and (b) reproduces the same
condition ordering on its covered items, the Layer-1b finding is not a DeepSeek
self-preference artifact. Because each subset is a random (shuffled) sample over
conditions, the per-condition means are unbiased though noisier than DeepSeek's
full-coverage means.

Run from Mingle-Reading-main/:
    python -m backend.eval.persona_thinking.judge_agreement --personas lu-xun zhang-ailing
    python -m backend.eval.persona_thinking.judge_agreement --judges deepseek claude codex
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from backend.eval.persona_thinking.conditions import CONDITIONS, CONTROL_CONDITIONS

HERE = Path(__file__).resolve().parent
HANDOFF = HERE / "judge_handoff"
RESULTS = HERE / "results"
ALL_CONDITIONS = list(CONDITIONS) + list(CONTROL_CONDITIONS)


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return None  # undefined (a judge gave constant scores on the overlap)
    return round(sxy / (sxx * syy) ** 0.5, 4)


# ---------- loaders: each returns {(item_id, condition): score} ----------

def _load_deepseek(persona: str) -> dict[tuple[str, str], int]:
    p = RESULTS / f"layer1b__{persona}.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], int] = {}
    for item in data["per_item"]:
        for cond, c in item["conditions"].items():
            sc = c.get("score_deepseek")
            if sc is not None:
                out[(item["id"], cond)] = int(sc)
    return out


def _load_handoff(persona: str, suffix: str) -> dict[tuple[str, str], int]:
    """suffix='' -> .judge_results.jsonl (claude); suffix='codex' -> ..._codex.jsonl."""
    tail = ".judge_results.jsonl" if not suffix else f".judge_results_{suffix}.jsonl"
    res = HANDOFF / f"layer1b_{persona}{tail}"
    key_path = HANDOFF / f"layer1b_{persona}.judge_key.json"
    if not res.exists() or not key_path.exists():
        return {}
    key = json.loads(key_path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], int] = {}
    for line in res.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        meta = key.get(r["task_id"])
        if meta and r.get("score") is not None:
            out[(meta["item_id"], meta["condition"])] = int(r["score"])
    return out


def load_judge(persona: str, judge: str) -> dict[tuple[str, str], int]:
    if judge == "deepseek":
        return _load_deepseek(persona)
    if judge == "claude":
        return _load_handoff(persona, "")
    return _load_handoff(persona, judge)  # codex / gpt / any custom suffix


def available_judges(persona: str) -> list[str]:
    found = ["deepseek"] if _load_deepseek(persona) else []
    if _load_handoff(persona, ""):
        found.append("claude")
    for f in sorted(HANDOFF.glob(f"layer1b_{persona}.judge_results_*.jsonl")):
        name = f.name.split(".judge_results_")[1][: -len(".jsonl")]
        if name not in found:
            found.append(name)
    return found


# ---------- analysis ----------

def analyze(persona: str, judges: list[str]) -> dict[str, Any]:
    scores = {j: load_judge(persona, j) for j in judges}
    scores = {j: s for j, s in scores.items() if s}  # drop empty
    out: dict[str, Any] = {"persona": persona, "judges": list(scores.keys())}

    # per-judge coverage + per-condition means on own covered items
    per_judge = {}
    for j, s in scores.items():
        by_cond = {}
        for c in ALL_CONDITIONS:
            vals = [v for (iid, cond), v in s.items() if cond == c]
            by_cond[c] = {"n": len(vals), "mean": _mean([float(x) for x in vals])}
        per_judge[j] = {"n_covered": len(s), "overall_mean": _mean([float(v) for v in s.values()]),
                        "by_condition": by_cond}
    out["per_judge"] = per_judge

    # pairwise agreement on shared items
    pairs = {}
    for a, b in combinations(scores.keys(), 2):
        shared = sorted(set(scores[a]) & set(scores[b]))
        xa = [scores[a][k] for k in shared]
        xb = [scores[b][k] for k in shared]
        pairs[f"{a}__vs__{b}"] = {
            "n_shared": len(shared),
            "exact_agreement": _mean([1.0 if u == v else 0.0 for u, v in zip(xa, xb)]),
            "within1_agreement": _mean([1.0 if abs(u - v) <= 1 else 0.0 for u, v in zip(xa, xb)]),
            "pearson_r": _pearson([float(x) for x in xa], [float(x) for x in xb]),
        }
    out["pairwise"] = pairs
    return out


def render(report: dict) -> str:
    L = ["# Layer-1b 跨族判官一致性（多裁判）", "",
         f"生成时间：{report['generated_at']}", "",
         "> DeepSeek 与被测 agent 同族（有自我偏好风险）。其余裁判为**跨族**，对盲化、打乱条件后的",
         "> 随机子集按同一 reference-based rubric 独立打分。跨族裁判**复现同一条件排序** + 与 DeepSeek",
         "> 一致 = 该结论非同族自偏所致。完全一致率 / ±1 内一致率以裁判**共同覆盖**的题计算。", ""]
    for pr in report["personas"]:
        L += [f"## {pr['persona']}", "", "**各裁判分条件均分（基于各自覆盖的题）**", "",
              "| 裁判 | 覆盖题数 | " + " | ".join(ALL_CONDITIONS) + " |",
              "|---|---|" + "---|" * len(ALL_CONDITIONS)]
        for j, d in pr["per_judge"].items():
            row = [j, str(d["n_covered"])]
            for c in ALL_CONDITIONS:
                bc = d["by_condition"][c]
                row.append(f"{bc['mean']}（n={bc['n']}）")
            L.append("| " + " | ".join(row) + " |")
        L += ["", "**两两裁判一致性（基于共同覆盖的题）**", "",
              "| 裁判对 | 共同题数 | 完全一致 | ±1内一致 | Pearson r |",
              "|---|---|---|---|---|"]
        for name, p in pr["pairwise"].items():
            L.append(f"| {name.replace('__vs__', ' × ')} | {p['n_shared']} | "
                     f"{p['exact_agreement']} | {p['within1_agreement']} | {p['pearson_r']} |")
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", nargs="+", default=["lu-xun", "zhang-ailing"])
    ap.add_argument("--judges", nargs="+", default=None,
                    help="subset of judges (default: all auto-detected). e.g. deepseek claude codex")
    args = ap.parse_args()
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"), "personas": []}
    for persona in args.personas:
        judges = args.judges or available_judges(persona)
        report["personas"].append(analyze(persona, judges))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (RESULTS / f"judge_agreement_{stamp}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render(report)
    (RESULTS / f"judge_agreement_{stamp}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[judge_agreement] wrote judge_agreement_{stamp}.json / .md")


if __name__ == "__main__":
    main()
