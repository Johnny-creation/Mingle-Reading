# -*- coding: utf-8 -*-
"""Join Claude's blind scores with the DeepSeek scores + condition mapping, then
report (a) the headline conclusion under Claude-as-judge and (b) Claude-vs-DeepSeek
agreement — the cross-family reliability check.

Inputs (results/):
  - claude_judge_mapping.json  (from make_judge_packet.py)
  - claude_scores.json         {item_id: {dim_id: 1-5}}  ← produced by Claude

Run from Mingle-Reading-main/:
  python backend/eval/persona_thinking/tools/analyze_cross_judge.py
"""
from __future__ import annotations

import json
import statistics as stats
from pathlib import Path

HERE = Path(__file__).resolve()
PKG = HERE.parents[1]
RESULTS = PKG / "results"
CONDITIONS = ("full", "style_only", "neutral")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def main() -> None:
    mapping = json.loads((RESULTS / "claude_judge_mapping.json").read_text(encoding="utf-8"))
    claude = json.loads((RESULTS / "claude_scores.json").read_text(encoding="utf-8"))

    personas = sorted({m["persona"] for m in mapping.values()})
    report = {"personas": []}
    abs_diffs, claude_flat, ds_flat = [], [], []

    for persona in personas:
        items = {iid: m for iid, m in mapping.items() if m["persona"] == persona}
        dim_ids = list(next(iter(items.values()))["deepseek_dim_scores"].keys())

        # per-condition Claude totals (mean over probes), and per-probe totals for forced choice
        cond_totals = {c: [] for c in CONDITIONS}
        cond_dim = {c: {d: [] for d in dim_ids} for c in CONDITIONS}
        by_probe: dict[str, dict[str, float]] = {}

        for iid, m in items.items():
            cs = claude.get(iid, {})
            tot = mean([cs.get(d) for d in dim_ids])
            cond_totals[m["condition"]].append(tot)
            for d in dim_ids:
                if cs.get(d) is not None:
                    cond_dim[m["condition"]][d].append(cs[d])
            by_probe.setdefault(m["probe_id"], {})[m["condition"]] = tot
            # agreement accumulation
            for d in dim_ids:
                c_s, ds_s = cs.get(d), m["deepseek_dim_scores"].get(d)
                if c_s is not None and ds_s is not None:
                    abs_diffs.append(abs(c_s - ds_s))
                    claude_flat.append(c_s)
                    ds_flat.append(ds_s)

        mean_total = {c: mean(cond_totals[c]) for c in CONDITIONS}
        gap = None
        if mean_total["full"] is not None and mean_total["style_only"] is not None:
            gap = round(mean_total["full"] - mean_total["style_only"], 3)
        # forced-choice-style: full beats style_only per probe?
        wins = [1 for pid, t in by_probe.items()
                if t.get("full") is not None and t.get("style_only") is not None and t["full"] > t["style_only"]]
        decided = [pid for pid, t in by_probe.items()
                   if t.get("full") is not None and t.get("style_only") is not None]
        report["personas"].append({
            "persona": persona,
            "claude_mean_thinking_total": mean_total,
            "claude_headline_gap_full_minus_style_only": gap,
            "claude_full_beats_style_rate": round(len(wins) / len(decided), 3) if decided else None,
            "claude_mean_dim_scores": {c: {d: mean(cond_dim[c][d]) for d in dim_ids} for c in CONDITIONS},
        })

    # cross-judge agreement
    n = len(abs_diffs)
    within1 = sum(1 for d in abs_diffs if d <= 1) / n if n else None
    exact = sum(1 for d in abs_diffs if d == 0) / n if n else None
    try:
        pearson = stats.correlation(claude_flat, ds_flat) if n >= 2 else None
    except Exception:
        pearson = None
    report["cross_judge_agreement_claude_vs_deepseek"] = {
        "n_dim_scores": n,
        "mean_abs_diff": round(stats.fmean(abs_diffs), 3) if n else None,
        "exact_match_rate": round(exact, 3) if exact is not None else None,
        "within_1_point_rate": round(within1, 3) if within1 is not None else None,
        "pearson_r": round(pearson, 3) if pearson is not None else None,
    }

    out = RESULTS / "cross_judge_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
