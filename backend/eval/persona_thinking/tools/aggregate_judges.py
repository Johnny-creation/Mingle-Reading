# -*- coding: utf-8 -*-
"""Aggregate blind scores from multiple cross-family judges into an ensemble.

Judges currently supported:
  - DeepSeek  — scores live in claude_judge_mapping.json (built-in from run.py)
  - Claude    — claude_scores.json  (scored in-conversation, pasted back manually)
  - GPT       — gpt_scores.json    (scored via ChatGPT/Codex, pasted back manually)

Score file format (same for Claude and GPT):
  {
    "IT000": {"LX1_名实分离": 4, "LX2_二难推理": 3, ...},
    "IT001": {...},
    ...
  }
  Keys are item_ids from the judge packet; values are {dim_id: int(1-5)}.

Ensemble = mean of all available judge scores per (item, dim).

Usage (from Mingle-Reading-main/):
  python backend/eval/persona_thinking/tools/aggregate_judges.py

Outputs (results/):
  ensemble_report.json  — per-judge + ensemble headlines and agreement stats
  ENSEMBLE_SUMMARY.md   — readable markdown
"""
from __future__ import annotations

import json
import statistics as stats
from pathlib import Path

HERE = Path(__file__).resolve()
PKG = HERE.parents[1]
RESULTS = PKG / "results"
CONDITIONS = ("full", "style_only", "neutral")


def mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def load_scores(path: Path) -> dict[str, dict[str, float]] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Normalise: values may be int or float, nested or flat
    out: dict[str, dict[str, float]] = {}
    for iid, dims in raw.items():
        if isinstance(dims, dict):
            out[iid] = {k: float(v) for k, v in dims.items() if v is not None}
    return out


def agreement_stats(scores_a: list[float], scores_b: list[float]) -> dict:
    n = len(scores_a)
    if n == 0:
        return {"n": 0}
    abs_diffs = [abs(a - b) for a, b in zip(scores_a, scores_b)]
    try:
        pearson = stats.correlation(scores_a, scores_b) if n >= 2 else None
    except Exception:
        pearson = None
    return {
        "n": n,
        "mean_abs_diff": round(stats.fmean(abs_diffs), 3),
        "exact_match_rate": round(sum(1 for d in abs_diffs if d == 0) / n, 3),
        "within_1_point_rate": round(sum(1 for d in abs_diffs if d <= 1) / n, 3),
        "pearson_r": round(pearson, 3) if pearson is not None else None,
    }


def main() -> None:
    mapping = json.loads((RESULTS / "claude_judge_mapping.json").read_text(encoding="utf-8"))

    # Load all available external-judge score files
    judge_files = {
        "claude": RESULTS / "claude_scores.json",
        "gpt": RESULTS / "gpt_scores.json",
    }
    judge_scores: dict[str, dict[str, dict[str, float]]] = {}
    for jname, path in judge_files.items():
        loaded = load_scores(path)
        if loaded is not None:
            judge_scores[jname] = loaded
            print(f"[aggregate] loaded {jname}: {len(loaded)} items from {path.name}")
        else:
            print(f"[aggregate] {path.name} not found — skipping {jname}")

    available_judges = ["deepseek"] + list(judge_scores.keys())
    print(f"[aggregate] judges: {available_judges}")

    personas = sorted({m["persona"] for m in mapping.values()})
    per_persona = []

    for persona in personas:
        items = {iid: m for iid, m in mapping.items() if m["persona"] == persona}
        dim_ids = list(next(iter(items.values()))["deepseek_dim_scores"].keys())

        # Per-condition accumulators: judge_name → condition → [total_score]
        cond_totals: dict[str, dict[str, list]] = {j: {c: [] for c in CONDITIONS} for j in available_judges + ["ensemble"]}
        cond_dim_scores: dict[str, dict[str, dict[str, list]]] = {
            j: {c: {d: [] for d in dim_ids} for c in CONDITIONS}
            for j in available_judges + ["ensemble"]
        }

        # For agreement: accumulate (judge_a_score, judge_b_score) pairs per dim
        all_pairs_flat: dict[str, tuple[list, list]] = {}

        for iid, m in items.items():
            cond = m["condition"]
            ds_dims = m["deepseek_dim_scores"]

            # Gather all judge dim scores for this item
            item_judge_dims: dict[str, dict[str, float | None]] = {
                "deepseek": {d: ds_dims.get(d) for d in dim_ids}
            }
            for jname, jscores in judge_scores.items():
                item_dims = jscores.get(iid, {})
                item_judge_dims[jname] = {d: item_dims.get(d) for d in dim_ids}

            # Ensemble = mean of available scores per dim
            ensemble_dims: dict[str, float | None] = {}
            for d in dim_ids:
                vals = [item_judge_dims[j][d] for j in available_judges if item_judge_dims[j].get(d) is not None]
                ensemble_dims[d] = mean(vals)
            item_judge_dims["ensemble"] = ensemble_dims

            # Accumulate per-judge totals and per-dim
            for j in available_judges + ["ensemble"]:
                jd = item_judge_dims[j]
                tot = mean([jd[d] for d in dim_ids])
                if tot is not None:
                    cond_totals[j][cond].append(tot)
                for d in dim_ids:
                    if jd.get(d) is not None:
                        cond_dim_scores[j][cond][d].append(jd[d])

            # Accumulate agreement pairs (all judge combinations)
            for d in dim_ids:
                for j1 in available_judges:
                    for j2 in available_judges:
                        if j1 >= j2:
                            continue
                        s1 = item_judge_dims[j1].get(d)
                        s2 = item_judge_dims[j2].get(d)
                        if s1 is not None and s2 is not None:
                            pair_key = f"{j1}_vs_{j2}"
                            if pair_key not in all_pairs_flat:
                                all_pairs_flat[pair_key] = ([], [])
                            all_pairs_flat[pair_key][0].append(s1)
                            all_pairs_flat[pair_key][1].append(s2)

        # Summarise per-judge per-condition
        judge_summary: dict[str, dict] = {}
        for j in available_judges + ["ensemble"]:
            cond_means = {c: mean(cond_totals[j][c]) for c in CONDITIONS}
            gap = None
            if cond_means["full"] is not None and cond_means["style_only"] is not None:
                gap = round(cond_means["full"] - cond_means["style_only"], 3)
            judge_summary[j] = {
                "mean_thinking_total": cond_means,
                "headline_gap_full_minus_style_only": gap,
                "mean_dim_scores": {
                    c: {d: mean(cond_dim_scores[j][c][d]) for d in dim_ids}
                    for c in CONDITIONS
                },
            }

        per_persona.append({
            "persona": persona,
            "n_items": len(items),
            "judges": judge_summary,
            "inter_judge_agreement": {k: agreement_stats(v[0], v[1]) for k, v in all_pairs_flat.items()},
        })

    report = {
        "available_judges": available_judges,
        "personas": per_persona,
    }
    out_json = RESULTS / "ensemble_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = _render_md(report)
    out_md = RESULTS / "ENSEMBLE_SUMMARY.md"
    out_md.write_text(md, encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[aggregate] wrote {out_json}")
    print(f"[aggregate] wrote {out_md}")


def _render_md(report: dict) -> str:
    judges = report["available_judges"]
    lines = [
        "# 多裁判集成评测结果（Ensemble）",
        "",
        f"参与裁判：{', '.join(judges)}",
        "集成方式：各裁判各维度分取算数均值。",
        "",
        "## 各裁判 + 集成：思维总均分",
        "",
    ]

    # Header: judge columns
    header_judges = judges + ["ensemble"]
    col_header = "| 名家 | 条件 | " + " | ".join(j.capitalize() for j in header_judges) + " |"
    separator = "|---|---|" + "|---|" * len(header_judges)
    lines += [col_header, separator]

    for pr in report["personas"]:
        persona_label = pr["persona"]
        for cond in CONDITIONS:
            scores = []
            for j in header_judges:
                v = pr["judges"].get(j, {}).get("mean_thinking_total", {}).get(cond)
                scores.append(str(v) if v is not None else "—")
            lines.append(f"| {persona_label} | {cond} | " + " | ".join(scores) + " |")
    lines.append("")

    lines += ["## headline gap：full − style_only（风格剥离后思维增益）", "", "| 名家 | " + " | ".join(j.capitalize() for j in header_judges) + " |", "|---|" + "|---|" * len(header_judges)]
    for pr in report["personas"]:
        gaps = []
        for j in header_judges:
            g = pr["judges"].get(j, {}).get("headline_gap_full_minus_style_only")
            gaps.append(str(g) if g is not None else "—")
        lines.append(f"| {pr['persona']} | " + " | ".join(gaps) + " |")
    lines.append("")

    lines += ["## 裁判间一致性", ""]
    for pr in report["personas"]:
        lines.append(f"### {pr['persona']}")
        lines.append("")
        lines.append("| 裁判对 | n | 平均绝对差 | ±1内 | 完全一致 | Pearson r |")
        lines.append("|---|---|---|---|---|---|")
        for pair_key, ag in pr["inter_judge_agreement"].items():
            n = ag.get("n", 0)
            if n == 0:
                continue
            lines.append(
                f"| {pair_key} | {n} | {ag.get('mean_abs_diff')} | "
                f"{ag.get('within_1_point_rate')} | {ag.get('exact_match_rate')} | {ag.get('pearson_r')} |"
            )
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
