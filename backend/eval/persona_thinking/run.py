# -*- coding: utf-8 -*-
"""Persona-thinking evaluation orchestrator.

Pipeline per (persona, probe):
  1. generate `full` / `style_only` / `neutral` outputs (raw).
  2. style-strip each output to plain register.
  3. judge the thinking rubric on each STRIPPED output (N runs), grounded in
     real corpus excerpts, blind to condition.
  4. forced-choice: full vs style_only, full vs neutral (A/B order randomised).
  5. stylometry on the RAW outputs.

Headline result = thinking-rubric gap `full − style_only` AFTER style stripping,
plus the forced-choice win rate of `full` over `style_only`. A large positive
gap here is the evidence that the agent captures *thinking*, not just style.

Usage (from Mingle-Reading-main/):
  python -m backend.eval.persona_thinking.run --personas lu-xun zhang-ailing
  python -m backend.eval.persona_thinking.run --personas lu-xun --max-probes 2 --judge-runs 1
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as stats
from datetime import datetime
from typing import Any

from .common import (
    RESULTS_DIR,
    anchors_block,
    load_probes,
    load_rubric,
)
from .conditions import CONDITIONS, generate
from .judge import forced_choice, judge_rubric
from .style_strip import strip_style
from .stylometry import stylometry


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def evaluate_persona(
    persona: str,
    *,
    max_probes: int | None,
    judge_runs: int,
    use_cache: bool,
    rng: random.Random,
) -> dict[str, Any]:
    rubric = load_rubric(persona)
    persona_name = rubric["display_name"]
    anchors = anchors_block(persona)
    dim_ids = [d["id"] for d in rubric["dimensions"]]
    probes = load_probes(persona)
    if max_probes:
        probes = probes[:max_probes]

    per_probe: list[dict[str, Any]] = []
    for probe in probes:
        scenario = probe["prompt"]
        cond_data: dict[str, Any] = {}
        stripped_by_cond: dict[str, str] = {}
        for cond in CONDITIONS:
            raw = generate(persona, probe, cond, use_cache=use_cache)
            stripped = strip_style(raw, use_cache=use_cache, tag=f"{persona}::{probe['id']}::{cond}")
            stripped_by_cond[cond] = stripped
            runs = []
            for r in range(judge_runs):
                runs.append(
                    judge_rubric(
                        persona_name=persona_name,
                        rubric=rubric,
                        anchors=anchors,
                        scenario=scenario,
                        stripped_text=stripped,
                        run_idx=r,
                    )
                )
            # per-dimension mean across runs
            dim_means = {
                did: _mean([run[did]["score"] for run in runs if run[did]["score"] is not None])
                for did in dim_ids
            }
            total = _mean([v for v in dim_means.values() if v is not None])
            cond_data[cond] = {
                "raw": raw,
                "stripped": stripped,
                "stylometry": stylometry(raw, persona),
                "rubric_runs": runs,
                "dim_means": dim_means,
                "thinking_total": total,
            }

        # forced choice (randomised A/B order), blind
        fc: dict[str, Any] = {}
        for opponent in ("style_only", "neutral"):
            swap = rng.random() < 0.5
            a, b = ("full", opponent) if not swap else (opponent, "full")
            res = forced_choice(
                persona_name=persona_name,
                anchors=anchors,
                scenario=scenario,
                text_a=stripped_by_cond[a],
                text_b=stripped_by_cond[b],
                tag=f"{probe['id']}::full_vs_{opponent}",
            )
            winner_cond = a if res["winner"] == "A" else (b if res["winner"] == "B" else "tie")
            fc[f"full_vs_{opponent}"] = {
                "slot_A": a,
                "slot_B": b,
                "winner_slot": res["winner"],
                "winner_condition": winner_cond,
                "full_wins": winner_cond == "full",
                "reason": res["reason"],
            }

        per_probe.append(
            {
                "probe_id": probe["id"],
                "scenario": scenario,
                "target_dims": probe.get("target_dims", []),
                "conditions": cond_data,
                "forced_choice": fc,
            }
        )

    return _aggregate(persona, persona_name, rubric, dim_ids, per_probe)


def _aggregate(persona, persona_name, rubric, dim_ids, per_probe) -> dict[str, Any]:
    # thinking totals per condition
    totals = {c: [] for c in CONDITIONS}
    dim_by_cond = {c: {d: [] for d in dim_ids} for c in CONDITIONS}
    style_by_cond: dict[str, dict[str, list[float]]] = {c: {} for c in CONDITIONS}
    consistency: list[float] = []

    for p in per_probe:
        for c in CONDITIONS:
            cd = p["conditions"][c]
            if cd["thinking_total"] is not None:
                totals[c].append(cd["thinking_total"])
            for d in dim_ids:
                if cd["dim_means"][d] is not None:
                    dim_by_cond[c][d].append(cd["dim_means"][d])
            for k, v in cd["stylometry"].items():
                if isinstance(v, (int, float)):
                    style_by_cond[c].setdefault(k, []).append(float(v))
            # judge self-consistency: std of total across runs for this cell
            run_totals = [
                _mean([run[d]["score"] for d in dim_ids if run[d]["score"] is not None])
                for run in cd["rubric_runs"]
            ]
            run_totals = [t for t in run_totals if t is not None]
            if len(run_totals) >= 2:
                consistency.append(round(stats.pstdev(run_totals), 3))

    mean_total = {c: _mean(totals[c]) for c in CONDITIONS}
    mean_dim = {c: {d: _mean(dim_by_cond[c][d]) for d in dim_ids} for c in CONDITIONS}
    mean_style = {c: {k: _mean(v) for k, v in feats.items()} for c, feats in style_by_cond.items()}

    # forced-choice win rates for `full`
    def win_rate(key: str) -> dict[str, Any]:
        wins = [p["forced_choice"][key]["full_wins"] for p in per_probe if key in p["forced_choice"]]
        decided = [p["forced_choice"][key] for p in per_probe if key in p["forced_choice"]]
        ties = sum(1 for d in decided if d["winner_condition"] == "tie")
        return {
            "n": len(decided),
            "full_wins": sum(1 for w in wins if w),
            "ties": ties,
            "full_win_rate": _ratio(sum(1 for w in wins if w), len(decided)),
        }

    headline_gap = None
    if mean_total["full"] is not None and mean_total["style_only"] is not None:
        headline_gap = round(mean_total["full"] - mean_total["style_only"], 3)

    return {
        "persona": persona,
        "display_name": persona_name,
        "n_probes": len(per_probe),
        "mean_thinking_total": mean_total,
        "headline_gap_full_minus_style_only": headline_gap,
        "mean_dim_scores": mean_dim,
        "forced_choice": {
            "full_vs_style_only": win_rate("full_vs_style_only"),
            "full_vs_neutral": win_rate("full_vs_neutral"),
        },
        "mean_stylometry": mean_style,
        "judge_self_consistency_std_mean": _mean(consistency),
        "per_probe": per_probe,
    }


def _ratio(n: int, d: int) -> float:
    return round(n / d, 3) if d else 0.0


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# 名家思维评测结果", "", f"生成时间：{report['generated_at']}", ""]
    for pr in report["personas"]:
        mt = pr["mean_thinking_total"]
        lines += [
            f"## {pr['display_name']}（{pr['persona']}） — {pr['n_probes']} 题",
            "",
            "### 核心结论：思维分（风格剥离后，1-5）",
            "",
            "| 条件 | 思维总均分 |",
            "|---|---|",
            f"| full（思维+风格） | {mt['full']} |",
            f"| style_only（仅风格） | {mt['style_only']} |",
            f"| neutral（无人设） | {mt['neutral']} |",
            "",
            f"**full − style_only = {pr['headline_gap_full_minus_style_only']}**　"
            "（>0 且明显，说明剥离风格后思维仍更贴近名家 = 捕捉到的是思维而非风格）",
            "",
            "### 强制选择（盲评、A/B 随机）",
            "",
            f"- full vs style_only：full 胜率 **{pr['forced_choice']['full_vs_style_only']['full_win_rate']}**"
            f"（{pr['forced_choice']['full_vs_style_only']['full_wins']}/{pr['forced_choice']['full_vs_style_only']['n']}，"
            f"平 {pr['forced_choice']['full_vs_style_only']['ties']}）",
            f"- full vs neutral：full 胜率 **{pr['forced_choice']['full_vs_neutral']['full_win_rate']}**"
            f"（{pr['forced_choice']['full_vs_neutral']['full_wins']}/{pr['forced_choice']['full_vs_neutral']['n']}）",
            "",
            "### 各思维维度均分（full / style_only / neutral）",
            "",
            "| 维度 | full | style_only | neutral |",
            "|---|---|---|---|",
        ]
        for d in pr["mean_dim_scores"]["full"]:
            lines.append(
                f"| {d} | {pr['mean_dim_scores']['full'][d]} | "
                f"{pr['mean_dim_scores']['style_only'][d]} | {pr['mean_dim_scores']['neutral'][d]} |"
            )
        lines += [
            "",
            f"裁判自一致性（多次评分总分标准差均值，越小越稳）：{pr['judge_self_consistency_std_mean']}",
            "",
            "### 风格被控制的证据（原始输出的风格特征均值）",
            "",
            "若 full 与 style_only 的风格特征接近、而二者都明显高于 neutral，"
            "则说明风格已被复制并在两条件间大致持平，思维分的差距不能用风格差异解释。",
            "",
        ]
        feats = sorted({k for c in CONDITIONS for k in pr["mean_stylometry"][c]})
        lines.append("| 风格特征 | full | style_only | neutral |")
        lines.append("|---|---|---|---|")
        for k in feats:
            row = pr["mean_stylometry"]
            lines.append(f"| {k} | {row['full'].get(k)} | {row['style_only'].get(k)} | {row['neutral'].get(k)} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", nargs="+", default=["lu-xun", "zhang-ailing"])
    ap.add_argument("--max-probes", type=int, default=None)
    ap.add_argument("--judge-runs", type=int, default=1, help="重复评分次数，>1 才能算自一致性")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--seed", type=int, default=20260531)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "personas": [],
    }
    for persona in args.personas:
        print(f"[run] evaluating {persona} ...", flush=True)
        report["personas"].append(
            evaluate_persona(
                persona,
                max_probes=args.max_probes,
                judge_runs=args.judge_runs,
                use_cache=not args.no_cache,
                rng=rng,
            )
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"result_{stamp}.json"
    md_path = RESULTS_DIR / f"result_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"[run] wrote {json_path}")
    print(f"[run] wrote {md_path}")
    # console headline
    for pr in report["personas"]:
        print(
            f"  {pr['display_name']}: full={pr['mean_thinking_total']['full']} "
            f"style_only={pr['mean_thinking_total']['style_only']} "
            f"neutral={pr['mean_thinking_total']['neutral']} "
            f"| gap(full-style)={pr['headline_gap_full_minus_style_only']} "
            f"| full>style win={pr['forced_choice']['full_vs_style_only']['full_win_rate']}"
        )


if __name__ == "__main__":
    main()
