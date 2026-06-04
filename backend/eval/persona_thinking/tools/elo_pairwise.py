# -*- coding: utf-8 -*-
"""ELO-ranking evaluation of the three generation conditions.

For each probe, runs all three pairwise forced-choice comparisons on style-
stripped text (full vs style_only, full vs neutral, style_only vs neutral),
then applies the Elo algorithm to produce a per-condition rating.

Reads the latest result JSON for stripped texts so no new generation or style-
stripping tokens are spent. Forced-choice calls go to DeepSeek and are cached.

Usage (from Mingle-Reading-main/):
  python backend/eval/persona_thinking/tools/elo_pairwise.py
  python backend/eval/persona_thinking/tools/elo_pairwise.py --result results/result_20260531_162758.json
  python backend/eval/persona_thinking/tools/elo_pairwise.py --k 16 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]  # Mingle-Reading-main/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.eval.persona_thinking.common import RESULTS_DIR, anchors_block
from backend.eval.persona_thinking.judge import forced_choice

CONDITIONS = ("full", "style_only", "neutral")
# All three canonical pairs; ELO needs the full triangle, not just full-vs-X.
PAIRS = [
    ("full", "style_only"),
    ("full", "neutral"),
    ("style_only", "neutral"),
]


def latest_full_result() -> Path | None:
    files = sorted(RESULTS_DIR.glob("result_*.json"))
    best, best_n = None, -1
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        n = sum(p["n_probes"] for p in d["personas"])
        if n >= best_n:
            best, best_n = f, n
    return best


def elo_update(ra: float, rb: float, winner: str, k: float = 32.0) -> tuple[float, float]:
    """Single Elo match update. winner in {'A', 'B', 'TIE'}, A=cond_a, B=cond_b."""
    ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
    eb = 1.0 - ea
    sa, sb = {"A": (1.0, 0.0), "B": (0.0, 1.0), "TIE": (0.5, 0.5)}.get(winner, (0.5, 0.5))
    return ra + k * (sa - ea), rb + k * (sb - eb)


def run_persona(pr: dict, rng: random.Random, k: float) -> dict:
    persona = pr["persona"]
    persona_name = pr["display_name"]
    anchors = anchors_block(persona)

    ratings = {c: 1000.0 for c in CONDITIONS}
    match_log: list[dict] = []

    probes = pr["per_probe"][:]
    rng.shuffle(probes)

    for p in probes:
        scenario = p["scenario"]
        stripped = {c: p["conditions"][c]["stripped"] for c in CONDITIONS}

        pairs_order = PAIRS[:]
        rng.shuffle(pairs_order)

        for cond_a, cond_b in pairs_order:
            # Randomise which goes into slot A to cancel position bias.
            if rng.random() < 0.5:
                slot_a, slot_b = cond_a, cond_b
            else:
                slot_a, slot_b = cond_b, cond_a

            res = forced_choice(
                persona_name=persona_name,
                anchors=anchors,
                scenario=scenario,
                text_a=stripped[slot_a],
                text_b=stripped[slot_b],
                tag=f"elo::{p['probe_id']}::{slot_a}_vs_{slot_b}",
            )

            # Resolve winner slot back to condition.
            winner_slot = res["winner"].upper()
            if winner_slot == "A":
                winner_cond = slot_a
            elif winner_slot == "B":
                winner_cond = slot_b
            else:
                winner_cond = "tie"

            # Elo update uses canonical pair order (cond_a / cond_b).
            if winner_cond == cond_a:
                elo_winner = "A"
            elif winner_cond == cond_b:
                elo_winner = "B"
            else:
                elo_winner = "TIE"

            ratings[cond_a], ratings[cond_b] = elo_update(ratings[cond_a], ratings[cond_b], elo_winner, k)

            match_log.append({
                "probe_id": p["probe_id"],
                "pair": f"{cond_a}_vs_{cond_b}",
                "slot_A": slot_a,
                "slot_B": slot_b,
                "winner_slot": winner_slot,
                "winner_condition": winner_cond,
                "reason": res["reason"],
                "elo_after": {c: round(ratings[c], 1) for c in CONDITIONS},
            })

    # Win-rate matrix for summary
    win_matrix: dict[str, dict[str, dict]] = {c: {} for c in CONDITIONS}
    for cond_a, cond_b in PAIRS:
        matches = [m for m in match_log if m["pair"] == f"{cond_a}_vs_{cond_b}"]
        wins_a = sum(1 for m in matches if m["winner_condition"] == cond_a)
        wins_b = sum(1 for m in matches if m["winner_condition"] == cond_b)
        ties = sum(1 for m in matches if m["winner_condition"] == "tie")
        n = len(matches)
        win_matrix[cond_a][cond_b] = {
            "wins": wins_a, "losses": wins_b, "ties": ties, "n": n,
            "win_rate": round(wins_a / n, 3) if n else None,
        }
        win_matrix[cond_b][cond_a] = {
            "wins": wins_b, "losses": wins_a, "ties": ties, "n": n,
            "win_rate": round(wins_b / n, 3) if n else None,
        }

    return {
        "persona": persona,
        "display_name": persona_name,
        "n_probes": len(probes),
        "n_matches": len(match_log),
        "final_elo": {c: round(ratings[c], 1) for c in CONDITIONS},
        "elo_gap_full_minus_style_only": round(ratings["full"] - ratings["style_only"], 1),
        "win_matrix": win_matrix,
        "match_log": match_log,
    }


def render_md(report: dict) -> str:
    lines = ["# ELO 成对比较评测结果", "", f"数据源：{report['source']}", ""]
    lines += [
        "## Elo 评分总览",
        "",
        "初始 Elo = 1000，K-factor = " + str(report["k_factor"]) + "；",
        "全部三组成对比较（full/style_only/neutral 两两相比），随机对局顺序。",
        "",
        "| 名家 | full | style_only | neutral | gap(full-style) |",
        "|---|---|---|---|---|",
    ]
    for pr in report["personas"]:
        e = pr["final_elo"]
        lines.append(f"| {pr['display_name']} | {e['full']} | {e['style_only']} | {e['neutral']} | {pr['elo_gap_full_minus_style_only']} |")

    lines += ["", "## 各名家成对胜率矩阵", ""]
    for pr in report["personas"]:
        lines.append(f"### {pr['display_name']}")
        lines.append("")
        lines.append("| 对局 | 胜 | 负 | 平 | 胜率 |")
        lines.append("|---|---|---|---|---|")
        for cond_a, cond_b in PAIRS:
            m = pr["win_matrix"][cond_a][cond_b]
            lines.append(f"| {cond_a} vs {cond_b} | {m['wins']} | {m['losses']} | {m['ties']} | {m['win_rate']} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, default=None, help="Path to result JSON; default: latest")
    ap.add_argument("--k", type=float, default=32.0, help="Elo K-factor (default 32)")
    ap.add_argument("--seed", type=int, default=20260601)
    args = ap.parse_args()

    src = args.result or latest_full_result()
    if src is None:
        print("No result_*.json found in results/. Run run.py first.")
        return

    data = json.loads(src.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)

    report: dict = {"source": src.name, "k_factor": args.k, "personas": []}
    for pr in data["personas"]:
        n_matches = pr["n_probes"] * 3
        print(f"[elo] {pr['display_name']} — {pr['n_probes']} probes × 3 pairs = {n_matches} matches", flush=True)
        result = run_persona(pr, rng, args.k)
        report["personas"].append(result)
        e = result["final_elo"]
        print(f"  Elo:  full={e['full']}  style_only={e['style_only']}  neutral={e['neutral']}")
        print(f"  gap(full-style_only) = {result['elo_gap_full_minus_style_only']}")
        for cond_a, cond_b in PAIRS:
            m = result["win_matrix"][cond_a][cond_b]
            print(f"  {cond_a} vs {cond_b}: {m['wins']}/{m['n']} win_rate={m['win_rate']}")

    out_json = RESULTS_DIR / "elo_report.json"
    out_md = RESULTS_DIR / "elo_report.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_md(report), encoding="utf-8")
    print(f"\n[elo] wrote {out_json}")
    print(f"[elo] wrote {out_md}")


if __name__ == "__main__":
    main()
