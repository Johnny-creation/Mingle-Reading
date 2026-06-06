# -*- coding: utf-8 -*-
"""2x2 factorial MAIN EFFECTS for the Layer-1b continuation (T) and style (S) axes.

The conditions form a 2x2 over two factors:
    STYLE  (off/on)  x  THINKING (off/on)
    neutral       = style off, thinking off
    style_only    = style on,  thinking off
    thinking_only = style off, thinking on
    full          = style on,  thinking on

Pairwise cell contrasts (full vs neutral, etc.) waste power: with n≈30-39 items and
strict 0/1/2 scoring piled at 0, no single cell pair reaches significance. The
correct, more powerful test for a 2x2 is the MAIN EFFECT of each factor, which
pools the two cells on each side (doubling the per-side n) and is paired by item:

    THINKING effect_i = (full_i + thinking_only_i)/2 - (neutral_i + style_only_i)/2
    STYLE    effect_i = (style_only_i + full_i)/2   - (neutral_i + thinking_only_i)/2

The DOUBLE DISSOCIATION, stated as main effects:
    T axis (continuation): THINKING effect > 0 and significant; STYLE effect ~ 0
    S axis (style ratio):  STYLE effect   > 0 and significant; THINKING effect ~ 0

Each effect is tested with the same paired bootstrap CI + paired permutation p as
stats.py (here the "b" arm is the off-side mean, so contrast(a=on, b=off)). We
report per persona AND pooled across personas (same hypothesis, more power), for
both the in-pipeline DeepSeek judge and, where available, the cross-family Claude
judge. wrong_persona is reported separately as full - wrong_persona (the null:
gains come from THIS author's thinking, not from "an elaborate persona prompt").

Run from Mingle-Reading-main/:
    python -m backend.eval.persona_thinking.factorial_effects --personas lu-xun zhang-ailing
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.eval.persona_thinking import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DISPLAY = {"lu-xun": "鲁迅", "zhang-ailing": "张爱玲"}


def _load(name: str) -> dict | None:
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _t_cells(l1b: dict, judge: str) -> dict[str, list[float]] | None:
    """Per-item continuation score per condition for one judge; None if that judge
    didn't score every item (so the paired vectors stay aligned)."""
    per = l1b["per_item"]
    key = f"score_{judge}"
    out: dict[str, list[float]] = {}
    for c in ("neutral", "style_only", "thinking_only", "full", "wrong_persona"):
        vals = [p["conditions"].get(c, {}).get(key) for p in per]
        if not vals or any(v is None for v in vals):
            return None
        out[c] = [float(v) for v in vals]
    return out


def _s_cells(style: dict) -> dict[str, list[float]]:
    return {c: list(style["summary"][c]["ratios"])
            for c in ("neutral", "style_only", "thinking_only", "full", "wrong_persona")}


def _factor_vectors(cells: dict[str, list[float]]) -> dict[str, tuple[list[float], list[float]]]:
    """Return {effect_name: (on_arm, off_arm)} paired per item."""
    n = len(cells["neutral"])
    think_on = [(cells["full"][i] + cells["thinking_only"][i]) / 2 for i in range(n)]
    think_off = [(cells["neutral"][i] + cells["style_only"][i]) / 2 for i in range(n)]
    style_on = [(cells["style_only"][i] + cells["full"][i]) / 2 for i in range(n)]
    style_off = [(cells["neutral"][i] + cells["thinking_only"][i]) / 2 for i in range(n)]
    return {
        "thinking_main_effect": (think_on, think_off),
        "style_main_effect": (style_on, style_off),
        "full_vs_wrong_persona": (cells["full"], cells["wrong_persona"]),
    }


def _effects(cells: dict[str, list[float]]) -> dict[str, dict]:
    out = {}
    for name, (on, off) in _factor_vectors(cells).items():
        out[name] = stats.contrast(name, on, off).as_dict()
    return out


def _pool(cells_list: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    keys = cells_list[0].keys()
    return {k: [v for c in cells_list for v in c[k]] for k in keys}


def analyze(personas: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                              "axes": {}}
    # gather per-persona cells for each axis/judge
    t_cells = {"deepseek": {}, "claude": {}}
    s_cells = {}
    for p in personas:
        l1b = _load(f"layer1b__{p}.json")
        style = _load(f"layer2_style__{p}.json")
        if l1b:
            for judge in ("deepseek", "claude"):
                c = _t_cells(l1b, judge)
                if c:
                    t_cells[judge][p] = c
        if style:
            s_cells[p] = _s_cells(style)

    # T axis (continuation) per judge
    report["axes"]["T_continuation"] = {}
    for judge in ("deepseek", "claude"):
        jd = {}
        for p, c in t_cells[judge].items():
            jd[p] = {"n": len(c["neutral"]), "effects": _effects(c)}
        if len(t_cells[judge]) > 1:
            pooled = _pool(list(t_cells[judge].values()))
            jd["POOLED"] = {"n": len(pooled["neutral"]), "effects": _effects(pooled)}
        if jd:
            report["axes"]["T_continuation"][judge] = jd

    # S axis (style ratio)
    sd = {}
    for p, c in s_cells.items():
        sd[p] = {"n": len(c["neutral"]), "effects": _effects(c)}
    if len(s_cells) > 1:
        pooled = _pool(list(s_cells.values()))
        sd["POOLED"] = {"n": len(pooled["neutral"]), "effects": _effects(pooled)}
    report["axes"]["S_style_ratio"] = sd
    return report


def _row(name: str, e: dict) -> str:
    return (f"| {name} | {e['n']} | {e['mean_a']} | {e['mean_b']} | {e['diff']} | "
            f"{e['ci95']} | {e['p_value']} | {e['significant_0.05']} |")


def render(report: dict) -> str:
    L = ["# 名家思维评测 · 2×2 析因主效应（更高功效的正确检验）", "",
         f"生成时间：{report['generated_at']}", "",
         "> 单格两两对比（full vs neutral 等）在 n≈30–39、严判 0/1/2 多堆在 0 时功效不足。",
         "> 2×2 的正确检验是**因子主效应**：每侧合并两格（每侧 n 翻倍），按题配对。",
         "> **双重分离（主效应版）**：T·续写——思维主效应>0 且显著、风格主效应≈0；",
         "> S·风格强度——风格主效应>0 且显著、思维主效应≈0。",
         "> `thinking_main_effect = (full+thinking_only)/2 − (neutral+style_only)/2`；",
         "> `style_main_effect = (style_only+full)/2 − (neutral+thinking_only)/2`。", ""]
    axis_titles = {"T_continuation": "T 轴·推理续写（产出；diff>0=加该因子更接近作者推理）",
                   "S_style_ratio": "S 轴·方向性风格强度（diff>0=加该因子文风更像作者）"}
    for axis, title in axis_titles.items():
        ax = report["axes"].get(axis)
        if not ax:
            continue
        L += [f"## {title}", ""]
        if axis == "T_continuation":
            for judge, label in (("deepseek", "DeepSeek 同族判官（全量覆盖）"),
                                 ("claude", "Claude 跨族盲评（随机子集）")):
                jd = ax.get(judge)
                if not jd:
                    continue
                L += [f"### {label}", "",
                      "| 单元/合并 | n | on均值 | off均值 | diff | 95%CI | p | 显著 |",
                      "|---|---|---|---|---|---|---|---|"]
                for key in list(jd.keys()):
                    disp = "合并两作者" if key == "POOLED" else DISPLAY.get(key, key)
                    e = jd[key]["effects"]
                    L.append(_row(f"{disp}·思维主效应", e["thinking_main_effect"]))
                    L.append(_row(f"{disp}·风格主效应", e["style_main_effect"]))
                    L.append(_row(f"{disp}·full−wrong", e["full_vs_wrong_persona"]))
                L.append("")
        else:
            L += ["| 单元/合并 | n | on均值 | off均值 | diff | 95%CI | p | 显著 |",
                  "|---|---|---|---|---|---|---|---|"]
            for key in list(ax.keys()):
                disp = "合并两作者" if key == "POOLED" else DISPLAY.get(key, key)
                e = ax[key]["effects"]
                L.append(_row(f"{disp}·风格主效应", e["style_main_effect"]))
                L.append(_row(f"{disp}·思维主效应", e["thinking_main_effect"]))
            L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", nargs="+", default=["lu-xun", "zhang-ailing"])
    args = ap.parse_args()
    report = analyze(args.personas)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (RESULTS / f"factorial_effects_{stamp}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render(report)
    (RESULTS / f"factorial_effects_{stamp}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[factorial_effects] wrote factorial_effects_{stamp}.json / .md")


if __name__ == "__main__":
    main()
