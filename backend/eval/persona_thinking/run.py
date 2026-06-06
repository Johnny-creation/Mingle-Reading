# -*- coding: utf-8 -*-
"""Assemble the double-dissociation report from the layer result files.

Reads:
  results/layer1b__<persona>.json       (T axis PRIMARY: continuation score 0-2)
  results/layer1_stance__<persona>.json  (T axis diagnostic: held-out MCQ accuracy)
  results/layer2_style__<persona>.json    (S axis: directional style intensity)

Headline = the DOUBLE DISSOCIATION across the 2x2 (neutral / style_only /
thinking_only / full):

  S (style intensity ratio, HIGHER = more author-like style):
      full ~ style_only   >>   thinking_only ~ neutral      (moves with +style)
  T (thinking, HIGHER = reproduces the author's real reasoning):
      full ~ thinking_only >>   style_only ~ neutral          (moves with +thinking)

If both hold, the style gain cannot explain the thinking gain and vice versa.

T-axis evidence, primary vs diagnostic — and WHY:
  * PRIMARY = Layer-1b reasoning CONTINUATION (production). Asked to continue the
    author's setup, the agent must PRODUCE the author's next inferential move;
    this is scored against the author's real verbatim continuation. Production
    resists the base-model recognition ceiling.
  * DIAGNOSTIC = Layer-1a stance MCQ (recognition). A DeepSeek base that has read
    the author answers most stance MCQs correctly even with NO persona (neutral
    ≈ ceiling), so MCQ has little headroom to show a persona effect. That ceiling
    is itself the reason we lead with production, not recognition; the MCQ numbers
    (esp. the tiny `distinctive` = neutral-wrong subset) are reported for honesty,
    not as the load-bearing claim.

Every headline contrast carries a 95% bootstrap CI and a paired permutation
p-value (stats.py). The wrong_persona null control guards against "any elaborate
persona prompt helps."

Run from Mingle-Reading-main/:
    python -m backend.eval.persona_thinking.run --personas lu-xun zhang-ailing
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.eval.persona_thinking import stats
from backend.eval.persona_thinking.conditions import CONDITIONS, CONTROL_CONDITIONS

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
ALL_CONDITIONS = list(CONDITIONS) + list(CONTROL_CONDITIONS)
DISPLAY = {"lu-xun": "鲁迅", "zhang-ailing": "张爱玲"}

# Contrasts shared by both T sources (higher = better). diff = mean(a) - mean(b).
_T_PAIRS = {
    "full_vs_neutral": ("full", "neutral"),
    "thinking_only_vs_neutral": ("thinking_only", "neutral"),
    "full_vs_style_only": ("full", "style_only"),
    "thinking_only_vs_style_only": ("thinking_only", "style_only"),
    "full_vs_wrong_persona": ("full", "wrong_persona"),
}
# Style contrasts (higher ratio = more author-like). diff = mean(a) - mean(b),
# positive => a more author-like.
_S_PAIRS = {
    "style_only_vs_neutral": ("style_only", "neutral"),
    "full_vs_neutral": ("full", "neutral"),
    "full_vs_thinking_only": ("full", "thinking_only"),
    "thinking_only_vs_neutral": ("thinking_only", "neutral"),
}


def _load(name: str) -> dict | None:
    p = RESULTS_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------- per-condition paired vectors ----------

def _t_continuation_vectors(l1b: dict, judge: str) -> dict[str, list[float]]:
    """Per-item continuation score (0-2) per condition for one judge, paired by
    item. judge in {'deepseek','claude'}. Returns {} if that judge has no scores."""
    per = l1b["per_item"]
    key = f"score_{judge}"
    out: dict[str, list[float]] = {}
    for c in ALL_CONDITIONS:
        vals = [p["conditions"].get(c, {}).get(key) for p in per]
        if vals and all(v is not None for v in vals):
            out[c] = [float(v) for v in vals]
    return out


def _t_mcq_vectors(stance: dict, subset: str) -> dict[str, list[float]]:
    """Per-item 0/1 majority-correct vectors per condition (paired). subset:
    'all' or 'distinctive' (= items the neutral baseline gets wrong)."""
    items = stance["per_item"]
    if subset == "distinctive":
        items = [p for p in items if p["distinctive"]]
    return {
        c: [1.0 if p["conditions"][c]["majority_correct"] else 0.0 for p in items]
        for c in ALL_CONDITIONS
    }


def _s_vectors(style: dict) -> dict[str, list[float]]:
    """Per-probe directional style-intensity ratio per condition (paired)."""
    return {c: list(style["summary"][c]["ratios"]) for c in ALL_CONDITIONS}


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def _contrasts(vec: dict[str, list[float]], pairs: dict[str, tuple[str, str]]) -> dict[str, dict]:
    res = {}
    for name, (a, b) in pairs.items():
        if vec.get(a) and vec.get(b) and len(vec[a]) == len(vec[b]):
            res[name] = stats.contrast(name, vec[a], vec[b]).as_dict()
    return res


# ---------- per-persona analysis ----------

def analyze_persona(persona: str) -> dict[str, Any]:
    l1b = _load(f"layer1b__{persona}.json")
    stance = _load(f"layer1_stance__{persona}.json")
    style = _load(f"layer2_style__{persona}.json")
    out: dict[str, Any] = {"persona": persona, "display_name": DISPLAY[persona]}

    # T axis — PRIMARY: continuation (production)
    t_primary_vec = None
    if l1b:
        cont = {}
        for judge in ("deepseek", "claude"):
            v = _t_continuation_vectors(l1b, judge)
            if v:
                cont[judge] = {
                    "mean": {c: _mean(v[c]) for c in v},
                    "contrasts": _contrasts(v, _T_PAIRS),
                }
                if t_primary_vec is None:  # deepseek is the in-pipeline default
                    t_primary_vec = v
        out["thinking_continuation"] = {"n_items": l1b["n_items"], "judges": cont}

    # T axis — DIAGNOSTIC: stance MCQ (recognition)
    if stance:
        t_all = _t_mcq_vectors(stance, "all")
        t_dist = _t_mcq_vectors(stance, "distinctive")
        out["thinking_mcq"] = {
            "n_items": stance["n_items"],
            "n_distinctive": stance["n_distinctive"],
            "acc_all": {c: _mean(t_all[c]) for c in ALL_CONDITIONS},
            "acc_distinctive": {c: _mean(t_dist[c]) for c in ALL_CONDITIONS},
            "contrasts_all": _contrasts(t_all, _T_PAIRS),
            "contrasts_distinctive": _contrasts(t_dist, _T_PAIRS),
        }

    # S axis — style intensity ratio
    s_vec = None
    if style:
        s_vec = _s_vectors(style)
        out["style"] = {
            "n_probes": style["n_probes"],
            "author_baseline": style.get("author_style_baseline"),
            "mean_ratio": {c: _mean(s_vec[c]) for c in ALL_CONDITIONS},
            "contrasts": _contrasts(s_vec, _S_PAIRS),
        }

    # Verdict uses PRIMARY T (continuation) + S (style ratio); falls back to MCQ
    # distinctive if continuation is not yet available.
    if t_primary_vec is None and stance:
        t_primary_vec = _t_mcq_vectors(stance, "distinctive")
    if t_primary_vec and s_vec:
        out["dissociation"] = _dissociation_verdict(t_primary_vec, s_vec)
    return out


def _dissociation_verdict(t: dict[str, list[float]], s: dict[str, list[float]]) -> dict:
    """Booleans for the two predicted patterns. T higher = more author-thinking,
    S higher = more author-style. 'flat on off-factor' = the off-factor swing is
    less than half the on-factor swing."""
    tm = {c: _mean(t[c]) for c in t}
    sm = {c: _mean(s[c]) for c in s}

    def gt(d, a, b):
        return d.get(a) is not None and d.get(b) is not None and d[a] > d[b]

    def flat(d, on_a, on_b, off_a, off_b):
        if any(d.get(k) is None for k in (on_a, on_b, off_a, off_b)):
            return None
        on = abs(d[on_a] - d[on_b])
        off = abs(d[off_a] - d[off_b])
        return off < on / 2 + 1e-9

    style_tracks_style = gt(sm, "full", "neutral") and gt(sm, "style_only", "neutral")
    style_flat_on_thinking = flat(sm, "style_only", "neutral", "thinking_only", "neutral")
    think_tracks_think = gt(tm, "full", "neutral") and gt(tm, "thinking_only", "neutral")
    think_flat_on_style = flat(tm, "full", "neutral", "style_only", "neutral")
    return {
        "style_axis_tracks_style_factor": style_tracks_style,
        "style_axis_flat_on_thinking_factor": style_flat_on_thinking,
        "thinking_axis_tracks_thinking_factor": think_tracks_think,
        "thinking_axis_flat_on_style_factor": think_flat_on_style,
        "double_dissociation_holds": bool(
            style_tracks_style and think_tracks_think
            and style_flat_on_thinking and think_flat_on_style
        ),
    }


# ---------- markdown rendering ----------

def _ctab(title: str, contrasts: dict, unit: str) -> list[str]:
    L = [f"### {title}", "", f"| 对比 | n | diff({unit}) | 95%CI | p | 显著 |",
         "|---|---|---|---|---|---|"]
    for name, c in contrasts.items():
        L.append(f"| {name} | {c['n']} | {c['diff']} | {c['ci95']} | {c['p_value']} | {c['significant_0.05']} |")
    L.append("")
    return L


def render_markdown(report: dict) -> str:
    L = ["# 名家思维评测 · 双重分离报告", "", f"生成时间：{report['generated_at']}", "",
         "> **核心主张**：名家 agent 复刻的是作者的**内在思维方式**，而非表层文字风格。",
         "> 把“思维”“风格”做成两个相互独立、各有客观度量的轴跑双重分离：",
         "> **S 风格强度**（标志性标记密度／作者本人密度，越高越像作者文风）只应随 **+风格** 变；",
         "> **T 思维**只应随 **+思维** 变。T 以 **Layer-1b 推理续写（产出）** 为主证据"
         "（对照作者真实的下一步推理打分，能抵抗基座模型的“识别天花板”）；",
         "> Layer-1a 立场 MCQ（识别）作诊断量并列报告——基座已读过作者，**neutral 也近满分**，"
         "MCQ 几无区分度，这恰恰是我们以“产出”而非“识别”作主证据的理由。", ""]

    for pr in report["personas"]:
        L += [f"## {pr['display_name']}（{pr['persona']}）", ""]
        tc = pr.get("thinking_continuation")
        tm = pr.get("thinking_mcq")
        st = pr.get("style")

        # headline table
        L += ["### 双重分离总表", "",
              "| 条件 | T·续写(DeepSeek,↑) | T·续写(Claude,↑) | T·MCQ全集 | T·MCQ-distinct | S·风格强度ratio(↑) |",
              "|---|---|---|---|---|---|"]
        ds = (tc or {}).get("judges", {}).get("deepseek", {}).get("mean", {})
        cl = (tc or {}).get("judges", {}).get("claude", {}).get("mean", {})
        for c in ALL_CONDITIONS:
            L.append(
                f"| {c} | {ds.get(c, '—')} | {cl.get(c, '—')} | "
                f"{(tm or {}).get('acc_all', {}).get(c, '—')} | "
                f"{(tm or {}).get('acc_distinctive', {}).get(c, '—')} | "
                f"{(st or {}).get('mean_ratio', {}).get(c, '—')} |"
            )
        notes = []
        if tc:
            notes.append(f"续写题 n={tc['n_items']}")
        if tm:
            notes.append(f"MCQ n={tm['n_items']}（distinctive={tm['n_distinctive']}）")
        if st:
            notes.append(f"风格探针 n={st['n_probes']}，作者基线密度={st.get('author_baseline')}/100字")
        L += ["", "（" + "；".join(notes) + "）", ""]

        dv = pr.get("dissociation")
        if dv:
            L += ["### 双重分离判定（基于 续写T + 风格强度S）", "",
                  f"- 风格轴随 +风格 变化：**{dv['style_axis_tracks_style_factor']}**",
                  f"- 风格轴对 +思维 基本不变：**{dv['style_axis_flat_on_thinking_factor']}**",
                  f"- 思维轴随 +思维 变化：**{dv['thinking_axis_tracks_thinking_factor']}**",
                  f"- 思维轴对 +风格 基本不变：**{dv['thinking_axis_flat_on_style_factor']}**",
                  f"- **双重分离成立：{dv['double_dissociation_holds']}**", ""]

        # primary T contrasts
        if tc:
            for judge, label in (("deepseek", "DeepSeek 同族判官"), ("claude", "Claude 跨族盲评")):
                jc = tc["judges"].get(judge)
                if jc and jc["contrasts"]:
                    L += _ctab(f"思维轴·续写主证据（{label}，diff>0=前者更接近作者推理）",
                               jc["contrasts"], "0-2")
            L += ["（full_vs_wrong_persona>0 且显著 = 增益来自**这位作者**的思维，而非“有个复杂人设”。）", ""]

        # style contrasts
        if st and st["contrasts"]:
            L += _ctab("风格轴·关键对比（diff>0=前者风格强度更高、更像作者文风）",
                       st["contrasts"], "Δratio")

        # MCQ diagnostic
        if tm:
            L += ["### 思维轴·MCQ 诊断（识别，受基座天花板限制——仅供参照）", ""]
            if tm["contrasts_distinctive"]:
                L += _ctab("distinctive 子集（neutral 答错的题）", tm["contrasts_distinctive"], "Δacc")
            L += [f"> 说明：MCQ 全集 neutral 命中率已近天花板，distinctive 子集 n="
                  f"{tm['n_distinctive']}/{tm['n_items']}，样本过小，不作主证据。", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", nargs="+", default=["lu-xun", "zhang-ailing"])
    args = ap.parse_args()
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"), "personas": []}
    for persona in args.personas:
        report["personas"].append(analyze_persona(persona))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (RESULTS_DIR / f"report_{stamp}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render_markdown(report)
    (RESULTS_DIR / f"report_{stamp}.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[run] wrote report_{stamp}.json / .md")


if __name__ == "__main__":
    main()
