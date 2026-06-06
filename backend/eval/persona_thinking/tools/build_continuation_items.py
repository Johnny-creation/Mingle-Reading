# -*- coding: utf-8 -*-
"""Build reasoning-CONTINUATION items from held-out essays (Layer-1b ground truth).

Where Layer-1a tests whether a model can RECOGNISE the author's stance (which a
base model that has memorised the author can partly fake), Layer-1b tests whether
it can PRODUCE the author's next reasoning move from scratch — much harder to fake
by recognition. Each item gives a verbatim SETUP (the author's premise/situation)
and holds out the author's verbatim GOLD continuation (the key inference) as the
reference for reference-based judging.

We locate the split with an LLM-supplied `split_anchor` (a short verbatim phrase
that begins the author's key inference); the setup/gold spans are then sliced
VERBATIM from the essay around that anchor — the model only points at the seam,
it does not rewrite the text, so the ground truth stays the author's own words.

Run from Mingle-Reading-main/:
    python -m backend.eval.persona_thinking.tools.build_continuation_items --personas lu-xun zhang-ailing
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from backend.eval.persona_thinking.common import call_model, parse_json_loose
from backend.eval.persona_thinking.corpus import load_essays, slug

HERE = Path(__file__).resolve()
BASE = HERE.parents[1]
OUT_DIR = BASE / "continuation_items"
DISPLAY = {"lu-xun": "鲁迅", "zhang-ailing": "张爱玲"}

_SENT_END = "。！？”』」"

_SYS = (
    "你是一个严谨的中文文学标注助手。给你一篇{author}的真实文章。"
    "请找出 1-{n} 处文章中作者从‘铺垫 / 前提 / 现象’转入‘关键推断 / 判断 / 揭示’的转折点，"
    "用来测试另一个模型能否像{author}那样把分析推进下去。\n"
    "对每一处，输出：\n"
    "- split_anchor：作者**开始**那个关键推断处的前 8~16 个字，必须**逐字**来自原文（用于定位，不要改写）。\n"
    "- move：该处的推理动作简称（如‘归谬’‘历史循环’‘名实分离’‘参差对照’）。\n"
    "- gist：用一句平实白话概括作者在该处得出的判断。\n"
    "只选**真正体现作者思维特征**的转折点，宁缺毋滥。只输出 JSON："
    "{{\"items\":[{{\"split_anchor\":\"...\",\"move\":\"...\",\"gist\":\"...\"}}]}}"
)


def _norm(s: str) -> str:
    return "".join(c for c in (s or "") if ("一" <= c <= "鿿") or c.isalnum())


def _find_anchor(anchor: str, text: str) -> int:
    """Return char index of the anchor in text, robust to punctuation, else -1."""
    if anchor and anchor in text:
        return text.find(anchor)
    na = _norm(anchor)
    if len(na) < 5:
        return -1
    # map normalized positions back to raw indices
    raw_idx, norm_str = [], []
    for i, c in enumerate(text):
        if ("一" <= c <= "鿿") or c.isalnum():
            raw_idx.append(i)
            norm_str.append(c)
    j = "".join(norm_str).find(na)
    return raw_idx[j] if j != -1 else -1


def _back_to_sentence_start(text: str, pos: int, want_back: int = 260) -> int:
    """Start of the SETUP: back up ~want_back chars from the anchor, then move
    forward to the next sentence boundary so the setup begins on a clean sentence.
    This captures the PRECEDING premise sentences (not the anchor sentence itself —
    the anchor typically sits at a sentence start, so rewinding only to the current
    sentence start would give an empty setup)."""
    lo = max(0, pos - want_back)
    if lo == 0:
        return 0
    seg = text[lo:pos]
    # first sentence terminator inside the window; begin setup just after it
    firsts = [seg.find(c) for c in _SENT_END + "\n" if seg.find(c) != -1]
    return lo + min(firsts) + 1 if firsts else lo


def _forward_to_sentence_end(text: str, pos: int, want: int = 220) -> int:
    hi = min(len(text), pos + want)
    seg = text[pos:hi]
    # extend to the next sentence end after `want`
    nxt = len(text)
    for c in _SENT_END:
        k = text.find(c, hi)
        if k != -1:
            nxt = min(nxt, k + 1)
    return min(nxt, pos + want + 80)


def build_for_essay(persona: str, essay, n: int, use_cache: bool) -> list[dict]:
    author = DISPLAY[persona]
    raw = call_model(
        endpoint_key="neutral",
        system_prompt=_SYS.format(author=author, n=n),
        user_prompt=f"【作者】{author}\n【篇名】《{essay.title}》\n\n【正文】\n{essay.text[:3800]}",
        temperature=0.2,
        max_tokens=900,
        json_object=True,
        use_cache=use_cache,
        tag=f"cont::{persona}::{essay.title}::v1",
    )
    try:
        items = parse_json_loose(raw).get("items", [])
    except Exception:
        return []
    out = []
    for k, it in enumerate(items):
        anchor = (it.get("split_anchor") or "").strip()
        pos = _find_anchor(anchor, essay.text)
        if pos < 40:  # need at least a little setup before the anchor
            continue
        s0 = _back_to_sentence_start(essay.text, pos)
        g1 = _forward_to_sentence_end(essay.text, pos)
        setup = essay.text[s0:pos].strip()
        gold = essay.text[pos:g1].strip()
        if len([c for c in setup if c.strip()]) < 30 or len([c for c in gold if c.strip()]) < 20:
            continue
        out.append({
            "id": f"{persona.split('-')[0]}_{slug(essay.title)}_c{k+1}",
            "persona": persona,
            "source_title": essay.title,
            "move": (it.get("move") or "").strip(),
            "gist": (it.get("gist") or "").strip(),
            "setup": setup,
            "gold": gold,
        })
    return out


def render_review(persona: str, items: list[dict]) -> str:
    author = DISPLAY[persona]
    L = [f"# {author} 推理续写题（Layer-1b，可抽检）", "",
         f"共 {len(items)} 题。setup=作者铺垫(给模型)，gold=作者真实的下一步推理(留作参照判分)。", ""]
    for it in items:
        L += [f"### [{it['id']}]　move：{it['move']}",
              f"- **gist**：{it['gist']}",
              f"- **setup**：{it['setup']}",
              f"- **gold（参照）**：{it['gold']}", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", nargs="+", default=["lu-xun", "zhang-ailing"])
    ap.add_argument("--per-essay", type=int, default=2)
    ap.add_argument("--max-essays", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for persona in args.personas:
        essays = load_essays(persona)
        if args.max_essays:
            essays = essays[: args.max_essays]
        items = []
        for essay in essays:
            got = build_for_essay(persona, essay, args.per_essay, not args.no_cache)
            print(f"  [{persona}] 《{essay.title}》-> {len(got)} continuation items")
            items.extend(got)
        (OUT_DIR / f"{persona}.jsonl").write_text(
            "\n".join(json.dumps(it, ensure_ascii=False) for it in items), encoding="utf-8")
        (OUT_DIR / f"{persona}.review.md").write_text(render_review(persona, items), encoding="utf-8")
        print(f"[{persona}] {len(items)} continuation items -> {persona}.jsonl / .review.md")


if __name__ == "__main__":
    main()
