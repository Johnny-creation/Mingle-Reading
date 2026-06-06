# -*- coding: utf-8 -*-
"""Four factorial conditions + a wrong-persona null control.

The v1 design had three conditions (full / style_only / neutral). v2 adds the
missing 2x2 cell, `thinking_only`, so the experiment can attribute any gap to
*thinking* rather than to "a longer/more elaborate system prompt":

                       - style              + style
        - thinking   neutral              style_only
        + thinking   thinking_only        full

`thinking_only` = the SKILL's cognitive / value / methodology content with the
explicitly-labelled STYLE sections removed AND a max-priority plain-register
override appended (the override is load-bearing: the 心智模型 itself mandates
文白夹杂, so deletion alone would not yield plain output). We measure style with
Burrows's Delta to VERIFY thinking_only actually comes out plain.

`wrong_persona` (null control) runs the *other* author's full SKILL on this
author's items: if the Layer-1 gain came from "having an elaborate persona
prompt" rather than from this specific author's thinking, the wrong persona
would score just as well. It should not.
"""
from __future__ import annotations

import re

from backend.eval.persona_thinking.common import skill_body
from backend.eval.persona_thinking.conditions import (
    NEUTRAL_PROMPT,
    STYLE_ONLY_PROMPTS,
    _LEN_HINT,
)

# Core 2x2 conditions (order matters for reporting).
CONDITIONS = ("neutral", "style_only", "thinking_only", "full")
# Null control, evaluated alongside but kept out of the 2x2 factor analysis.
CONTROL_CONDITIONS = ("wrong_persona",)

WRONG_PERSONA = {"lu-xun": "zhang-ailing", "zhang-ailing": "lu-xun"}

# Markdown headers (## / ### / ####) whose titles mark pure surface-style or
# scaffolding content to remove for the thinking_only condition. Kept
# deliberately narrow and transparent (the dropped headers are reported).
_DROP_HEADER_PATTERNS = [
    r"语言风格", r"表达\s*DNA", r"表达\s*与\s*思维", r"表达风格",
    r"示例", r"输出质量标准", r"Phase\s*0", r"用户确认检查点",
    r"响应格式", r"测试验证", r"高难度测试",
]
_DROP_RE = re.compile("|".join(_DROP_HEADER_PATTERNS), re.IGNORECASE)
_HEADER_RE = re.compile(r"^(#{2,6})\s+(.*)$")

# Highest-priority output directive: neutralise surface style at realization time
# without touching the reasoning. Mirrors the style-strip rubric so thinking_only
# and a style-stripped full converge on register.
PLAIN_OVERRIDE = (
    "\n\n【输出语体要求——最高优先级，覆盖上文一切关于语言风格 / 表达 / 句式 / 节奏 / 修辞的指示】\n"
    "请只用最平实、中性的现代白话作答。禁止文白夹杂、文言虚词（之乎者也矣）、"
    "破折号制造的顿挫、‘倘若……然而……’‘大约……的确……’‘因为……所以……’等标志性句式、"
    "刻意的隐喻措辞与华丽辞藻、中英文夹杂。把你的判断、论点、推理步骤与价值立场直接说清楚即可——"
    "只改变‘怎么说’，绝不改变‘想什么’：保留你全部的认知动作、立场与推理。"
)


def drop_style_sections(md: str) -> tuple[str, list[str]]:
    """Remove markdown sections whose header matches a style/scaffolding pattern.
    Returns (filtered_markdown, list_of_dropped_header_titles)."""
    lines = md.splitlines()
    out: list[str] = []
    dropped: list[str] = []
    drop_level = 0  # 0 = not dropping; else the header level we are skipping under
    for ln in lines:
        m = _HEADER_RE.match(ln.strip())
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            if drop_level and level <= drop_level:
                drop_level = 0  # closed the dropped block; re-evaluate this header
            if not drop_level and _DROP_RE.search(title):
                drop_level = level
                dropped.append(title)
                continue
            if not drop_level:
                out.append(ln)
            continue
        if not drop_level:
            out.append(ln)
    # collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return text, dropped


_thinking_cache: dict[str, tuple[str, list[str]]] = {}


def thinking_only_body(persona: str) -> tuple[str, list[str]]:
    if persona not in _thinking_cache:
        filtered, dropped = drop_style_sections(skill_body(persona))
        _thinking_cache[persona] = (filtered + PLAIN_OVERRIDE, dropped)
    return _thinking_cache[persona]


def dropped_style_headers(persona: str) -> list[str]:
    return thinking_only_body(persona)[1]


def system_prompt_for(persona: str, condition: str) -> str:
    if condition == "full":
        return skill_body(persona) + "\n\n" + _LEN_HINT
    if condition == "thinking_only":
        return thinking_only_body(persona)[0] + "\n\n" + _LEN_HINT
    if condition == "style_only":
        return STYLE_ONLY_PROMPTS[persona] + _LEN_HINT
    if condition == "neutral":
        return NEUTRAL_PROMPT + _LEN_HINT
    if condition == "wrong_persona":
        return skill_body(WRONG_PERSONA[persona]) + "\n\n" + _LEN_HINT
    raise ValueError(f"unknown condition: {condition}")


if __name__ == "__main__":
    for p in ("lu-xun", "zhang-ailing"):
        body, dropped = thinking_only_body(p)
        full_len = len(skill_body(p))
        print(f"\n=== {p} ===")
        print(f"full SKILL chars: {full_len}")
        print(f"thinking_only chars: {len(body)} (dropped {full_len - len(body) + len(PLAIN_OVERRIDE)} net)")
        print(f"dropped style/scaffold headers ({len(dropped)}):")
        for d in dropped:
            print(f"   - {d}")
