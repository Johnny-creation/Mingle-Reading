# -*- coding: utf-8 -*-
"""Generate persona outputs under three conditions for each probe.

- full        : the actual SKILL.md (thinking model + style) the product deploys.
- style_only  : ONLY surface language style, with every thinking/value/method
                instruction removed. This is the load-bearing baseline — if
                `full` beats `style_only` on the thinking rubric *after style is
                stripped*, the gain is attributable to thinking, not style.
- neutral     : a plain assistant, no persona at all (floor reference).
"""
from __future__ import annotations

from .common import call_model, skill_body

CONDITIONS = ("full", "style_only", "neutral")

# Pure-surface style prompts. Deliberately contain NO cognitive moves, no value
# stance, no methodology — only diction / sentence shape / tone. Drawn from the
# SKILL 表达DNA sections with all 心智模型 content excluded.
STYLE_ONLY_PROMPTS = {
    "lu-xun": (
        "请用以下语言风格写一段中文评论：文白夹杂，偶尔用‘之乎者也矣’等文言虚词；"
        "可用‘倘若……然而……’‘大约……的确……’这类句式和破折号；长短句交替，短句简短有力；"
        "结尾只用句号、问号，不用感叹号；不要使用网络流行语和商业黑话。"
        "只需模仿这种语言腔调，正常地、直接地谈论用户给出的话题即可。"
    ),
    # NOTE: intentionally surface-only. Earlier versions mentioned ‘画面/意象/
    # 苍凉/物质细节’, but those are *thinking* dimensions in the rubric (物质即心理、
    # 苍凉美学) — including them leaks thought into the style baseline and collapses
    # the full−style_only gap. Kept here: diction, color lexicon, syntax, register.
    "zhang-ailing": (
        "请用以下语言风格写一段中文评论：用词华丽精致、书面化，句子偏长而绵密；"
        "可用‘因为……所以……’这类因果倒置的句式，偶尔夹一两个英文词（如 powder pink）；"
        "多用色彩词，语气克制、冷静、从容。"
        "只需模仿这种语言腔调，正常地、直接地谈论用户给出的话题即可。"
    ),
}

NEUTRAL_PROMPT = (
    "你是一个清晰、克制的中文评论助手。请就用户给出的话题，给出一段有条理的分析。"
)

# Keep style_only / neutral output length comparable to full so stylometry and
# rubric scoring are not confounded by length.
_LEN_HINT = "（篇幅控制在 300 字左右。）"


def system_prompt_for(persona: str, condition: str) -> str:
    if condition == "full":
        return skill_body(persona) + "\n\n" + _LEN_HINT
    if condition == "style_only":
        return STYLE_ONLY_PROMPTS[persona] + _LEN_HINT
    if condition == "neutral":
        return NEUTRAL_PROMPT + _LEN_HINT
    raise ValueError(f"unknown condition: {condition}")


def generate(persona: str, probe: dict, condition: str, *, use_cache: bool = True) -> str:
    system = system_prompt_for(persona, condition)
    # All conditions share the same neutral user probe; the persona is injected
    # only via the system prompt, so the three outputs answer an identical task.
    user = probe["prompt"]
    return call_model(
        endpoint_key=persona,                 # agent runs on the persona's endpoint
        system_prompt=system,
        user_prompt=user,
        temperature=0.7,
        max_tokens=700,
        use_cache=use_cache,
        tag=f"gen::{persona}::{probe['id']}::{condition}",
    ).strip()
