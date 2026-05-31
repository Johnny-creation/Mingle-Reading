# -*- coding: utf-8 -*-
"""Corpus-grounded, blind LLM judging.

Two judgments:
  1. rubric scoring (1-5 per thinking dimension) of a single (style-stripped)
     analysis, grounded in real excerpts of the author so the judge compares
     against actual text rather than its stereotype.
  2. forced-choice discrimination between two analyses — robust to scale
     miscalibration; caller randomises A/B order to cancel position bias.

The judge never sees which condition produced a text (blind).
"""
from __future__ import annotations

import json
from typing import Any

from .common import call_model, parse_json_loose


def _rubric_block(rubric: dict[str, Any]) -> str:
    lines = []
    for d in rubric["dimensions"]:
        lines.append(
            f"- {d['id']}（{d['name']}）：{d['description']}\n"
            f"    · 高分(5)样貌：{d['anchor_high']}\n"
            f"    · 低分(1)样貌：{d['anchor_low']}"
        )
    return "\n".join(lines)


_RUBRIC_SYSTEM = (
    "你是文学思维方式的严格评审。你要判断的不是文字是否优美、风格是否像某位作家，"
    "而是其中体现的**思维方式**（认知动作、价值透镜、看问题的角度）在多大程度上贴近这位作家。"
    "注意：被评文本已被改写成平实语体、去除了语言风格，所以你只能依据‘怎么想’而非‘怎么说’来打分。"
    "请严格对照所给的‘作家真实原文’参照和评分量表，对每个维度给 1-5 的整数分，并附简短依据。"
    "只输出 JSON。"
)


def judge_rubric(
    *,
    persona_name: str,
    rubric: dict[str, Any],
    anchors: str,
    scenario: str,
    stripped_text: str,
    judge_key: str = "neutral",
    run_idx: int = 0,
) -> dict[str, Any]:
    dim_ids = [d["id"] for d in rubric["dimensions"]]
    user = (
        f"【作家】{persona_name}\n\n"
        f"【作家真实原文参照（这是该作家实际如何思考的证据）】\n{anchors}\n\n"
        f"【评分量表】\n5=该作家水准；4=明显运用但略生硬；3=有雏形但浅；2=仅表面痕迹；1=完全缺失或相反。\n\n"
        f"【思维维度】\n{_rubric_block(rubric)}\n\n"
        f"【被评分析所针对的情境】\n{scenario}\n\n"
        f"【被评分析（已去风格的平实版本）】\n{stripped_text}\n\n"
        "请对每个维度打分。只输出如下 JSON：\n"
        "{\"scores\": {\"维度ID\": {\"score\": 整数1-5, \"reason\": \"一句依据\"}, ...}}\n"
        f"维度ID 必须且只能是：{dim_ids}"
    )
    raw = call_model(
        endpoint_key=judge_key,
        system_prompt=_RUBRIC_SYSTEM,
        user_prompt=user,
        temperature=0.0,
        max_tokens=1100,
        json_object=True,
        tag=f"judge_rubric::{persona_name}::run{run_idx}::{hash(stripped_text) & 0xffff}",
    )
    data = parse_json_loose(raw)
    scores = data.get("scores", data)
    out: dict[str, Any] = {}
    for d in rubric["dimensions"]:
        item = scores.get(d["id"], {})
        try:
            sc = int(round(float(item.get("score")))) if isinstance(item, dict) else int(round(float(item)))
        except (TypeError, ValueError):
            sc = None
        out[d["id"]] = {
            "score": sc if (sc is None or 1 <= sc <= 5) else None,
            "reason": (item.get("reason") if isinstance(item, dict) else "") or "",
        }
    return out


_CHOICE_SYSTEM = (
    "你是文学思维方式的严格评审。下面有两段针对同一情境的分析（都已去除语言风格、改写成平实语体）。"
    "请判断哪一段在**思维方式**上更贴近指定作家——看认知动作、价值透镜、看问题的角度，而不是文采。"
    "请参照所给的作家真实原文。只输出 JSON。"
)


def forced_choice(
    *,
    persona_name: str,
    anchors: str,
    scenario: str,
    text_a: str,
    text_b: str,
    judge_key: str = "neutral",
    tag: str = "",
) -> dict[str, Any]:
    user = (
        f"【作家】{persona_name}\n\n"
        f"【作家真实原文参照】\n{anchors}\n\n"
        f"【情境】\n{scenario}\n\n"
        f"【分析 A】\n{text_a}\n\n"
        f"【分析 B】\n{text_b}\n\n"
        "哪一段在思维方式上更像这位作家？只输出 JSON：\n"
        "{\"winner\": \"A\" 或 \"B\" 或 \"tie\", \"reason\": \"一句依据\"}"
    )
    raw = call_model(
        endpoint_key=judge_key,
        system_prompt=_CHOICE_SYSTEM,
        user_prompt=user,
        temperature=0.0,
        max_tokens=400,
        json_object=True,
        tag=f"forced_choice::{persona_name}::{tag}",
    )
    data = parse_json_loose(raw)
    winner = str(data.get("winner", "")).strip().upper()
    if winner not in {"A", "B", "TIE"}:
        winner = "TIE"
    return {"winner": winner, "reason": data.get("reason", "")}
