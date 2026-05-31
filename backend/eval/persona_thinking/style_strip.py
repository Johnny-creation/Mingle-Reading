# -*- coding: utf-8 -*-
"""Normalise an output into plain modern Chinese, preserving its judgments and
reasoning while removing all stylistic markers.

This is the operational core of the style/thought dissociation: the thinking
rubric is scored on the *stripped* text, so a model that only mimics diction
(no real cognitive moves) collapses to a low thinking score once its style is
gone, while a model that genuinely reasons keeps its content.
"""
from __future__ import annotations

from .common import call_model

_STRIP_SYSTEM = (
    "你是一个严谨的文本改写器。任务：把输入文本改写成**平实、中性、现代白话**的表达，"
    "彻底去除一切语言风格痕迹——包括文白夹杂、文言虚词、破折号制造的顿挫、‘倘若…然而…’"
    "‘大约…的确…’‘因为…所以…’等标志性句式、华丽辞藻、中英文夹杂、刻意的意象与隐喻措辞。\n"
    "**铁律**：只改变‘怎么说’，绝不改变‘说了什么’。必须原样保留作者的每一个判断、论点、"
    "推理步骤、立场、所举的对照/类比的实质内容和因果关系；不得增删观点，不得补充你自己的看法，"
    "不得做任何评价。如果原文用了隐喻，请把隐喻所指的实质意思用平实语言说出来，但保留该论点。\n"
    "直接输出改写后的纯文本，不要任何前言、解释或标注。"
)


def strip_style(text: str, *, use_cache: bool = True, tag: str = "") -> str:
    return call_model(
        endpoint_key="neutral",
        system_prompt=_STRIP_SYSTEM,
        user_prompt=f"请改写下面这段文字：\n\n{text}",
        temperature=0.1,
        max_tokens=700,
        use_cache=use_cache,
        tag=f"strip::{tag}",
    ).strip()
