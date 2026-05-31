# -*- coding: utf-8 -*-
"""Cheap, deterministic stylometry on the RAW (non-stripped) outputs.

Purpose: show that `full` and `style_only` are stylistically similar (style is
controlled), so any thinking-rubric gap between them cannot be dismissed as a
style artifact. Features are surface-level on purpose — they are exactly the
markers the style/thought dissociation aims to factor out.
"""
from __future__ import annotations

import re
from typing import Any

_SENT_SPLIT = re.compile(r"[。！？!?；;\n]+")

# Persona-specific surface markers (the language-style fingerprints).
_LX_WENYAN = ("之", "乎", "者", "也", "矣", "焉", "其", "盖")
_LX_PATTERNS = ("倘若", "然而", "大约", "的确", "未必", "罢了")
_ZA_COLOR = ("苍凉", "荒凉", "华美", "红", "绿", "灰", "金", "苍白")
_ZA_PATTERN_CAUSAL = ("因为", "所以", "于是", "便")


def _ratio(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def stylometry(text: str, persona: str) -> dict[str, Any]:
    chars = [c for c in text if c.strip()]
    n_char = len(chars) or 1
    sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    n_sent = len(sents) or 1
    sent_lens = [len([c for c in s if c.strip()]) for s in sents]

    feats: dict[str, Any] = {
        "char_count": len(chars),
        "sentence_count": len(sents),
        "avg_sentence_len": round(sum(sent_lens) / n_sent, 2),
        "short_sentence_ratio": _ratio(sum(1 for l in sent_lens if l <= 8), n_sent),
        "dash_per_100char": round(text.count("—") / n_char * 100, 3),
        "exclamation_count": text.count("！") + text.count("!"),
        "latin_ratio": _ratio(sum(1 for c in text if ("a" <= c.lower() <= "z")), n_char),
    }
    if persona == "lu-xun":
        feats["wenyan_per_100char"] = round(sum(text.count(w) for w in _LX_WENYAN) / n_char * 100, 3)
        feats["signature_pattern_count"] = sum(text.count(p) for p in _LX_PATTERNS)
    elif persona == "zhang-ailing":
        feats["color_word_count"] = sum(text.count(w) for w in _ZA_COLOR)
        feats["causal_pattern_count"] = sum(text.count(p) for p in _ZA_PATTERN_CAUSAL)
    return feats
