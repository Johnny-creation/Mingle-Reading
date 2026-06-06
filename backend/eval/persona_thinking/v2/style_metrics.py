# -*- coding: utf-8 -*-
"""Objective STYLE metric: the S axis of the double dissociation.

PRIMARY metric — `style_ratio` (directional signature-marker intensity):
  We count the author's surface STYLE fingerprints (classical particles, the
  signature connective/modal frames, dashes, irony tells for Lu Xun; sensory /
  color words, simile markers, the causal/aphoristic frames for Zhang Ailing),
  normalise to a density per 100 chars, and divide by the SAME density measured
  on the author's real prose. So style_ratio ≈ 1.0 means "as stylistically
  saturated as the author actually writes," > 1 means even denser, ≈ 0 means
  plain. HIGHER = more author-like style. Because topic is held constant across
  conditions (same probe), differences in marker density reflect HOW it is said,
  not WHAT about. Prediction under the dissociation:
      ratio(full) ~ ratio(style_only)  >>  ratio(thinking_only) ~ ratio(neutral)

  Why not Burrows's Delta as the headline: Delta is a *symmetric* distance from
  the author's mean function-word vector, so a short, deliberately over-stylised
  passage (style_only packing 倘若/然而/之乎者也) pushes low-frequency features to
  extreme z-scores and lands FARTHER from the centroid than plain prose — i.e.
  Delta reads "more style" as "less author-like," inverting the S axis (observed
  empirically for Lu Xun: style_only delta 1.31 > neutral 0.84). A directional
  density does not have this failure mode: more of the author's markers = more of
  the author's style, monotonically.

SECONDARY metric — `delta_to_author` (Burrows 2002, kept for description only):
  z-scored function-word distance to the author's stylistic centroid; LOWER =
  closer. Reported as a descriptive cross-check, not the dissociation headline.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

from backend.eval.persona_thinking.v2 import corpus

# Fixed closed-class function-word feature set. Multi-char frames first so they
# are counted as units; all are topic-independent style markers. Includes the
# classical particles and signature frames that distinguish Lu Xun / Zhang
# Ailing register — but as language-level features, not author labels.
FUNCTION_FEATURES: list[str] = [
    # signature multi-char frames / connectives
    "倘若", "然而", "大约", "的确", "未必", "罢了", "于是", "因为", "所以",
    "虽然", "但是", "况且", "何况", "至于", "总之", "然则", "不过", "只是",
    "而已", "也许", "竟", "终于", "毕竟", "原来", "仿佛", "似乎",
    # classical particles
    "之", "乎", "者", "也", "矣", "焉", "哉", "耳", "兮", "其", "盖", "夫",
    # particles / aspect
    "的", "了", "着", "过", "们", "吗", "呢", "吧", "啊", "呵",
    # conjunctions / prepositions
    "而", "且", "并", "或", "及", "与", "和", "但", "却", "则", "倘", "若",
    "因", "为", "以", "于", "被", "把", "将", "给", "对", "向", "从", "自",
    "在", "和", "跟", "比", "由",
    # adverbs / negation / degree
    "不", "没", "无", "非", "未", "莫", "很", "更", "最", "太", "都", "又",
    "还", "就", "才", "便", "即", "既", "只", "也", "再", "却",
    # pronouns / demonstratives
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "自己", "这", "那",
    "这样", "那样", "怎样", "什么", "谁", "如此", "彼", "此",
]


def featurize(text: str) -> dict[str, float]:
    """Frequency per 1000 characters for each function-word feature."""
    n = len([c for c in text if c.strip()]) or 1
    out: dict[str, float] = {}
    for f in FUNCTION_FEATURES:
        out[f] = text.count(f) / n * 1000.0
    return out


def _chunks(text: str, size: int = 500) -> list[str]:
    chars = [c for c in text if c.strip()]
    return ["".join(chars[i : i + size]) for i in range(0, len(chars), size) if len(chars[i : i + size]) >= size // 2]


@lru_cache(maxsize=8)
def author_profile(persona: str) -> tuple[dict[str, float], dict[str, float]]:
    """(means, stds) of function-word frequencies over chunks of the author's
    real prose — the stylistic centroid + spread used to z-score test texts."""
    ref = corpus.reference_corpus_text(persona)
    chunks = _chunks(ref)
    if not chunks:
        raise RuntimeError(f"no reference corpus for {persona}; run the corpus tools first")
    vecs = [featurize(c) for c in chunks]
    means, stds = {}, {}
    for f in FUNCTION_FEATURES:
        xs = [v[f] for v in vecs]
        mu = sum(xs) / len(xs)
        var = sum((x - mu) ** 2 for x in xs) / len(xs)
        means[f] = mu
        stds[f] = math.sqrt(var)
    return means, stds


def delta_to_author(text: str, persona: str) -> float:
    """Burrows's Delta: mean absolute z-score of the text's function-word
    frequencies relative to the author's profile. Lower = more author-like.
    SECONDARY / descriptive only — see module docstring for why it is not the
    S-axis headline."""
    means, stds = author_profile(persona)
    vec = featurize(text)
    zs = []
    for f in FUNCTION_FEATURES:
        sd = stds[f]
        if sd <= 1e-9:
            continue  # constant feature carries no discriminative info
        zs.append(abs((vec[f] - means[f]) / sd))
    return round(sum(zs) / len(zs), 4) if zs else float("nan")


# --- PRIMARY S axis: directional signature-marker intensity ---------------
#
# Surface STYLE fingerprints only (register / rhetoric markers), NOT content.
# Multi-char frames are counted as units. These extend the v1 stylometry.py
# marker set (already justified there) so the two stay consistent.
_SIG_MARKERS: dict[str, list[str]] = {
    "lu-xun": [
        # classical particles / wenyan register
        "之", "乎", "者", "也", "矣", "焉", "哉", "耳", "兮", "盖", "夫", "其",
        # signature connective / modal frames
        "倘若", "倘", "然而", "大约", "的确", "未必", "罢了", "而已", "何尝",
        "莫非", "简直", "岂", "竟", "决",
        # irony / distancing tells
        "所谓", "之类", "云云", "这一类", "这一流",
        # the 顿挫 dash
        "——",
    ],
    "zhang-ailing": [
        # sensory / color saturation
        "苍凉", "荒凉", "华美", "苍白", "葱绿", "桃红", "苍绿",
        "红", "绿", "蓝", "灰", "金", "黑", "白", "紫", "青", "艳",
        # simile / metaphor markers
        "仿佛", "似的", "一般", "如同", "譬如", "恰如", "像",
        # texture / aesthetic register
        "参差", "对照", "媚", "凄", "凉",
        # causal / aphoristic frames
        "因为", "所以", "于是", "便",
    ],
}


def style_intensity(text: str, persona: str) -> float:
    """Density of the author's signature STYLE markers per 100 chars. Directional:
    HIGHER = more of the author's surface style. Over-use raises it (correct for a
    style axis), unlike Delta which would penalise over-use."""
    markers = _SIG_MARKERS.get(persona, [])
    if not markers:
        raise ValueError(f"no signature markers for persona {persona}")
    n = len([c for c in text if c.strip()]) or 1
    hits = sum(text.count(m) for m in markers)
    return round(hits / n * 100.0, 3)


@lru_cache(maxsize=8)
def author_style_baseline(persona: str) -> float:
    """The author's own signature-marker density (per 100 chars), measured on the
    held-out real prose. This is the 1.0 reference for style_ratio."""
    ref = corpus.reference_corpus_text(persona)
    chunks = _chunks(ref)
    if not chunks:
        raise RuntimeError(f"no reference corpus for {persona}; run the corpus tools first")
    vals = [style_intensity(c, persona) for c in chunks]
    return round(sum(vals) / len(vals), 3)


def style_ratio(text: str, persona: str) -> float:
    """PRIMARY S axis. style_intensity(text) / author's own baseline density.
    ≈1.0 = as stylistically saturated as the author really writes; >1 = denser;
    ≈0 = plain. HIGHER = more author-like style."""
    base = author_style_baseline(persona)
    if base <= 1e-9:
        return float("nan")
    return round(style_intensity(text, persona) / base, 4)


if __name__ == "__main__":
    # sanity: author's own essays should score ratio ~1, plain modern text ~0,
    # and the other author's prose should be low on THIS author's markers.
    plain = (
        "短视频平台用算法推荐内容，目的是提高用户停留时间。这种做法有好处也有坏处，"
        "好处是更方便，坏处是可能让人接触的信息变窄。我们应该理性看待，合理使用。"
    )
    for p in corpus.PERSONAS:
        essays = corpus.load_essays(p)
        other_p = "zhang-ailing" if p == "lu-xun" else "lu-xun"
        base = author_style_baseline(p)
        print(f"\n{p}: author baseline density={base} /100char")
        print(f"  ratio(own_essay)   ={style_ratio(essays[0].text[:600], p)}  "
              f"delta={delta_to_author(essays[0].text[:600], p)}")
        print(f"  ratio(other_author)={style_ratio(corpus.load_essays(other_p)[0].text[:600], p)}  "
              f"delta={delta_to_author(corpus.load_essays(other_p)[0].text[:600], p)}")
        print(f"  ratio(plain_modern)={style_ratio(plain, p)}  "
              f"delta={delta_to_author(plain, p)}")
