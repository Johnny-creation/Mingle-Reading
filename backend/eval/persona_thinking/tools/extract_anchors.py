# -*- coding: utf-8 -*-
"""Extract genuine primary-text anchors from the persona corpora EPUBs.

These anchors are shown to the judge as evidence of *how the author actually
reasons*, so rubric scoring is grounded in real text rather than the judge's
stereotype of the author. Output is written to ../corpus_anchors/<persona>.json.

Run from Mingle-Reading-main/:
    python backend/eval/persona_thinking/tools/extract_anchors.py
"""
from __future__ import annotations

import glob
import html as htmllib
import json
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]                      # Mingle-Reading-main/
WORKSPACE = ROOT.parent                     # MingleReading/
OUT_DIR = HERE.parents[1] / "corpus_anchors"

# Each anchor target: a human-readable label + title markers to locate the piece
# + the cognitive moves it exemplifies (for the judge's reference).
LU_XUN_TARGETS = [
    {"label": "《拿来主义》", "markers": ["拿来主义"], "moves": ["辩证逻辑(占有挑选)", "二难拆解", "讽刺"]},
    {"label": "《论“费厄泼赖”应该缓行》", "markers": ["费厄泼赖", "费厄泼"], "moves": ["二难推理", "落水狗", "拒绝中庸调和"]},
    {"label": "《我之节烈观》", "markers": ["我之节烈观", "节烈"], "moves": ["归谬法", "弱者正义", "去蔽"]},
    {"label": "《灯下漫笔》", "markers": ["灯下漫笔"], "moves": ["历史循环视角", "国民性", "奴性"]},
    {"label": "《狂人日记》", "markers": ["狂人日记"], "moves": ["吃人隐喻", "去蔽", "礼教批判"]},
]

ZHANG_TARGETS = [
    {"label": "《自己的文章》", "markers": ["自己的文章", "不彻底的人物"], "moves": ["苍凉美学", "参差对照", "反壮烈"]},
    {"label": "《金锁记》", "markers": ["金锁记", "曹七巧"], "moves": ["物质细节作心理索引(黄金的枷)", "不愿承认的欲望"]},
    {"label": "《倾城之恋》", "markers": ["倾城之恋", "白流苏"], "moves": ["爱情即生存算计", "战略撤退", "双城空间"]},
    {"label": "《天才梦》", "markers": ["天才梦", "华美的袍"], "moves": ["参差对照", "自嘲先扬后抑"]},
    {"label": "《红玫瑰与白玫瑰》", "markers": ["红玫瑰与白玫瑰", "佟振保"], "moves": ["心理现实主义", "道德面具下的懦弱"]},
]


def find_epub(predicate) -> str | None:
    for f in glob.glob(str(WORKSPACE / "*.epub")):
        names = zipfile.ZipFile(f).namelist()
        if predicate(names):
            return f
    return None


_CJK = r"一-鿿㐀-䶿＀-￯　-〿"


def page_text(z: zipfile.ZipFile, name: str) -> str:
    raw = z.read(name).decode("utf-8", "ignore")
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    text = htmllib.unescape(raw)
    # The Lu Xun EPUB inserts whitespace between every CJK glyph ("拿 来 主 义");
    # drop whitespace that sits between two CJK characters so markers match.
    text = re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])", "", text)
    return re.sub(r"[ \t]+", " ", text)


def collect(epub: str, targets: list[dict], max_chars: int = 1600) -> list[dict]:
    z = zipfile.ZipFile(epub)
    pages = [n for n in z.namelist() if n.lower().endswith((".html", ".xhtml", ".htm"))]
    # cache page text once
    cache: dict[str, str] = {}
    for n in pages:
        try:
            cache[n] = page_text(z, n)
        except Exception:
            cache[n] = ""

    anchors: list[dict] = []
    for tgt in targets:
        best = None
        for n in pages:
            txt = cache[n]
            score = sum(txt.count(m) for m in tgt["markers"])
            if score <= 0:
                continue
            # prefer the page where the title appears earliest (likely the start)
            first_pos = min((txt.find(m) for m in tgt["markers"] if m in txt), default=10**9)
            cand = (score, -first_pos, n, txt)
            if best is None or cand[:2] > best[:2]:
                best = cand
        if best is None:
            anchors.append({"label": tgt["label"], "moves": tgt["moves"], "found": False, "excerpt": ""})
            continue
        _, _, name, txt = best
        # trim to a window starting near the first marker
        start = max(0, min((txt.find(m) for m in tgt["markers"] if m in txt)) - 20)
        excerpt = txt[start:start + max_chars].strip()
        anchors.append(
            {
                "label": tgt["label"],
                "moves": tgt["moves"],
                "found": True,
                "source_entry": name,
                "excerpt": excerpt,
            }
        )
    return anchors


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lx = find_epub(lambda ns: any(n.startswith("EPUB/page_") for n in ns))
    zh = find_epub(lambda ns: any("OEBPS/text000" in n for n in ns))
    print("lu-xun epub:", bool(lx), "| zhang-ailing epub:", bool(zh))

    if lx:
        lx_anchors = collect(lx, LU_XUN_TARGETS)
        (OUT_DIR / "lu-xun.json").write_text(
            json.dumps({"persona": "lu-xun", "source": Path(lx).name, "anchors": lx_anchors},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        for a in lx_anchors:
            print(f"  [lu-xun] {a['label']}: found={a['found']} len={len(a['excerpt'])}")

    if zh:
        zh_anchors = collect(zh, ZHANG_TARGETS)
        (OUT_DIR / "zhang-ailing.json").write_text(
            json.dumps({"persona": "zhang-ailing", "source": Path(zh).name, "anchors": zh_anchors},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        for a in zh_anchors:
            print(f"  [zhang-ailing] {a['label']}: found={a['found']} len={len(a['excerpt'])}")


if __name__ == "__main__":
    sys.exit(main())
