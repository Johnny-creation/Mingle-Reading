# -*- coding: utf-8 -*-
"""Clean held-out corpus access for the persona-thinking evaluation.

The held-out works (data/heldout_manifest.json -> data/clean/<persona>/*.json)
are the OBJECTIVE ground truth: the author's own judgments, used by Layer-1
stance / continuation tasks and as the function-word reference for Burrows's
Delta. They are kept strictly disjoint from the SKILL examples / persona-KB
sources (see the `exclude` list per persona in the manifest).

Sources differ by author because the raw corpora differ in quality:
  - zhang-ailing : extracted from the clean 《张爱玲大全集》EPUB
                   (tools/extract_zhang_essays.py)
  - lu-xun       : fetched simplified from zh.wikisource.org, the project's own
                   canonical source (tools/fetch_luxun_wikisource.py), because
                   the 《鲁迅全集》EPUB is a corrupted Internet-Archive OCR scan.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CLEAN = DATA / "clean"
MANIFEST_PATH = DATA / "heldout_manifest.json"

PERSONAS = ("lu-xun", "zhang-ailing")


@dataclass
class Essay:
    persona: str
    title: str
    text: str
    moves: list[str]
    source: str
    char_count: int


def slug(title: str) -> str:
    return re.sub(r"\s+", "_", title.strip())


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def manifest_for(persona: str) -> dict:
    return load_manifest()["personas"][persona]


def excluded_titles(persona: str) -> set[str]:
    return set(manifest_for(persona).get("exclude", []))


def load_essays(persona: str, *, min_chars: int = 400) -> list[Essay]:
    """Load all clean held-out essays for a persona, enforcing the exclusion
    list and a minimum length (very short pieces make weak stance ground truth)."""
    if persona not in PERSONAS:
        raise ValueError(f"unknown persona: {persona}")
    excl = excluded_titles(persona)
    out: list[Essay] = []
    pdir = CLEAN / persona
    if not pdir.exists():
        return out
    for fp in sorted(pdir.glob("*.json")):
        rec = json.loads(fp.read_text(encoding="utf-8"))
        title = rec.get("title", fp.stem)
        if title in excl:
            continue
        text = (rec.get("text") or "").strip()
        cc = rec.get("char_count") or len([c for c in text if c.strip()])
        if cc < min_chars:
            continue
        out.append(
            Essay(
                persona=persona,
                title=title,
                text=text,
                moves=rec.get("moves", []),
                source=rec.get("source", ""),
                char_count=cc,
            )
        )
    return out


def reference_corpus_text(persona: str) -> str:
    """Concatenated clean prose for the author — the stylistic reference used to
    fit the Burrows's Delta function-word z-score profile. Held-out essays are
    fine here: Delta needs a representative sample of the author's real prose."""
    return "\n\n".join(e.text for e in load_essays(persona))


def split_paragraphs(text: str, *, min_chars: int = 60) -> list[str]:
    """Paragraph segmentation for continuation / passage tasks. Splits on blank
    lines and the essay's own newlines, keeping paragraphs of substance."""
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    return [p for p in paras if len([c for c in p if c.strip()]) >= min_chars]


if __name__ == "__main__":
    for p in PERSONAS:
        essays = load_essays(p)
        total = sum(e.char_count for e in essays)
        print(f"{p}: {len(essays)} essays, {total} chars")
        for e in essays[:50]:
            print(f"  - {e.title}  ({e.char_count})  moves={e.moves}")
