# -*- coding: utf-8 -*-
"""Fetch CLEAN, SIMPLIFIED Lu Xun essays from zh.wikisource.org.

Why: the local 《鲁迅全集》EPUB is a corrupted Internet-Archive OCR scan (see
notice.html in that file), so it cannot serve as ground truth. zh.wikisource.org
is the project's own canonical source for Lu Xun (every KB work record cites a
`zh.wikisource.org/zh-hans/...` source_url). We pull the simplified rendering so
the function-word profile matches the (simplified) model outputs scored by
Burrows's Delta.

Resolution: Lu Xun essays are TOP-LEVEL pages (traditional titles, e.g.
《娜拉走後怎樣》), indexed from their collection pages. We merge the TOCs of the
relevant collections into one {simplified_title -> traditional_pagetitle} map,
resolve the held-out titles in data/heldout_manifest.json against it, then fetch
and clean each essay body. Resolved page titles + char counts are written so the
split is frozen and reproducible.

Run from Mingle-Reading-main/:
    python -m backend.eval.persona_thinking.tools.fetch_luxun_wikisource
"""
from __future__ import annotations

import html as htmllib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve()
BASE = HERE.parents[1]
MANIFEST = BASE / "data" / "heldout_manifest.json"
OUT_DIR = BASE / "data" / "clean" / "lu-xun"

# Collections (traditional names) that contain the held-out essays. Merged into
# one title map, so a wrong per-essay collection guess in the manifest is fine.
COLLECTIONS = ["墳", "而已集", "二心集", "南腔北調集", "准風月談", "偽自由書", "花邊文學"]

_NS = ("Special:", "Wikisource:", "Help:", "Author:", "File:", "Category:",
       "Template:", "Portal:", "Index:", "Page:", "作者:", "W:", "Talk:")
_BOILER = ("维基文库", "姊妹计划", "数据项", "跳转", "署名", "发表", "下载",
           "本作品", "公有领域", "Public domain", "版权", "逝世", "↑", "false",
           "目录", "编辑")
_UA = {"User-Agent": "Mozilla/5.0 (MingleReading persona-eval research)"}


def _get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers=_UA)
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "ignore")


def _cjk(s: str) -> int:
    return sum(1 for c in s if "一" <= c <= "鿿")


def _page_hans(title: str) -> str:
    return _get("https://zh.wikisource.org/zh-hans/" + urllib.parse.quote(title))


def collection_map(collection: str) -> dict[str, str]:
    """{simplified link text -> traditional page title} for one collection page."""
    html = _page_hans(collection)
    i = html.find('mw-parser-output')
    if i == -1:
        return {}
    for mk in ('id="catlinks"', 'class="printfooter"'):
        j = html.find(mk, i)
        if j != -1:
            html = html[:j]
    html = html[i:]
    out: dict[str, str] = {}
    for href, _attrs, text in re.findall(r'<a href="/wiki/([^"#?]+)"([^>]*)>(.*?)</a>', html):
        title = urllib.parse.unquote(href)
        if any(title.startswith(ns) for ns in _NS):
            continue
        txt = re.sub(r"<[^>]+>", "", htmllib.unescape(text)).strip()
        if txt and txt not in out and title not in COLLECTIONS:
            out[txt] = title
    return out


def clean_body(html: str) -> str:
    i = html.find('mw-parser-output')
    if i == -1:
        return ""
    body = html[i:]
    for marker in ('class="printfooter"', 'id="catlinks"',
                   'class="mw-authority-control"', 'id="mw-navigation"', "<footer"):
        j = body.find(marker)
        if j != -1:
            body = body[:j]
    body = re.sub(r"<table.*?</table>", " ", body, flags=re.S | re.I)
    body = re.sub(r"<ref[^>]*>.*?</ref>", " ", body, flags=re.S | re.I)
    body = re.sub(r"<(script|style|sup|h1|h2|h3)[^>]*>.*?</\1>", " ", body, flags=re.S | re.I)
    body = re.sub(r"<[^>]+>", " ", body)
    t = htmllib.unescape(body)
    t = re.sub(r"\[\d+\]", " ", t)
    raw = [re.sub(r"[ \t]+", " ", ln).strip() for ln in t.splitlines()]
    raw = [ln for ln in raw if ln]
    start = 0
    for k, ln in enumerate(raw):
        if _cjk(ln) >= 12 and not any(b in ln for b in _BOILER):
            start = k
            break
    keep: list[str] = []
    for ln in raw[start:]:
        if any(b in ln for b in ("公有领域", "Public domain", "本作品", "维基文库")):
            break
        # drop mid-body navigation links the start-scan would have skipped,
        # e.g. the per-section "[ 编辑 ]" edit links from the HTML.
        if re.fullmatch(r"\[?\s*编辑\s*\]?", ln):
            continue
        keep.append(ln)
    return "\n".join(keep)


def fetch_essay(page_title: str) -> str:
    return clean_body(_page_hans(page_title))


def slug(title: str) -> str:
    return re.sub(r"\s+", "_", title.strip())


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    spec = manifest["personas"]["lu-xun"]
    wanted = [e["title"] for e in spec["heldout_essays"]]
    moves_by_title = {e["title"]: e.get("moves", []) for e in spec["heldout_essays"]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) build merged simplified->traditional title map
    merged: dict[str, str] = {}
    for coll in COLLECTIONS:
        m = collection_map(coll)
        print(f"  [toc] {coll}: {len(m)} entries")
        for k, v in m.items():
            merged.setdefault(k, v)
        time.sleep(0.4)

    # 2) resolve + fetch
    resolved, missing = {}, []
    for title in wanted:
        page = merged.get(title)
        if not page:
            # fuzzy contains fallback
            for k, v in merged.items():
                if title in k or k in title:
                    page = v
                    break
        if not page:
            missing.append(title)
            print(f"  MISS  {title}: no page in merged TOC")
            continue
        text = fetch_essay(page)
        time.sleep(0.4)
        cc = _cjk(text)
        if cc < 300:
            missing.append(title)
            print(f"  THIN  {title} -> {page}: only {cc} cjk, skipped")
            continue
        rec = {
            "persona": "lu-xun",
            "title": title,
            "source": "zh.wikisource.org",
            "source_entry": f"https://zh.wikisource.org/zh-hans/{urllib.parse.quote(page)}",
            "wikisource_page": page,
            "moves": moves_by_title.get(title, []),
            "char_count": len([c for c in text if c.strip()]),
            "text": text,
        }
        (OUT_DIR / f"{slug(title)}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        resolved[title] = page
        print(f"  OK    {title} -> {page}  ({rec['char_count']} chars)")

    print(f"\n[fetch_luxun] resolved {len(resolved)}/{len(wanted)} essays into {OUT_DIR}")
    if missing:
        print(f"[fetch_luxun] MISSING {len(missing)}: {missing}")


if __name__ == "__main__":
    main()
