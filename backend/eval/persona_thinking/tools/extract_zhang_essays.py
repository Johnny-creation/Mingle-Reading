# -*- coding: utf-8 -*-
"""Extract CLEAN Zhang Ailing essays from the formatted EPUB into per-essay JSON.

Unlike the Lu Xun corpus (an Internet-Archive OCR scan that is badly corrupted),
the 《张爱玲大全集【18册】》 EPUB is a properly typeset ebook: toc.ncx maps each
essay title to an OEBPS html file and the body text is intact. We resolve the
held-out essay titles (data/heldout_manifest.json) to their files via the NCX and
write clean UTF-8 text to data/clean/zhang-ailing/<slug>.json.

Run from Mingle-Reading-main/:
    python -m backend.eval.persona_thinking.tools.extract_zhang_essays
"""
from __future__ import annotations

import glob
import html as htmllib
import json
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve()
BASE = HERE.parents[1]                                # ./persona_thinking/
ROOT = HERE.parents[4]                                # Mingle-Reading-main/
WORKSPACE = ROOT.parent                               # MingleReading/
MANIFEST = BASE / "data" / "heldout_manifest.json"
OUT_DIR = BASE / "data" / "clean" / "zhang-ailing"

_CJK = r"一-鿿㐀-䶿＀-￯　-〿"


def _slug(title: str) -> str:
    return re.sub(r"\s+", "_", title.strip())


def find_zhang_epub() -> str:
    for f in glob.glob(str(WORKSPACE / "*.epub")):
        if "张爱玲" in Path(f).name:
            return f
    raise FileNotFoundError("张爱玲 EPUB not found at workspace root")


def page_text(z: zipfile.ZipFile, name: str) -> str:
    raw = z.read(name).decode("utf-8", "ignore")
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    raw = re.sub(r"<\s*br\s*/?>", "\n", raw, flags=re.I)
    raw = re.sub(r"</p>", "\n", raw, flags=re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    text = htmllib.unescape(raw)
    # collapse stray whitespace that the typesetter left between glyphs
    text = re.sub(rf"(?<=[{_CJK}])[ \t]+(?=[{_CJK}])", "", text)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def ncx_title_to_src(z: zipfile.ZipFile) -> dict[str, str]:
    ncx = next(n for n in z.namelist() if n.lower().endswith(".ncx"))
    raw = z.read(ncx).decode("utf-8", "ignore")
    base = "OEBPS/"
    mapping: dict[str, str] = {}
    for m in re.finditer(r"<navPoint.*?<text>(.*?)</text>.*?<content src=\"(.*?)\"", raw, flags=re.S):
        title = htmllib.unescape(m.group(1)).strip()
        src = m.group(2).split("#")[0]
        if title in {"封面", "版权信息", "目录"}:
            continue
        mapping.setdefault(title, base + src if not src.startswith("OEBPS/") else src)
    return mapping


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    spec = manifest["personas"]["zhang-ailing"]
    exclude = set(spec["exclude"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    epub = find_zhang_epub()
    z = zipfile.ZipFile(epub)
    title2src = ncx_title_to_src(z)

    written, missing = [], []
    for entry in spec["heldout_essays"]:
        title = entry["title"]
        if title in exclude:
            print(f"  SKIP (excluded): {title}")
            continue
        src = title2src.get(title)
        if not src:
            missing.append(title)
            print(f"  MISS (no NCX entry): {title}")
            continue
        text = page_text(z, src)
        # drop a leading repeated title line if present
        body = text
        rec = {
            "persona": "zhang-ailing",
            "title": title,
            "source": Path(epub).name,
            "source_entry": src,
            "moves": entry.get("moves", []),
            "char_count": len([c for c in body if c.strip()]),
            "text": body,
        }
        (OUT_DIR / f"{_slug(title)}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written.append((title, rec["char_count"]))
        print(f"  OK  {title}  ({rec['char_count']} chars)  <- {src}")

    print(f"\n[extract_zhang] wrote {len(written)} essays to {OUT_DIR}")
    if missing:
        print(f"[extract_zhang] MISSING {len(missing)}: {missing}")


if __name__ == "__main__":
    main()
