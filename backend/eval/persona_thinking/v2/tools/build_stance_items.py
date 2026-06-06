# -*- coding: utf-8 -*-
"""Semi-automatic builder for held-out STANCE items (Layer-1a ground truth).

For each clean held-out essay we ask an LLM to mine multiple-choice items whose
*correct* option is a judgment the author ACTUALLY makes in that essay (grounded
in a verbatim quote), with three plausible-but-wrong distractors. The author's
own text is the ground truth — no LLM judge is involved at scoring time.

Design choices that protect validity:
  - Every correct option must be backed by a verbatim `evidence` span; we verify
    the span really occurs in the essay and flag items where it does not.
  - Options are plain-register paraphrases (no signature style), so a model
    cannot guess the answer from wording — only from the *judgment*.
  - One distractor is deliberately the mainstream / common-sense view, so items
    where the author is non-obvious become the discriminating `distinctive`
    subset (a generic baseline picks the mainstream distractor and is wrong).
  - Output is a REVIEW file for human spot-checking (the agreed workflow); the
    finalized <persona>.jsonl is what Layer-1a consumes.

Run from Mingle-Reading-main/:
    python -m backend.eval.persona_thinking.v2.tools.build_stance_items --personas lu-xun zhang-ailing
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from backend.eval.persona_thinking.common import call_model, parse_json_loose
from backend.eval.persona_thinking.v2.corpus import load_essays, slug

HERE = Path(__file__).resolve()
V2 = HERE.parents[1]
OUT_DIR = V2 / "stance_items"

DISPLAY = {"lu-xun": "鲁迅", "zhang-ailing": "张爱玲"}

_SYS = (
    "你是一个严谨的中文文学命题助手。给你一篇{author}的真实文章。"
    "请基于文中{author}**明确表达的判断 / 立场 / 价值取向**，出若干道单项选择题，"
    "用来检验另一个模型是否真的复现了{author}本人的判断（而非泛泛而谈）。\n"
    "【铁律】\n"
    "1. 每题的‘correct’必须是{author}在本文里**真实持有**的判断，并给出一段可以逐字在原文中找到的"
    "`evidence`（直接摘录原文，不要改写、不要加引号以外的内容）。\n"
    "2. ‘distractors’给三个：必须貌似合理但并非作者本意，且至少包含——"
    "(a) 一个大众/常识性的看法，(b) 一个与作者相反的看法，(c) 一个看似接近却偏离要害的看法。"
    "干扰项不得明显荒谬或与主题无关。\n"
    "3. correct 与三个 distractor 一律用**平实概括的现代白话**转述判断，"
    "不要照抄作者原话、不要带文白腔或‘倘若…然而…’之类标志性句式，四个选项长度语气尽量一致"
    "（防止凭文风猜答案）。\n"
    "4. ‘topic’用中性、**不泄露结论**的一两句话，描述这道题针对的具体问题或现象。\n"
    "5. 优先出{author}**不同于常人、反直觉**的判断（这些最能区分‘真捕捉到其思维’与‘套话’）。\n"
    "6. 出题数量以能被原文可靠支撑为准，最多 {n} 题；宁缺毋滥。\n"
    "只输出 JSON：{{\"items\":[{{\"topic\":\"...\",\"move\":\"针对的思维维度\","
    "\"correct\":\"...\",\"distractors\":[\"...\",\"...\",\"...\"],\"evidence\":\"原文逐字片段\"}}]}}"
)


def _norm(s: str) -> str:
    # keep only CJK / latin / digits so the substring check is robust to
    # punctuation and quote-mark variants between the quote and the source.
    return "".join(c for c in (s or "") if ("一" <= c <= "鿿") or c.isalnum())


def _verify_evidence(evidence: str, essay_text: str) -> bool:
    e = _norm(evidence)
    return len(e) >= 4 and e in _norm(essay_text)


def build_for_essay(persona: str, essay, n: int, rng: random.Random, use_cache: bool) -> list[dict]:
    author = DISPLAY[persona]
    text = essay.text[:3800]
    sys = _SYS.format(author=author, n=n)
    user = (
        f"【作者】{author}\n【篇名】《{essay.title}》\n"
        f"【本篇侧重的思维维度（命题时可参考）】{', '.join(essay.moves)}\n\n"
        f"【文章正文】\n{text}"
    )
    raw = call_model(
        endpoint_key="neutral",
        system_prompt=sys,
        user_prompt=user,
        temperature=0.2,
        max_tokens=2200,
        json_object=True,
        use_cache=use_cache,
        tag=f"stance::{persona}::{essay.title}::v1",
    )
    try:
        data = parse_json_loose(raw)
    except Exception:
        return []
    items = data.get("items", []) if isinstance(data, dict) else []
    out: list[dict] = []
    for k, it in enumerate(items):
        correct = (it.get("correct") or "").strip()
        distractors = [d.strip() for d in (it.get("distractors") or []) if d and d.strip()]
        evidence = (it.get("evidence") or "").strip()
        if not correct or len(distractors) < 3:
            continue
        distractors = distractors[:3]
        options = distractors + [correct]
        rng.shuffle(options)
        correct_index = options.index(correct)
        out.append(
            {
                "id": f"{persona.split('-')[0]}_{slug(essay.title)}_{k+1}",
                "persona": persona,
                "source_title": essay.title,
                "source": essay.source,
                "move": (it.get("move") or "").strip(),
                "topic": (it.get("topic") or "").strip(),
                "question": f"在这一问题上，{author}的判断最接近以下哪一项？",
                "options": options,
                "correct_index": correct_index,
                "evidence": evidence,
                "evidence_verified": _verify_evidence(evidence, essay.text),
            }
        )
    return out


def render_review(persona: str, items: list[dict]) -> str:
    author = DISPLAY[persona]
    lines = [
        f"# {author} 立场题候选（人工抽检）",
        "",
        f"共 {len(items)} 题。请删除/修改不合格者，定稿存为 `{persona}.jsonl`。",
        "审核要点：① evidence 是否真支持‘正确项’；② 正确项确为作者本意；"
        "③ 干扰项貌似合理但非作者观点；④ 选项不含文风线索；⑤ topic 不泄露结论。",
        f"（自动校验 evidence 命中：{sum(1 for it in items if it['evidence_verified'])}/{len(items)}；"
        "未命中者标 ⚠️，需人工核对原文。）",
        "",
    ]
    by_src: dict[str, list[dict]] = {}
    for it in items:
        by_src.setdefault(it["source_title"], []).append(it)
    for title, group in by_src.items():
        lines.append(f"## 《{title}》")
        lines.append("")
        for it in group:
            flag = "" if it["evidence_verified"] else " ⚠️evidence未命中"
            lines.append(f"### [{it['id']}]　维度：{it['move']}{flag}")
            lines.append(f"- **topic**：{it['topic']}")
            lines.append(f"- **{it['question']}**")
            for i, opt in enumerate(it["options"]):
                mark = "✅" if i == it["correct_index"] else "　"
                lines.append(f"  - {mark} {chr(65+i)}. {opt}")
            lines.append(f"- **evidence**：{it['evidence']}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", nargs="+", default=["lu-xun", "zhang-ailing"])
    ap.add_argument("--per-essay", type=int, default=3)
    ap.add_argument("--max-essays", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--seed", type=int, default=20260605)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for persona in args.personas:
        rng = random.Random(args.seed)
        essays = load_essays(persona)
        if args.max_essays:
            essays = essays[: args.max_essays]
        all_items: list[dict] = []
        for essay in essays:
            items = build_for_essay(persona, essay, args.per_essay, rng, not args.no_cache)
            print(f"  [{persona}] 《{essay.title}》-> {len(items)} items "
                  f"({sum(1 for i in items if i['evidence_verified'])} verified)")
            all_items.extend(items)
        cand_path = OUT_DIR / f"{persona}.candidates.jsonl"
        review_path = OUT_DIR / f"{persona}.review.md"
        cand_path.write_text(
            "\n".join(json.dumps(it, ensure_ascii=False) for it in all_items), encoding="utf-8"
        )
        review_path.write_text(render_review(persona, all_items), encoding="utf-8")
        verified = sum(1 for it in all_items if it["evidence_verified"])
        print(f"[{persona}] {len(all_items)} candidate items "
              f"({verified} evidence-verified) -> {cand_path.name}, {review_path.name}")


if __name__ == "__main__":
    main()
