# -*- coding: utf-8 -*-
"""Apply the human spot-check fixes to the stance candidates -> final .jsonl.

The user (a literature-background reviewer) spot-checked the auto-generated
stance candidates and recorded conclusions in
`stance_items/persona_stance_review_fix_notes.md`. This script encodes those
conclusions DETERMINISTICALLY so the final item set is auditable and
reproducible (no LLM in the loop here): we read `<persona>.candidates.jsonl`,
apply a small table of per-id edits, and write `<persona>.jsonl` (which
layer1_stance.load_items prefers over the candidates).

Edit types, mirroring the reviewer's guidance ("prefer supplementing evidence;
only narrow option text when evidence cannot be supplemented"):
  - evidence    : replace the evidence span with a fuller VERBATIM quote pulled
                  from the held-out clean essay (data/clean/<persona>/*.json).
  - topic       : narrow the question situation to remove single-choice ambiguity.
  - options/correct_index : reformulate an item whose evidence cannot support the
                  original option (only used where the reviewer asked for it).
  - verified    : clear a false `evidence未命中` misfire flag the reviewer
                  confirmed was wrong (set evidence_verified=True).

Run from Mingle-Reading-main/:
    python -m backend.eval.persona_thinking.v2.tools.apply_review_fixes
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve()
V2 = HERE.parents[1]
STANCE = V2 / "stance_items"

# --- Priority edits (verbatim evidence taken from the held-out clean essays) ---

EDITS: dict[str, dict] = {
    # ----------------------------- 鲁迅 -----------------------------
    # 夜颂_1: keep D; evidence only had the exclusionary opening — add the
    # positive "看夜的眼睛 / 领受了夜所给与的光明" lines.
    "lu_夜颂_1": {
        "evidence": (
            "爱夜的人，也不但是孤独者，有闲者，不能战斗者，怕光明者。"
            "……爱夜的人要有听夜的耳朵和看夜的眼睛，自在暗中，看一切暗。"
            "……爱夜的人于是领受了夜所给与的光明。"
        ),
        "evidence_verified": True,
    },
    # 春末闲谈_3: keep D; supplement the satire context so "为现有秩序辩护"
    # (说一切状态都已够好) is grounded, not just "并未超出前贤范围".
    "lu_春末闲谈_3": {
        "evidence": (
            "世上挺生了一种所谓“特殊知识阶级”的留学生，在研究室中研究之结果，"
            "说医学不发达是有益于人种改良的，中国妇女的境遇是极其平等的，"
            "一切道理都已不错，一切状态都已够好。……便是留学生的特别发见，"
            "其实也并未轶出了前贤的范围。"
        ),
        "evidence_verified": True,
    },
    # 现代史_3: keep D; supplement the "总是这一套…又要静几天" cycle lines that
    # make the title↔content irony (变戏法即现代史) legible.
    "lu_现代史_3": {
        "evidence": (
            "其实是许多年间，总是这一套，也总有人看，总有人Huazaa，"
            "不过其间必须经过沉寂的几日。我的话说完了，意思也浅得很，"
            "不过说大家HuazaaHuazaa一通之后，又要静几天了，然后再来这一套。"
            "到这里我才记得写错了题目，这真是成了“不死不活”的东西。"
        ),
        "evidence_verified": True,
    },
    # Confirmed-correct items whose evidence未命中 flag was a false misfire.
    "lu_未有天才之前_2": {"evidence_verified": True},
    "lu_现代史_1": {"evidence_verified": True},
    "lu_论睁了眼看_1": {"evidence_verified": True},

    # ---------------------------- 张爱玲 ----------------------------
    # 私语_1: keep C; the original evidence supported "精致完全的体系" but not
    # "并非真正的家" — add the "真的家应当是合身的，随着我生长的" sentence.
    "zhang_私语_1": {
        "evidence": (
            "然而我对于我姑姑的家却有一种天长地久的感觉。"
            "……她的家对于我一直是一个精致完全的体系，无论如何不能让它稍有毁损。"
            "……因为现在的家于它的本身是细密完全的，而我只是在里面撞来撞去打碎东西，"
            "而真的家应当是合身的，随着我生长的，我想起我从前的家了。"
        ),
        "evidence_verified": True,
    },
    # 姑姑语录_3: keep B; add the 披霞 cause (欠好/颜色难配/形状有缺陷) so the
    # conclusion "生命没有意义" has its premise.
    "zhang_姑姑语录_3": {
        "evidence": (
            "只有一块淡红的披霞，还留到现在，因为欠好的缘故。"
            "……襟上挂着做个装饰品罢，衬着什么底子都不好看。"
            "……除非把它悬空宕着，做个扇坠什么的。然而它只有一面是光滑的，"
            "反面就不中看；上头的一个洞，位置又不对，在宝石的正中。"
            "姑姑叹了口气，说：“看着这块披霞，使人觉得生命没有意义。”"
        ),
        "evidence_verified": True,
    },
    # 谈画_1: single-choice ambiguity (恋人-reading D also had textual support in
    # the pre-identification passage). Narrow the situation to AFTER the model is
    # identified as a young wife — there Zhang gives the mother-child reading (C).
    "zhang_谈画_1": {
        "topic": (
            "在《蒙纳·丽萨》的模特儿被考证出是一位年轻太太之后，"
            "张爱玲更倾向于用哪一种解释来理解那神秘的微笑？"
        ),
        "evidence_verified": True,
    },
    # 传奇再版的话_3: the 苍凉 vs 壮烈 comparison lives in 《自己的文章》(held-out
    # EXCLUDED). Reformulate around what THIS essay actually supports: why she
    # most often uses the word 荒凉 — the 惘惘的威胁 of a civilization about to pass.
    "zhang_传奇再版的话_3": {
        "topic": "张爱玲解释自己为什么最常用“荒凉”这个字",
        "options": [
            "因为“荒凉”只是她一时的个人情绪，与时代无关。",
            "因为她偏爱古典悲剧那种壮烈的余韵，“荒凉”是它的尾声。",
            "因为她的思想背景里有一种惘惘的威胁——预感眼前的文明终将成为过去。",
            "因为她生性悲观，对世上的一切都不抱希望。",
        ],
        "correct_index": 2,
        "evidence": (
            "个人即使等得及，时代是仓促的，已经在破坏中，还有更大的破坏要来。"
            "有一天我们的文明，不论是升华还是浮华，都要成为过去。"
            "如果我最常用的字是“荒凉”，那是因为思想背景里有这惘惘的威胁。"
        ),
        "evidence_verified": True,
    },
    # Confirmed-correct items whose evidence未命中 flag was a false misfire.
    "zhang_传奇再版的话_1": {"evidence_verified": True},
    "zhang_到底是上海人_1": {"evidence_verified": True},
    "zhang_必也正名乎_2": {"evidence_verified": True},
    "zhang_必也正名乎_3": {"evidence_verified": True},
    "zhang_我看苏青_2": {"evidence_verified": True},
    "zhang_烬余录_3": {"evidence_verified": True},
    "zhang_论写作_3": {"evidence_verified": True},
    "zhang_诗与胡说_3": {"evidence_verified": True},
    "zhang_谈跳舞_2": {"evidence_verified": True},
    "zhang_谈跳舞_3": {"evidence_verified": True},
    "zhang_谈音乐_3": {"evidence_verified": True},
}

# Items the reviewer judged not cleanly fixable / out of scope — none dropped,
# but recorded for transparency.
DROP_IDS: set[str] = set()

PERSONAS = ("lu-xun", "zhang-ailing")
_ALLOWED = {"evidence", "evidence_verified", "topic", "options", "correct_index", "question"}


def _load_candidates(persona: str) -> list[dict]:
    path = STANCE / f"{persona}.candidates.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    applied: set[str] = set()
    for persona in PERSONAS:
        items = _load_candidates(persona)
        out, changes = [], []
        for it in items:
            iid = it["id"]
            if iid in DROP_IDS:
                changes.append(f"  DROP    {iid}")
                continue
            if iid in EDITS:
                edit = EDITS[iid]
                bad = set(edit) - _ALLOWED
                if bad:
                    raise ValueError(f"{iid}: unknown edit keys {bad}")
                # sanity: if reformulating, correct option must exist
                if "options" in edit:
                    ci = edit.get("correct_index", it["correct_index"])
                    if not (0 <= ci < len(edit["options"])):
                        raise ValueError(f"{iid}: correct_index {ci} out of range")
                fields = ", ".join(sorted(edit))
                it = {**it, **edit}
                changes.append(f"  EDIT    {iid}  [{fields}]")
                applied.add(iid)
            out.append(it)
        final_path = STANCE / f"{persona}.jsonl"
        final_path.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in out), encoding="utf-8")
        n_verified = sum(1 for x in out if x.get("evidence_verified"))
        print(f"[{persona}] {len(out)} items -> {final_path.name}  "
              f"(evidence_verified {n_verified}/{len(out)})")
        for c in changes:
            print(c)

    missing = set(EDITS) - applied
    if missing:
        raise SystemExit(f"ERROR: edits not matched to any item id: {sorted(missing)}")
    print(f"\nAll {len(EDITS)} edits applied; 0 unmatched.")


if __name__ == "__main__":
    main()
