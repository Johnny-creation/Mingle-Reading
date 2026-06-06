# -*- coding: utf-8 -*-
"""Blind cross-family judge handoff (file-based).

The user has Claude/GPT memberships but no API key, so we cannot wire a
cross-family judge into the Python pipeline. Instead the pipeline EXPORTS a
blind, anonymized judging task file; a cross-family judge (Claude running in
Claude Code, i.e. a different model family from the DeepSeek generator —
exactly the self-preference-bias control we want) fills in scores in-session;
the pipeline IMPORTS them back by task_id and merges.

This keeps the hardest evidence (Layer-1a stance accuracy) judge-free, and uses
the judge only for the reference-based Layer-1b reasoning-continuation scoring,
where the judge always sees the author's REAL continuation as the reference
(reference-based judging is far more reliable than reference-free).

Export schema (one JSON object per line in <name>.judge_tasks.jsonl):
    {"task_id", "persona", "setup", "reference", "candidate", "rubric"}
  - condition labels are NOT included (blind); a salted task_id hides them.
Result schema (the judge writes <name>.judge_results.jsonl):
    {"task_id", "score", "note"}        # score in {0,1,2}
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
HANDOFF_DIR = HERE / "judge_handoff"

RUBRIC = (
    "对照【参考】(作者本人接下来的真实推理)，判断【候选】是否做出了同一个推理动作 / 得到同一判断："
    "2=核心推理动作与结论一致（措辞可不同）；1=方向相关但偏浅或只对一半；0=不同的论点、跑题或泛泛而谈。"
)


def _task_id(salt: str, persona: str, item_id: str, condition: str) -> str:
    h = hashlib.sha256(f"{salt}|{persona}|{item_id}|{condition}".encode("utf-8")).hexdigest()
    return h[:16]


def export_tasks(
    name: str,
    rows: list[dict[str, Any]],
    *,
    salt: str = "mingle-v2",
    seed: int = 7,
) -> tuple[Path, Path]:
    """rows: [{persona, item_id, condition, setup, reference, candidate}].
    Writes a blind, shuffled task file + a private key file mapping task_id ->
    (persona, item_id, condition) for re-merge. Returns (tasks_path, key_path)."""
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    tasks, key = [], {}
    for r in rows:
        tid = _task_id(salt, r["persona"], r["item_id"], r["condition"])
        key[tid] = {"persona": r["persona"], "item_id": r["item_id"], "condition": r["condition"]}
        tasks.append({
            "task_id": tid,
            "persona": r["persona"],
            "setup": r["setup"],
            "reference": r["reference"],
            "candidate": r["candidate"],
            "rubric": RUBRIC,
        })
    random.Random(seed).shuffle(tasks)
    tasks_path = HANDOFF_DIR / f"{name}.judge_tasks.jsonl"
    key_path = HANDOFF_DIR / f"{name}.judge_key.json"
    tasks_path.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in tasks), encoding="utf-8")
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    return tasks_path, key_path


def import_results(name: str) -> dict[str, dict[str, Any]]:
    """Merge <name>.judge_results.jsonl with the key file ->
    {(persona,item_id,condition): {score, note}} keyed by a tuple-string."""
    key = json.loads((HANDOFF_DIR / f"{name}.judge_key.json").read_text(encoding="utf-8"))
    res_path = HANDOFF_DIR / f"{name}.judge_results.jsonl"
    if not res_path.exists():
        raise FileNotFoundError(f"judge results not found: {res_path} (the cross-family judge must fill it in)")
    out: dict[str, dict[str, Any]] = {}
    for line in res_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        meta = key.get(r["task_id"])
        if not meta:
            continue
        k = f"{meta['persona']}::{meta['item_id']}::{meta['condition']}"
        out[k] = {"score": r.get("score"), "note": r.get("note", "")}
    return out
