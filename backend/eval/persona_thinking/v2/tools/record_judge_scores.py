# -*- coding: utf-8 -*-
"""Append a batch of cross-family judge scores to <name>.judge_results.jsonl.

The cross-family judge (Claude) reads blind tasks and records scores here in
batches, so the long task dumps need not stay resident in context. Idempotent:
re-recording a task_id overwrites its prior line (last write wins), so a batch
can be safely re-run.

Usage (from Mingle-Reading-main/):
    python -X utf8 -m backend.eval.persona_thinking.v2.tools.record_judge_scores \
        --name layer1b_lu-xun --json '[{"task_id":"..","score":1,"note":".."}, ...]'
Or pass a path to a json file with --file.

A third (or further) judge writes its own file with --judge, e.g. --judge codex
appends to <name>.judge_results_codex.jsonl (what judge_agreement.py expects for
the codex judge). Default (no --judge) writes <name>.judge_results.jsonl (claude).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
HANDOFF = HERE / "judge_handoff"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--judge", default="", help="judge suffix, e.g. 'codex' -> ..._codex.jsonl; "
                    "empty (default) -> claude's .judge_results.jsonl")
    ap.add_argument("--json", default=None, help="inline JSON list of {task_id,score,note}")
    ap.add_argument("--file", default=None, help="path to JSON list file")
    args = ap.parse_args()
    if args.json:
        batch = json.loads(args.json)
    elif args.file:
        batch = json.loads(Path(args.file).read_text(encoding="utf-8"))
    else:
        raise SystemExit("need --json or --file")

    tail = ".judge_results.jsonl" if not args.judge else f".judge_results_{args.judge}.jsonl"
    out = HANDOFF / f"{args.name}{tail}"
    existing: dict[str, dict] = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                existing[r["task_id"]] = r
    for r in batch:
        assert r["score"] in (0, 1, 2), f"bad score for {r['task_id']}: {r['score']}"
        existing[r["task_id"]] = {"task_id": r["task_id"], "score": r["score"], "note": r.get("note", "")}
    out.write_text("\n".join(json.dumps(existing[k], ensure_ascii=False) for k in existing), encoding="utf-8")
    print(f"[record] {args.name}: +{len(batch)} this batch; {len(existing)} total recorded -> {out.name}")


if __name__ == "__main__":
    main()
