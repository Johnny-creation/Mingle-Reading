# -*- coding: utf-8 -*-
"""Pretty-print blind judge tasks in a range, for the cross-family judge (Claude)
to read and score. Read-only; does not touch results.

Usage (from Mingle-Reading-main/):
    python -X utf8 -m backend.eval.persona_thinking.tools.show_judge_tasks \
        --name layer1b_lu-xun --start 0 --count 15
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
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=15)
    args = ap.parse_args()
    path = HANDOFF / f"{args.name}.judge_tasks.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    total = len(rows)
    end = min(args.start + args.count, total)
    print(f"### {args.name}: tasks {args.start}..{end-1} of {total}\n")
    for i in range(args.start, end):
        t = rows[i]
        print(f"===== [{i}] task_id={t['task_id']} =====")
        print(f"【开头/setup】\n{t['setup']}\n")
        print(f"【参照/reference（作者真实下一步）】\n{t['reference']}\n")
        print(f"【候选/candidate】\n{t['candidate']}\n")


if __name__ == "__main__":
    main()
