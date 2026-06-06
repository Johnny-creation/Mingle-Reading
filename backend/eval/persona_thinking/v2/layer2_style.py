# -*- coding: utf-8 -*-
"""Layer-2: the STYLE axis (S) of the double dissociation.

Each condition writes a short prose analysis of the SAME contemporary probes
(reused from v1). The HEADLINE S metric is `style_ratio` — directional
signature-marker intensity relative to the author's own prose (HIGHER = more
author-like style); Burrows's Delta is recorded as a descriptive cross-check.
S must move with the +style factor and stay flat across the thinking factor:

    ratio(full) ~ ratio(style_only)  >>  ratio(thinking_only) ~ ratio(neutral)

Paired by probe so it lines up with the stats layer. Prose (not the Layer-1
multiple-choice answers) is required here because style is only measurable on
free text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.eval.persona_thinking.common import call_model, load_probes
from backend.eval.persona_thinking.stylometry import stylometry
from backend.eval.persona_thinking.v2.conditions_v2 import (
    CONDITIONS,
    CONTROL_CONDITIONS,
    system_prompt_for,
)
from backend.eval.persona_thinking.v2.style_metrics import (
    author_style_baseline,
    delta_to_author,
    style_intensity,
    style_ratio,
)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
ALL_CONDITIONS = list(CONDITIONS) + list(CONTROL_CONDITIONS)


def generate_prose(persona: str, probe: dict, condition: str, *, use_cache: bool) -> str:
    system = system_prompt_for(persona, condition)
    return call_model(
        endpoint_key=persona,
        system_prompt=system,
        user_prompt=probe["prompt"],
        temperature=0.7,
        max_tokens=700,
        use_cache=use_cache,
        tag=f"v2_prose::{persona}::{probe['id']}::{condition}",
    ).strip()


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def evaluate(persona: str, *, use_cache: bool, max_probes: int | None) -> dict[str, Any]:
    probes = load_probes(persona)
    if max_probes:
        probes = probes[:max_probes]
    per_probe: list[dict] = []
    for probe in probes:
        cond_out: dict[str, Any] = {}
        for cond in ALL_CONDITIONS:
            text = generate_prose(persona, probe, cond, use_cache=use_cache)
            cond_out[cond] = {
                "text": text,
                "style_ratio": style_ratio(text, persona),          # PRIMARY S
                "style_intensity": style_intensity(text, persona),  # raw density
                "delta_to_author": delta_to_author(text, persona),  # descriptive
                "stylometry": stylometry(text, persona),
            }
        per_probe.append({"probe_id": probe["id"], "conditions": cond_out})

    # paired-by-probe lists + means. `ratios` is the headline S; deltas descriptive.
    ratios = {c: [p["conditions"][c]["style_ratio"] for p in per_probe] for c in ALL_CONDITIONS}
    deltas = {c: [p["conditions"][c]["delta_to_author"] for p in per_probe] for c in ALL_CONDITIONS}
    summary = {
        c: {
            "mean_ratio": _mean(ratios[c]),
            "ratios": ratios[c],
            "mean_delta": _mean(deltas[c]),
            "deltas": deltas[c],
        }
        for c in ALL_CONDITIONS
    }
    return {
        "persona": persona,
        "n_probes": len(per_probe),
        "author_style_baseline": author_style_baseline(persona),
        "summary": summary,
        "per_probe": per_probe,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", nargs="+", default=["lu-xun", "zhang-ailing"])
    ap.add_argument("--max-probes", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for persona in args.personas:
        rep = evaluate(persona, use_cache=not args.no_cache, max_probes=args.max_probes)
        out = RESULTS_DIR / f"layer2_style__{persona}.json"
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        s = rep["summary"]
        print(f"\n[{persona}] n_probes={rep['n_probes']}  author baseline density={rep['author_style_baseline']}/100char")
        print(f"  {'cond':14s} {'style_ratio(↑)':>14s} {'delta(↓desc)':>13s}")
        for c in ALL_CONDITIONS:
            print(f"  {c:14s} {str(s[c]['mean_ratio']):>14s} {str(s[c]['mean_delta']):>13s}")
        print(f"  -> wrote {out.name}")


if __name__ == "__main__":
    main()
