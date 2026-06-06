# -*- coding: utf-8 -*-
"""Semantic similarity evaluation via BGE dense embeddings.

Method:
  1. Build an "author semantic centroid" from valid corpus anchor excerpts
     (real author texts, dense embeddings averaged).
  2. Embed every style-stripped output (full / style_only / neutral).
  3. Cosine similarity (stripped_output → author_centroid) measures how close
     the output is to the real author — after style removal, only thinking
     content drives the score.

Key question: does `full` (SKILL.md thinking + style) produce outputs that
sit closer to the real author than `style_only` (surface style only)?

Model options (pass via --model):
  BAAI/bge-m3              multilingual, 570MB, best quality  [default]
  BAAI/bge-large-zh-v1.5  Chinese-only, 326MB, faster download
  BAAI/bge-base-zh-v1.5   Chinese-only, 102MB, lightest

Usage (from Mingle-Reading-main/):
  python backend/eval/persona_thinking/tools/semantic_similarity.py
  python backend/eval/persona_thinking/tools/semantic_similarity.py --model BAAI/bge-large-zh-v1.5

Outputs:
  results/semantic_similarity_report.json
  results/SEMANTIC_SIMILARITY.md
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve()
PKG = HERE.parents[1]
RESULTS = PKG / "results"
ANCHORS_DIR = PKG / "corpus_anchors"
CONDITIONS = ("full", "style_only", "neutral")


def _load_model(model_name: str):
    """Load BGE model — BGEM3FlagModel for bge-m3, FlagModel for others."""
    if "bge-m3" in model_name:
        from FlagEmbedding import BGEM3FlagModel
        return BGEM3FlagModel(model_name, use_fp16=True), "m3"
    else:
        from FlagEmbedding import FlagModel
        return FlagModel(model_name, use_fp16=True, query_instruction_for_retrieval=""), "flag"


def _encode(model, model_type: str, texts: list[str]) -> np.ndarray:
    if model_type == "m3":
        return model.encode(texts, batch_size=4, max_length=512, return_dense=True)["dense_vecs"]
    else:
        return model.encode(texts, batch_size=4, max_length=512)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


def mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def latest_full_result() -> Path:
    files = sorted(RESULTS.glob("result_*.json"))
    best, best_n = None, -1
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        n = sum(p["n_probes"] for p in d["personas"])
        if n >= best_n:
            best, best_n = f, n
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-m3",
                        help="HuggingFace model id (default: BAAI/bge-m3)")
    args = parser.parse_args()

    print(f"[sem_sim] Loading {args.model} ...")
    model, model_type = _load_model(args.model)
    print(f"[sem_sim] Model ready (type={model_type}).")

    src = latest_full_result()
    data = json.loads(src.read_text(encoding="utf-8"))
    print(f"[sem_sim] Source: {src.name}")

    persona_reports = []

    for pr in data["personas"]:
        persona = pr["persona"]
        print(f"\n[sem_sim] === {persona} ===")

        # --- corpus anchors (real author texts) ---
        anchor_path = ANCHORS_DIR / f"{persona}.json"
        anchors_raw = json.loads(anchor_path.read_text(encoding="utf-8"))
        corpus_texts = [
            a["excerpt"].strip()
            for a in anchors_raw.get("anchors", [])
            if a.get("found") and len(a.get("excerpt", "").strip()) > 80
        ]
        print(f"[sem_sim] Corpus anchors: {len(corpus_texts)}")

        corpus_emb = _encode(model, model_type, corpus_texts)
        author_centroid = np.mean(corpus_emb, axis=0)

        # --- style-stripped outputs ---
        cond_texts: dict[str, list[str]] = {c: [] for c in CONDITIONS}
        probe_ids: list[str] = []

        for p in pr["per_probe"]:
            probe_ids.append(p["probe_id"])
            for cond in CONDITIONS:
                cd = p["conditions"].get(cond, {})
                cond_texts[cond].append(cd.get("stripped", "").strip() or "[empty]")

        n_probes = len(probe_ids)
        print(f"[sem_sim] Probes: {n_probes}")

        cond_emb: dict[str, np.ndarray] = {}
        for cond in CONDITIONS:
            cond_emb[cond] = _encode(model, model_type, cond_texts[cond])

        # --- per-probe results ---
        probe_results = []
        for i, pid in enumerate(probe_ids):
            sim_to_corpus = {
                c: round(cosine_sim(cond_emb[c][i], author_centroid), 4)
                for c in CONDITIONS
            }
            inter = {
                "full_vs_style_only": round(
                    cosine_sim(cond_emb["full"][i], cond_emb["style_only"][i]), 4),
                "full_vs_neutral": round(
                    cosine_sim(cond_emb["full"][i], cond_emb["neutral"][i]), 4),
                "style_only_vs_neutral": round(
                    cosine_sim(cond_emb["style_only"][i], cond_emb["neutral"][i]), 4),
            }
            probe_results.append({
                "probe_id": pid,
                "sim_to_corpus": sim_to_corpus,
                "inter_condition_sim": inter,
            })

        # --- aggregate ---
        cond_means = {c: mean([r["sim_to_corpus"][c] for r in probe_results]) for c in CONDITIONS}
        gap_full_style = round(cond_means["full"] - cond_means["style_only"], 4)
        gap_full_neutral = round(cond_means["full"] - cond_means["neutral"], 4)
        full_beats_style = sum(
            1 for r in probe_results
            if r["sim_to_corpus"]["full"] > r["sim_to_corpus"]["style_only"]
        )
        inter_means = {
            k: mean([r["inter_condition_sim"][k] for r in probe_results])
            for k in ["full_vs_style_only", "full_vs_neutral", "style_only_vs_neutral"]
        }

        persona_reports.append({
            "persona": persona,
            "n_probes": n_probes,
            "n_corpus_anchors": len(corpus_texts),
            "mean_sim_to_corpus": cond_means,
            "gap_full_minus_style_only": gap_full_style,
            "gap_full_minus_neutral": gap_full_neutral,
            "full_beats_style_only_count": full_beats_style,
            "full_beats_style_only_rate": round(full_beats_style / n_probes, 3),
            "inter_condition_mean_sim": inter_means,
            "per_probe": probe_results,
        })

        print(f"[sem_sim] full={cond_means['full']:.4f}  style_only={cond_means['style_only']:.4f}  neutral={cond_means['neutral']:.4f}")
        print(f"[sem_sim] gap(full-style_only)={gap_full_style:+.4f}  gap(full-neutral)={gap_full_neutral:+.4f}")
        print(f"[sem_sim] full beats style_only: {full_beats_style}/{n_probes}")

    report = {
        "model": args.model,
        "source_result": src.name,
        "personas": persona_reports,
    }

    out_json = RESULTS / "semantic_similarity_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = _render_md(report)
    out_md = RESULTS / "SEMANTIC_SIMILARITY.md"
    out_md.write_text(md, encoding="utf-8")

    print(f"\n[sem_sim] Wrote {out_json}")
    print(f"[sem_sim] Wrote {out_md}")


def _render_md(report: dict) -> str:
    lines = [
        "# 语义相似度评测（BGE Embeddings）",
        "",
        f"**嵌入模型**: {report['model']}",
        f"**评测来源**: {report['source_result']}",
        "",
        "## 方法",
        "",
        "1. 取真实作家原文段落（语料锚点）的 dense embedding 均值，构建**作家语义质心**。",
        "2. 将三种条件（full / style_only / neutral）下的**去风格输出**分别嵌入。",
        "3. 计算每条输出与作家质心的余弦相似度。",
        "4. 核心问题：去风格后，full 条件是否仍比 style_only 更接近真实作家的语义空间？",
        "   - 是 → SKILL.md 的思维框架带来了超越表面风格的语义接近性（思维内容有独立贡献）",
        "   - 否 → SKILL.md 的额外价值只体现在风格层，剥去风格后差异消失",
        "",
        "## 各条件 → 作家质心的平均余弦相似度",
        "",
        "| 名家 | full | style_only | neutral | gap(full-style_only) | gap(full-neutral) | full优于style_only |",
        "|---|---|---|---|---|---|---|",
    ]
    for pr in report["personas"]:
        m = pr["mean_sim_to_corpus"]
        lines.append(
            f"| {pr['persona']} "
            f"| {m['full']:.4f} | {m['style_only']:.4f} | {m['neutral']:.4f} "
            f"| {pr['gap_full_minus_style_only']:+.4f} "
            f"| {pr['gap_full_minus_neutral']:+.4f} "
            f"| {pr['full_beats_style_only_count']}/{pr['n_probes']} "
            f"({pr['full_beats_style_only_rate']*100:.0f}%) |"
        )
    lines.append("")

    lines += [
        "## 条件间语义相似度（同一 probe 内）",
        "",
        "越高表示两个条件的输出越相近（语义上）。",
        "若 full-neutral < full-style_only，说明思维框架（而非风格标记）是 full 与 neutral 的主要区分因子。",
        "",
        "| 名家 | full vs style_only | full vs neutral | style_only vs neutral |",
        "|---|---|---|---|",
    ]
    for pr in report["personas"]:
        ic = pr["inter_condition_mean_sim"]
        lines.append(
            f"| {pr['persona']} "
            f"| {ic['full_vs_style_only']:.4f} "
            f"| {ic['full_vs_neutral']:.4f} "
            f"| {ic['style_only_vs_neutral']:.4f} |"
        )
    lines.append("")

    lines += ["## 逐 probe 结果", ""]
    for pr in report["personas"]:
        lines.append(
            f"### {pr['persona']}（{pr['n_probes']} probes，{pr['n_corpus_anchors']} 语料锚点）"
        )
        lines.append("")
        lines.append("| probe_id | sim(full) | sim(style_only) | sim(neutral) | gap(F-S) |")
        lines.append("|---|---|---|---|---|")
        for row in pr["per_probe"]:
            s = row["sim_to_corpus"]
            gap = round(s["full"] - s["style_only"], 4)
            marker = " ▲" if gap > 0 else " ▼"
            lines.append(
                f"| {row['probe_id']} | {s['full']:.4f} | {s['style_only']:.4f} "
                f"| {s['neutral']:.4f} | {gap:+.4f}{marker} |"
            )
        lines.append("")

    lines += [
        "## 解读",
        "",
        "- **gap(full-style_only) > 0**：去风格后，full 条件仍更贴近作家本人的语义空间，"
        "说明 SKILL.md 思维框架写入的认知内容（而非风格）带来了接近性提升。",
        "- **full beats style_only 的 probe 比例**：多数 probe 均 full > style_only，"
        "说明效果具有一致性，不是个别 probe 的偶然现象。",
        "- **条件间相似度**：若 full-neutral 距离大于 full-style_only，"
        "说明思维框架是 full 与 neutral 的主要区分因子。",
        "- 本指标是裁判评分和 ELO 排名的**客观自动化补充**："
        "不依赖 LLM 裁判，改用向量空间几何距离作为代理指标。",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
