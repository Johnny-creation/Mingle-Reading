# -*- coding: utf-8 -*-
"""Shared plumbing for the persona-thinking evaluation.

Importing this module loads backend.config (which reads the workspace-root
.env and resolves the DeepSeek credentials for every persona + the neutral
agent). All model calls go through here and are cached on disk so reruns and
incremental development do not re-spend tokens.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                       # Mingle-Reading-main/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.config  # noqa: F401  (side effect: creates runtime dirs, loads its own .env)
from backend.agents.celebrity.model_client import invoke_openai_compatible_messages


def _bootstrap_env() -> None:
    """Load .env from the likely locations, independent of backend.config's own
    loader. The credentials live in the workspace-root .env (one level above
    Mingle-Reading-main/); this keeps the eval runnable wherever .env sits, and
    without requiring any change to backend/config.py. Set-if-absent, so real
    environment variables always win."""
    for env_path in (ROOT.parent / ".env", ROOT / ".env"):
        if not env_path.exists():
            continue
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and not os.environ.get(key, "").strip():
                os.environ[key] = value


_bootstrap_env()

RUBRICS_DIR = HERE / "rubrics"
PROBES_DIR = HERE / "probes"
ANCHORS_DIR = HERE / "corpus_anchors"
RESULTS_DIR = HERE / "results"
CACHE_DIR = HERE / "_cache"

# Persona id -> the env-var prefix used in .env. We call DeepSeek directly with
# custom system prompts rather than the product RAG pipeline, because the claim
# under test is about the *thinking model*, not the book-companion plumbing.
PERSONA_ENV = {
    "lu-xun": ("LU_XUN_API_KEY", "LU_XUN_BASE_URL", "LU_XUN_MODEL_NAME"),
    "zhang-ailing": ("ZHANG_AILING_API_KEY", "ZHANG_AILING_BASE_URL", "ZHANG_AILING_MODEL_NAME"),
    "neutral": ("MUSE_NEUTRAL_API_KEY", "MUSE_NEUTRAL_BASE_URL", "MUSE_NEUTRAL_MODEL_NAME"),
}

# The judge / style-stripper run on the neutral (DeepSeek) credentials. NOTE:
# judge and agent share the DeepSeek family — a self-preference bias risk that
# DESIGN.md documents and mitigates (corpus-grounded scoring + blind conditions
# + forced-choice). Swapping in a cross-family judge is just an env change here.
JUDGE_ENV = "neutral"

PERSONA_PACK_PATH = {
    "lu-xun": "backend/assets/Celebrity-skill/Celebrity-skill/LuXun-skill-main/SKILL.md",
    "zhang-ailing": "backend/assets/Celebrity-skill/Celebrity-skill/ZhangAiLing-skill-main/SKILL.md",
}


@dataclass
class ModelEndpoint:
    api_key: str
    base_url: str
    model_name: str


def resolve_endpoint(persona_or_judge: str) -> ModelEndpoint:
    keys = PERSONA_ENV.get(persona_or_judge)
    if keys is None:
        raise KeyError(f"unknown endpoint key: {persona_or_judge}")
    api_env, base_env, model_env = keys
    ep = ModelEndpoint(
        api_key=os.getenv(api_env, "").strip(),
        base_url=os.getenv(base_env, "").strip(),
        model_name=os.getenv(model_env, "").strip(),
    )
    if not (ep.api_key and ep.base_url and ep.model_name):
        raise RuntimeError(f"endpoint `{persona_or_judge}` not fully configured (check .env: {api_env}/{base_env}/{model_env})")
    return ep


def _cache_key(parts: dict[str, Any]) -> str:
    blob = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def call_model(
    *,
    endpoint_key: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 900,
    json_object: bool = False,
    use_cache: bool = True,
    tag: str = "",
) -> str:
    """Single chat call with disk caching. `tag` only affects the cache key /
    bookkeeping, letting callers force-separate otherwise-identical prompts."""
    ep = resolve_endpoint(endpoint_key)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(
        {
            "model": ep.model_name,
            "base": ep.base_url,
            "sys": system_prompt,
            "user": user_prompt,
            "temp": temperature,
            "max": max_tokens,
            "json": json_object,
            "tag": tag,
        }
    )
    cache_file = CACHE_DIR / f"{key}.json"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))["content"]

    response_format = {"type": "json_object"} if json_object else None
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            content = invoke_openai_compatible_messages(
                api_key=ep.api_key,
                base_url=ep.base_url,
                model_name=ep.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=120,
                response_format=response_format,
            )
            cache_file.write_text(
                json.dumps({"content": content, "model": ep.model_name}, ensure_ascii=False),
                encoding="utf-8",
            )
            return content
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"call_model failed after retries ({endpoint_key}): {last_err}")


def parse_json_loose(text: str) -> Any:
    """DeepSeek json_object mode is usually clean, but strip fences just in case."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        if t.startswith("json"):
            t = t[4:]
    t = t.strip()
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1:
        t = t[start : end + 1]
    return json.loads(t)


# ----- data loaders -----

def load_rubric(persona: str) -> dict[str, Any]:
    return json.loads((RUBRICS_DIR / f"{persona}.json").read_text(encoding="utf-8"))


def load_probes(persona: str) -> list[dict[str, Any]]:
    rows = []
    for line in (PROBES_DIR / f"{persona}.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_anchors(persona: str) -> list[dict[str, Any]]:
    path = ANCHORS_DIR / f"{persona}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [a for a in data.get("anchors", []) if a.get("found") and a.get("excerpt")]


def skill_body(persona: str) -> str:
    """Return the SKILL.md body (frontmatter stripped) — the actual thinking
    model the product deploys, used as the `full` condition system prompt."""
    rel = PERSONA_PACK_PATH[persona]
    text = (ROOT / rel).read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return text.strip()


def anchors_block(persona: str, max_chars_each: int = 900) -> str:
    anchors = load_anchors(persona)
    if not anchors:
        return "（暂无原文锚点，请依据你对该作家公认思维方式的理解评分。）"
    lines = []
    for a in anchors:
        ex = a["excerpt"][:max_chars_each]
        moves = "、".join(a.get("moves", []))
        lines.append(f"【{a['label']}｜体现的思维动作：{moves}】\n{ex}")
    return "\n\n".join(lines)
