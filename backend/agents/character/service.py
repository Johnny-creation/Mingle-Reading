from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend.api.schemas import (
    BookChunk,
    ChatMessage,
    CharacterCandidate,
    CharacterChatResponse,
    CharacterProfile,
    CharacterRelationship,
    InlineBubble,
)
from backend.agents.celebrity.model_client import invoke_openai_compatible_messages
from backend.agents.celebrity.persona_service import (
    PersonaAgentInvocationError,
    resolve_persona_runtime,
)
from backend.agents.celebrity.retrieval import retrieve_chunks
from backend.knowledge_graph.orchestration.models import ReadingProgress
from backend.knowledge_graph.orchestration.service import OrchestrationService
from backend.knowledge_graph.storage import graph_exists, load_graph
from backend.safety.anti_spoiler import is_spoiler_question

# ── Bubble tone injection from celebrity SKILL.md ──

_SKILL_ROOT = Path(__file__).resolve().parents[2] / "assets" / "Celebrity-skill" / "Celebrity-skill"

_PERSONA_SKILL_MAP = {
    "persona_lu_xun":       _SKILL_ROOT / "LuXun-skill-main" / "SKILL.md",
    "persona_zhang_ailing": _SKILL_ROOT / "ZhangAiLing-skill-main" / "SKILL.md",
    "persona_mark_twain":   _SKILL_ROOT / "MarkTwain-skill-main" / "SKILL.md",
}

# Hand-curated bubble tone snippets extracted from each SKILL.md 表达DNA section.
# These are concise so bubble generation stays fast — full thinking framework is
# for long-form answers, not 60-character marginalia.
_BUBBLE_TONE: dict[str, str] = {}

def _load_bubble_tone(persona_id: str) -> str:
    """Extract the language-style portion of SKILL.md for bubble tone injection."""
    if persona_id in _BUBBLE_TONE:
        return _BUBBLE_TONE[persona_id]

    skill_path = _PERSONA_SKILL_MAP.get(persona_id)
    if skill_path is None or not skill_path.exists():
        _BUBBLE_TONE[persona_id] = ""
        return ""

    text = skill_path.read_text(encoding="utf-8")
    # Strip YAML frontmatter
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2]

    # Extract language-style sections only — skip thinking methodology
    markers = [
        "语言风格", "表达DNA", "交流风格", "句式特征", "修辞策略",
        "节奏控制", "绝对禁止的温暖表达", "绝对禁令", "句式要求",
        "反指标", "不说教",
    ]
    lines: list[str] = []
    in_target = False
    for line in text.splitlines():
        stripped = line.strip()
        if any(m in stripped for m in markers):
            in_target = True
            lines.append(stripped)
            continue
        if in_target:
            if stripped.startswith("###") or stripped.startswith("## "):
                # Section boundary — keep going if it's still style-related
                if any(m in stripped for m in ["表达", "风格", "交流", "句式", "修辞", "节奏", "禁令", "指标"]):
                    lines.append(stripped)
                    continue
                else:
                    in_target = False
                    continue
            if stripped and not stripped.startswith("---"):
                lines.append(stripped)

    tone = "\n".join(lines[:60])

    # For bubbles, use ultra-short hand-crafted tone (~150 chars max).
    # Long SKILL.md excerpts cause 10-15s latency and confuse JSON output.
    _SHORT_TONE = {
        "persona_lu_xun": (
            "用鲁迅风格写批注：文白夹杂、冷峻锋利、短句如刀。"
            "可用「倘若…然而…」「大约…的确…」句式和破折号。"
            "拒绝温暖词汇和感叹号，批判即目的。"
        ),
        "persona_zhang_ailing": (
            "用张爱玲风格写批注：华丽克制、从物质细节切入。"
            "用「因为…所以…」制造宿命感，参差对照华美与苍凉。"
            "中英文可自然混杂，绝不说教。"
        ),
        "persona_mark_twain": (
            "用马克吐温风格写批注：口语化、扑克脸、轻描淡写。"
            "用最平实的词说最重的话，假谦虚包裹锋利观察。"
        ),
    }
    _BUBBLE_TONE[persona_id] = _SHORT_TONE.get(persona_id, tone)
    return _BUBBLE_TONE[persona_id]


_CHARACTER_PROFILE_CACHE: dict[tuple[str, str, int], CharacterProfile] = {}
_CHARACTER_CANDIDATE_CACHE: dict[tuple[str, int], list[CharacterCandidate]] = {}
_INLINE_BUBBLE_CACHE: dict[tuple[str, int, tuple[str, ...], str, str], list[InlineBubble]] = {}


def _extract_json_payload(text: str) -> Any:
    fenced = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1).strip())
    start_object = text.find("{")
    start_array = text.find("[")
    starts = [value for value in (start_object, start_array) if value >= 0]
    if not starts:
        raise ValueError("model response did not contain JSON")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        raise ValueError("model response did not contain a complete JSON payload")
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Model sometimes produces subtly broken JSON (missing commas, trailing text).
        # Try common repairs before giving up.
        repaired = re.sub(r'"\s*\n\s*"', '",\n"', candidate)  # missing comma between fields
        repaired = re.sub(r'}\s*\n\s*{', '},\n{', repaired)     # missing comma between objects
        repaired = re.sub(r']\s*\n\s*{', '],\n{', repaired)     # missing comma array→object
        return json.loads(repaired)


def _character_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9一-鿿]+", "-", name.lower()).strip("-")
    return slug or "candidate"


def _build_model_messages(
    system_prompt: str,
    user_prompt: str,
    history: list[ChatMessage] | None = None,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    for turn in (history or [])[-8:]:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _invoke_runtime(
    persona_id: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 900,
    temperature: float = 0.4,
) -> tuple[str, str]:
    _, api_key, base_url, model_name = resolve_persona_runtime(persona_id)
    try:
        answer = invoke_openai_compatible_messages(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        raise PersonaAgentInvocationError(f"character service model call failed: {exc}") from exc
    return answer, model_name


def _graph_character_candidates(book, current_chapter: int, limit: int = 10) -> list[CharacterCandidate]:
    if not graph_exists(book.book_id):
        return []
    try:
        graph = load_graph(book.book_id)
    except Exception:
        return []

    candidates: list[CharacterCandidate] = []
    for entity in graph.entities.values():
        if entity.entity_type != "character":
            continue
        if entity.first_seen_chapter > current_chapter:
            continue
        chapter_span = entity.metadata.get("chapter_span", []) if entity.metadata else []
        preview = entity.summary or f"{entity.canonical_name} appears in visible chapters."
        candidates.append(
            CharacterCandidate(
                character_id=f"char-{_character_slug(entity.canonical_name)}",
                character_name=entity.canonical_name,
                mention_count=entity.mention_count,
                chapter_hits=sorted(chapter_span) if chapter_span else [entity.first_seen_chapter],
                preview=preview,
            )
        )
    candidates.sort(key=lambda item: item.mention_count, reverse=True)
    return candidates[:limit]


def list_character_candidates(book, current_chapter: int, limit: int = 10) -> list[CharacterCandidate]:
    cache_key = (book.book_id, current_chapter)
    if cache_key in _CHARACTER_CANDIDATE_CACHE:
        return _CHARACTER_CANDIDATE_CACHE[cache_key][:limit]

    candidates = _graph_character_candidates(book, current_chapter, limit=200)
    deduped: list[CharacterCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.character_name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    _CHARACTER_CANDIDATE_CACHE[cache_key] = deduped
    return deduped[:limit]


def _character_evidence(
    chunks: list[BookChunk],
    character_name: str,
    current_chapter: int,
    top_k: int = 8,
) -> list[BookChunk]:
    visible = [chunk for chunk in chunks if chunk.chapter_index <= current_chapter]
    direct = [chunk for chunk in visible if character_name in chunk.text]
    if direct:
        return direct[:top_k]
    ranked = retrieve_chunks(visible, query=character_name, max_chapter=current_chapter, top_k=top_k)
    ranked_ids = {item.chunk_id for item in ranked}
    return [chunk for chunk in visible if chunk.chunk_id in ranked_ids][:top_k]


def _build_character_graph_block_from_network(network) -> str:
    """Build a graph knowledge block from an EntityNetworkResult (entity-centric retrieval)."""
    if network is None:
        return ""

    sections: list[str] = []

    if network.summary:
        sections.append(f"Character profile: {network.summary}")

    fam = [r for r in network.relations if r["relation_type"] == "FAMILY_OF"]
    if fam:
        lines = [f"- {r['source_name']} --[FAMILY_OF]--> {r['target_name']} | {r['fact']}" for r in fam]
        sections.append(f"Family relations ({len(fam)}):\n" + "\n".join(lines))

    inter = [r for r in network.relations if r["relation_type"] in ("SPOKE_WITH", "CARES_ABOUT", "CONFLICTS_WITH")]
    if inter:
        lines = [f"- {r['source_name']} --[{r['relation_type']}]--> {r['target_name']} | {r['fact']}" for r in inter[:12]]
        sections.append(f"Interactions ({len(inter)}):\n" + "\n".join(lines))

    other = [r for r in network.relations if r not in fam and r not in inter]
    if other:
        lines = [f"- {r['source_name']} --[{r['relation_type']}]--> {r['target_name']} | {r['fact']}" for r in other[:5]]
        sections.append(f"Other relations ({len(other)}):\n" + "\n".join(lines))

    # Source text evidence (first 3 relations with source_text)
    texts = [r for r in network.relations if r.get("source_text")]
    if texts:
        lines = [f"[ch{r['source_chapter']}] \"{r['source_text'][:200]}...\"" for r in texts[:3]]
        sections.append(f"Source text evidence:\n" + "\n".join(lines))

    if network.neighbour_entities:
        lines = [f"- {n['canonical_name']} ({n['entity_type']}, mc={n['mention_count']})" for n in network.neighbour_entities[:8]]
        sections.append(f"Related entities ({len(network.neighbour_entities)}):\n" + "\n".join(lines))

    return "\n\n".join(sections)


def generate_character_profile(book, character_name: str, current_chapter: int) -> CharacterProfile:
    cache_key = (book.book_id, character_name, current_chapter)
    if cache_key in _CHARACTER_PROFILE_CACHE:
        return _CHARACTER_PROFILE_CACHE[cache_key]

    evidence_chunks = _character_evidence(book.chunks, character_name, current_chapter, top_k=10)
    if not evidence_chunks:
        raise PersonaAgentInvocationError(f"character `{character_name}` has no visible evidence in current reading scope")

    # Entity-centric retrieval: pull the full ego-network
    try:
        graph = load_graph(book.book_id)
        network = OrchestrationService().retrieve_entity_network(
            graph, entity_name=character_name, max_chapter=current_chapter,
        )
    except Exception:
        network = None

    graph_block = _build_character_graph_block_from_network(network)
    evidence_block = "\n\n".join(
        f"[{chunk.chunk_id} | chapter {chunk.chapter_index}]\n{chunk.text}" for chunk in evidence_chunks
    )

    system_prompt = (
        "你是一个严格受阅读进度约束的人物分析助手。"
        "只根据用户当前可见章节中的证据生成人物画像，不要使用未来剧情。"
        "请输出 JSON，字段必须包含 summary, core_traits, relationships, signature_tension, current_scope。"
        "relationships 必须是对象数组，每项包含 target 和 description。"
    )
    user_prompt = (
        f"书名: {book.title}\n"
        f"当前可见章节: {current_chapter}\n"
        f"人物: {character_name}\n\n"
        f"图谱上下文:\n{graph_block or '无'}\n\n"
        f"正文证据:\n{evidence_block}"
    )
    answer, model_name = _invoke_runtime(
        "neutral",
        _build_model_messages(system_prompt, user_prompt),
        max_tokens=1100,
        temperature=0.25,
    )
    payload = _extract_json_payload(answer)
    relationships = [
        CharacterRelationship(
            target=str(item.get("target", "")).strip(),
            description=str(item.get("description", "")).strip(),
        )
        for item in payload.get("relationships", [])
        if str(item.get("target", "")).strip() and str(item.get("description", "")).strip()
    ]
    profile = CharacterProfile(
        character_id=f"char-{_character_slug(character_name)}",
        character_name=character_name,
        summary=str(payload.get("summary", "")).strip(),
        core_traits=[str(item).strip() for item in payload.get("core_traits", []) if str(item).strip()],
        relationships=relationships,
        signature_tension=str(payload.get("signature_tension", "")).strip(),
        evidence_chunk_ids=[chunk.chunk_id for chunk in evidence_chunks],
        current_scope=str(payload.get("current_scope", "")).strip(),
        model_name=model_name,
        entity_id=network.entity_id if network is not None else "",
        arc_summary=network.summary if network is not None else "",
        visible_relationship_count=len(network.relations) if network is not None else 0,
        evidence_count=len(evidence_chunks),
    )
    _CHARACTER_PROFILE_CACHE[cache_key] = profile
    return profile


def answer_as_character(
    book,
    character_name: str,
    question: str,
    current_chapter: int,
    conversation_history: list[ChatMessage] | None = None,
    top_k: int = 6,
) -> CharacterChatResponse:
    safety = is_spoiler_question(question)
    if not safety.safe:
        return CharacterChatResponse(
            answer="你的问题涉及未来剧情，已根据当前阅读进度为你过滤。请改问已读范围内的问题。",
            character_name=character_name,
            safe=False,
            reason=safety.reason,
            profile=CharacterProfile(
                character_id=f"char-{_character_slug(character_name)}",
                character_name=character_name,
                summary="",
            ),
        )

    profile = generate_character_profile(book, character_name, current_chapter)
    evidence_chunks = _character_evidence(book.chunks, character_name, current_chapter, top_k=top_k)
    retrieval_hits = retrieve_chunks(
        [chunk for chunk in book.chunks if chunk.chapter_index <= current_chapter],
        query=f"{character_name} {question}",
        max_chapter=current_chapter,
        top_k=top_k,
    )
    seen = {chunk.chunk_id for chunk in evidence_chunks}
    for hit in retrieval_hits:
        if hit.chunk_id in seen:
            continue
        match = next((chunk for chunk in book.chunks if chunk.chunk_id == hit.chunk_id), None)
        if match is not None:
            evidence_chunks.append(match)
            seen.add(hit.chunk_id)

    # Entity-centric retrieval with query for relation ordering
    try:
        graph = load_graph(book.book_id)
        network = OrchestrationService().retrieve_entity_network(
            graph, entity_name=character_name, query=question, max_chapter=current_chapter,
        )
    except Exception:
        network = None

    graph_block = _build_character_graph_block_from_network(network)
    evidence_block = "\n\n".join(
        f"[{chunk.chunk_id} | chapter {chunk.chapter_index}]\n{chunk.text}" for chunk in evidence_chunks[:top_k]
    )
    system_prompt = (
        f"你现在扮演 {character_name}。"
        "你只能基于当前可见章节中的事实回答，不能泄露未来剧情，不能引用读者尚未看到的信息。"
        "如果证据不足，可以保留、迟疑、模糊，但不要编造未来事实。"
    )
    user_prompt = (
        f"书名: {book.title}\n"
        f"当前可见章节: {current_chapter}\n"
        f"角色摘要: {profile.summary}\n"
        f"核心特征: {', '.join(profile.core_traits)}\n"
        f"核心张力: {profile.signature_tension}\n"
        f"用户问题: {question}\n\n"
        f"图谱上下文:\n{graph_block or '无'}\n\n"
        f"正文证据:\n{evidence_block}"
    )
    answer, model_name = _invoke_runtime(
        "neutral",
        _build_model_messages(system_prompt, user_prompt, conversation_history),
        max_tokens=900,
        temperature=0.5,
    )
    return CharacterChatResponse(
        answer=answer.strip(),
        character_name=character_name,
        safe=True,
        reason="within_visible_scope",
        model_name=model_name,
        profile=profile,
    )


def generate_inline_bubbles(
    book,
    current_chapter: int,
    visible_chunk_ids: list[str],
    persona_id: str,
    assistant_mode: str,
    character_name: str,
    max_bubbles: int,
) -> list[InlineBubble]:
    cache_key = (book.book_id, current_chapter, tuple(sorted(visible_chunk_ids)), assistant_mode, character_name or persona_id)
    if cache_key in _INLINE_BUBBLE_CACHE:
        return _INLINE_BUBBLE_CACHE[cache_key]

    visible_chunks = [chunk for chunk in book.chunks if chunk.chunk_id in set(visible_chunk_ids)]
    if not visible_chunks:
        return []
    max_policy_bubbles = 5 if len(visible_chunks) == 1 else 3
    max_bubbles = max(1, min(max_bubbles, max_policy_bubbles))

    evidence_block = "\n\n".join(f"[{chunk.chunk_id}]\n{chunk.text}" for chunk in visible_chunks[:16])
    if assistant_mode == "character" and character_name:
        runtime_persona = "neutral"
        instruction = f"请以 {character_name} 的视角生成贴在文段旁边的短评气泡。"
        tone_block = ""
    else:
        runtime_persona = persona_id
        instruction = "请生成面向读者的短评气泡。"
        # Inject celebrity language-style DNA from SKILL.md
        tone_block = _load_bubble_tone(persona_id)
        if tone_block:
            tone_block = (
                "你必须用以下名家的语言风格来写气泡批注——不是写长文，"
                "而是像这个名家在书页边缘随手批注：\n"
                + tone_block
                + "\n\n"
            )

    # Map of short IDs → full IDs so the model can use either
    short_to_full: dict[str, str] = {}
    for chunk in visible_chunks[:16]:
        parts = chunk.chunk_id.rsplit("-c0", 1)
        if len(parts) == 2:
            short_to_full["-c0" + parts[1]] = chunk.chunk_id
        parts = chunk.chunk_id.rsplit("-p", 1)
        if len(parts) == 2:
            short_to_full["p" + parts[1]] = chunk.chunk_id

    system_prompt = (
        tone_block
        + "你是一个为阅读器生成行内批注气泡的助手。"
        "请输出 JSON 数组，每项包含 chunk_id, anchor_text, label, comment, bubble_type。"
        "【重要】chunk_id 必须使用下方正文中 [方括号] 内的完整 ID，原样复制。"
        "anchor_text 必须直接出现在对应 chunk 的正文中，一字不差。"
        "comment 最多 60 个字，label 最多 6 个字。"
        "bubble_type 可选 detail, emotion, relation, theme, question, character_inner_voice。"
        "只输出 JSON 数组，不要任何其他文字。"
    )
    user_prompt = (
        f"书名: {book.title}\n"
        f"当前可见章节: {current_chapter}\n"
        f"任务说明: {instruction}\n"
        f"最多生成 {max_bubbles} 条。\n\n"
        f"可见正文:\n{evidence_block}"
    )
    answer, _ = _invoke_runtime(
        runtime_persona,
        _build_model_messages(system_prompt, user_prompt),
        max_tokens=500,
        temperature=0.4,
    )
    try:
        payload = _extract_json_payload(answer)
    except (ValueError, json.JSONDecodeError) as exc:
        # Model JSON is sometimes unrepairable — return empty rather than 500
        _INLINE_BUBBLE_CACHE[cache_key] = []
        return []
    # Accept both {"bubbles": [...]} wrapper and bare [...]
    if isinstance(payload, dict):
        payload = payload.get("bubbles", payload.get("items", []))
    chunk_map = {chunk.chunk_id: chunk for chunk in visible_chunks}
    bubbles: list[InlineBubble] = []
    if isinstance(payload, list):
        for index, item in enumerate(payload[:max_bubbles], start=1):
            chunk_id = str(item.get("chunk_id", "")).strip()
            anchor_text = str(item.get("anchor_text", "")).strip()
            label = str(item.get("label", "")).strip()[:8]
            comment = str(item.get("comment", "")).strip()[:60]
            emphasis = str(item.get("emphasis", "detail")).strip()
            bubble_type = str(item.get("bubble_type", emphasis or "detail")).strip()
            chunk = chunk_map.get(chunk_id)
            # Fallback: the model may have truncated the chunk_id (e.g. "c001-p001")
            if not chunk and chunk_id and chunk_id not in chunk_map:
                for full_id in chunk_map:
                    if full_id.endswith(chunk_id) or full_id.endswith("-" + chunk_id):
                        chunk = chunk_map[full_id]
                        break
            if not chunk or not anchor_text or anchor_text not in chunk.text or not comment:
                continue
            if bubble_type not in {"detail", "emotion", "relation", "theme", "question", "character_inner_voice"}:
                bubble_type = "detail"
            bubbles.append(
                InlineBubble(
                    bubble_id=f"bubble-{chunk_id}-{index}",
                    chunk_id=chunk_id,
                    anchor_text=anchor_text,
                    label=label or "细读",
                    comment=comment,
                    emphasis=emphasis if emphasis in {"theme", "emotion", "relation", "foreshadow", "detail"} else "detail",
                    bubble_type=bubble_type,
                    citation_chunk_ids=[chunk_id],
                    trigger_reason="selected-passage" if len(visible_chunks) == 1 else "page-salience-policy",
                )
            )
    _INLINE_BUBBLE_CACHE[cache_key] = bubbles
    return bubbles
