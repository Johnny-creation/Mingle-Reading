from __future__ import annotations

import json, sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import SCHEMAS_DIR
from backend.api.schemas import QuestionRequest, BookRecord
from backend.agents.celebrity.answering import build_answer
from backend.agents.celebrity.chapter_summary import summarize_chapter
from backend.knowledge_graph.storage import graph_exists, load_graph

BOOK_ID = "百年孤独-根据马尔克斯指定版本翻译-未做任何增删-加西亚-马尔克斯-范晔-z-lib-org-rebuild-20260520-005151"
BENCHMARKS_DIR = ROOT / "backend" / "eval" / "benchmarks_100years"
RUNTIME_BOOK = ROOT / "backend" / "runtime" / "books" / f"{BOOK_ID}.json"


def load_book() -> BookRecord:
    data = json.loads(RUNTIME_BOOK.read_text(encoding="utf-8"))
    return BookRecord.model_validate(data)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _type_matches(value: Any, expected: str | list[str]) -> bool:
    expected_types = [expected] if isinstance(expected, str) else expected
    for et in expected_types:
        if et == "string" and isinstance(value, str): return True
        if et == "integer" and isinstance(value, int) and not isinstance(value, bool): return True
        if et == "number" and isinstance(value, (int, float)) and not isinstance(value, bool): return True
        if et == "object" and isinstance(value, dict): return True
        if et == "array" and isinstance(value, list): return True
        if et == "null" and value is None: return True
    return False


def validate_against_schema(sample: dict[str, Any], schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type and not _type_matches(sample, expected_type):
        return [f"{path}: expected type {expected_type!r}"]
    enum = schema.get("enum")
    if enum is not None and sample not in enum:
        errors.append(f"{path}: expected one of {enum!r}")
    if isinstance(sample, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in sample:
                errors.append(f"{path}.{key}: missing required field")
        if schema.get("additionalProperties") is False:
            extra_keys = sorted(set(sample) - set(properties))
            for key in extra_keys:
                errors.append(f"{path}.{key}: unexpected field")
        for key, value in sample.items():
            child_schema = properties.get(key)
            if child_schema:
                errors.extend(validate_against_schema(value, child_schema, f"{path}.{key}"))
    if isinstance(sample, list):
        item_schema = schema.get("items")
        if item_schema:
            for index, value in enumerate(sample):
                errors.extend(validate_against_schema(value, item_schema, f"{path}[{index}]"))
    if isinstance(sample, int):
        minimum, maximum = schema.get("minimum"), schema.get("maximum")
        if minimum is not None and sample < minimum: errors.append(f"{path}: expected >= {minimum}")
        if maximum is not None and sample > maximum: errors.append(f"{path}: expected <= {maximum}")
    if isinstance(sample, (int, float)) and not isinstance(sample, bool):
        minimum, maximum = schema.get("minimum"), schema.get("maximum")
        if minimum is not None and sample < minimum: errors.append(f"{path}: expected >= {minimum}")
        if maximum is not None and sample > maximum: errors.append(f"{path}: expected <= {maximum}")
    return errors


def evaluate_highlight_qa(book: BookRecord, schema: dict[str, Any]) -> dict[str, Any]:
    sample_files = sorted((BENCHMARKS_DIR / "highlight_qa").rglob("*.jsonl"))
    passed, failures = 0, []
    for sf in sample_files:
        for sample in load_jsonl(sf):
            errors = validate_against_schema(sample, schema)
            response = build_answer(
                QuestionRequest(
                    book_id=sample["book_id"],
                    question=sample["question"],
                    highlight_text=sample["highlight"]["text"],
                    current_chapter=sample["reader_progress"]["current_chapter_index"],
                    persona_id=sample.get("persona_id") or "neutral",
                ),
                book.chunks,
            )
            returned_chunk_ids = {c.chunk_id for c in response.contexts}
            expected_chunk_ids = set(sample["support_chunk_ids"])
            ok = (
                not errors
                and response.safe
                and bool(response.answer.strip())
                and expected_chunk_ids.issubset(returned_chunk_ids)
            )
            if ok:
                passed += 1
                print(f"  [PASS] {sample['sample_id']}")
            else:
                failures.append({
                    "sample_id": sample["sample_id"],
                    "errors": errors,
                    "response_safe": response.safe,
                    "response_reason": response.reason,
                    "returned_chunk_ids": sorted(returned_chunk_ids),
                    "expected_chunk_ids": sorted(expected_chunk_ids),
                })
                missing = expected_chunk_ids - returned_chunk_ids
                print(f"  [FAIL] {sample['sample_id']}: safe={response.safe}, answer_empty={not response.answer.strip()}, missing_chunks={sorted(missing)}")
    return {"sample_count": passed + len(failures), "passed": passed, "failed": len(failures), "failures": failures}


def _anti_spoiler_expectation_met(expected_behavior: str, response) -> bool:
    if expected_behavior == "refuse_future_plot":
        return response.safe is False and response.reason == "question_requests_future_plot"
    if expected_behavior in {"answer_within_boundary", "acknowledge_uncertainty", "reflect_on_known_text_only"}:
        return response.safe is True and bool(response.answer.strip())
    return False


def evaluate_anti_spoiler(book: BookRecord, schema: dict[str, Any]) -> dict[str, Any]:
    sample_files = sorted((BENCHMARKS_DIR / "anti_spoiler").rglob("*.jsonl"))
    passed, failures = 0, []
    for sf in sample_files:
        for sample in load_jsonl(sf):
            errors = validate_against_schema(sample, schema)
            response = build_answer(
                QuestionRequest(
                    book_id=sample["book_id"],
                    question=sample["prompt"],
                    current_chapter=sample["reader_progress"]["allowed_chapter_max"],
                    persona_id="neutral",
                ),
                book.chunks,
            )
            met = _anti_spoiler_expectation_met(sample["gold_label"]["expected_behavior"], response)
            ok = not errors and met
            if ok:
                passed += 1
                print(f"  [PASS] {sample['sample_id']}: expected={sample['gold_label']['expected_behavior']}, got safe={response.safe}")
            else:
                failures.append({
                    "sample_id": sample["sample_id"],
                    "errors": errors,
                    "response_safe": response.safe,
                    "response_reason": response.reason,
                    "expected_behavior": sample["gold_label"]["expected_behavior"],
                })
                print(f"  [FAIL] {sample['sample_id']}: expected={sample['gold_label']['expected_behavior']}, got safe={response.safe}, reason={response.reason}")
    return {"sample_count": passed + len(failures), "passed": passed, "failed": len(failures), "failures": failures}


def evaluate_chapter_summary(book: BookRecord) -> dict[str, Any]:
    sample_files = sorted((BENCHMARKS_DIR / "chapter_summary").rglob("*.jsonl"))
    passed, failures = 0, []
    for sf in sample_files:
        for sample in load_jsonl(sf):
            summary = summarize_chapter(book, current_chapter=sample["current_chapter_index"], persona_id=sample.get("persona_id", "neutral"))
            text = summary.summary
            missing = [p for p in sample["expected_phrases"] if p not in text]
            leaked = [p for p in sample.get("forbidden_phrases", []) if p in text]
            ok = not missing and not leaked
            if ok:
                passed += 1
                print(f"  [PASS] {sample['sample_id']}: all {len(sample['expected_phrases'])} phrases found, 0 leaked")
            else:
                failures.append({"sample_id": sample["sample_id"], "missing_phrases": missing, "forbidden_phrases_found": leaked})
                print(f"  [FAIL] {sample['sample_id']}: missing={missing}, leaked={leaked}")
    return {"sample_count": passed + len(failures), "passed": passed, "failed": len(failures), "failures": failures}


def run_evaluation() -> dict[str, Any]:
    print("📖 Loading 百年孤独...")
    book = load_book()
    print(f"   Title: {book.title}")
    print(f"   Chapters: {book.chapter_count}, Chunks: {len(book.chunks)}")
    print(f"   Graph built: {graph_exists(BOOK_ID)}")
    if graph_exists(BOOK_ID):
        g = load_graph(BOOK_ID)
        print(f"   Entities: {len(g.entities)}, Relations: {len(g.relations)}")

    hl_schema = json.loads((SCHEMAS_DIR / "highlight_qa.schema.json").read_text(encoding="utf-8"))
    as_schema = json.loads((SCHEMAS_DIR / "anti_spoiler_eval.schema.json").read_text(encoding="utf-8"))

    print("\n🔍 Highlight QA...")
    hl = evaluate_highlight_qa(book, hl_schema)

    print("\n🛡️ Anti-Spoiler...")
    asp = evaluate_anti_spoiler(book, as_schema)

    print("\n📝 Chapter Summary...")
    cs = evaluate_chapter_summary(book)

    total_passed = hl["passed"] + asp["passed"] + cs["passed"]
    total_failed = hl["failed"] + asp["failed"] + cs["failed"]
    result = {
        "book_id": BOOK_ID,
        "book_title": book.title,
        "highlight_qa": hl,
        "anti_spoiler": asp,
        "chapter_summary": cs,
        "overall": {"passed": total_passed, "failed": total_failed},
    }
    return result


def main() -> None:
    result = run_evaluation()
    print("\n" + "=" * 60)
    print("📊 百年孤独 Eval 结果")
    print("=" * 60)
    print(f"  Highlight QA:    {result['highlight_qa']['passed']}/{result['highlight_qa']['sample_count']} passed")
    print(f"  Anti-Spoiler:    {result['anti_spoiler']['passed']}/{result['anti_spoiler']['sample_count']} passed")
    print(f"  Chapter Summary: {result['chapter_summary']['passed']}/{result['chapter_summary']['sample_count']} passed")
    print(f"  ─────────────────────")
    print(f"  Overall:         {result['overall']['passed']}/{result['overall']['passed'] + result['overall']['failed']} passed")

    out_path = ROOT / "backend" / "eval" / "benchmarks_100years" / "result_100years.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Full result → {out_path}")


if __name__ == "__main__":
    main()
