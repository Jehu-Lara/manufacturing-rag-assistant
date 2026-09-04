from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

from src.core.paths import CHUNKS_FILE, EVAL_DIR
from src.features.evaluation import eval_set_integrity, regression_set_integrity

# CHUNKS_FILE comes from src.core.paths, which imports nothing but pathlib —
# so this guard still never pulls in the embedder import chain, the reason it
# used to redefine the path locally.
EVAL_SET_FILE = EVAL_DIR / "eval_set.json"
REGRESSION_SET_FILE = EVAL_DIR / "regression_queries.json"

# gate_holdout_v1.0.0.json stays at eval/ alongside eval_set.json and
# regression_queries.json — a frozen holdout is data, not application code, and
# is never packaged under src/ (mirrors eval_set_integrity /
# regression_set_integrity). The hash is over the parsed `questions`, not raw
# bytes, so reformatting the file without touching content keeps it valid.
GATE_HOLDOUT_FILE = EVAL_DIR / "gate_holdout_v1.0.0.json"

REQUIRED_TOTAL = 48
REQUIRED_PER_CLASS = 24
REQUIRED_PER_LANGUAGE = 24
REQUIRED_PAIRS = 24
_LANGUAGES = ("en", "es")
_COMMON_FIELDS = ("id", "pair_id", "question", "language", "answerable")


def canonical_questions_bytes(questions: list[dict[str, Any]]) -> bytes:
    return json.dumps(questions, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_hash(questions: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_questions_bytes(questions)).hexdigest()


def load_gate_holdout(path: Path = GATE_HOLDOUT_FILE) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return cast("dict[str, Any]", json.load(f))


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _known_chunk_ids(chunks_path: Path) -> set[str]:
    ids: set[str] = set()
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["chunk_id"])
    return ids


def _forbidden_question_texts(eval_set_path: Path, regression_set_path: Path) -> set[str]:
    """The holdout is de-duplicated against the FROZEN eval_set / regression
    sets — both must be present and pass their own integrity check, so the
    comparison can never be silently narrowed by a missing or tampered file."""
    for path, name in ((eval_set_path, "eval_set.json"), (regression_set_path, "regression_queries.json")):
        if not path.exists():
            raise ValueError(
                f"{name} not found at {path} - the holdout is de-duplicated against "
                f"the frozen {name}; it must be present and frozen before the holdout verifies"
            )
    eval_set_integrity.verify(eval_set_path)
    regression_set_integrity.verify(regression_set_path)

    forbidden: set[str] = set()
    for question in eval_set_integrity.load_eval_set(eval_set_path).get("questions", []):
        forbidden.add(_normalized(question["question"]))
    for query in regression_set_integrity.load_regression_set(regression_set_path).get("queries", []):
        forbidden.add(_normalized(query["query"]))
    return forbidden


def _require_nonempty_str(question: dict[str, Any], field: str, where: str) -> None:
    value = question.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: missing/empty required field {field!r}")


def _validate_question(
    question: dict[str, Any], known_chunk_ids: set[str], forbidden_texts: set[str]
) -> None:
    qid = question.get("id", "<no id>")
    where = f"gate holdout question {qid!r}"
    for field in _COMMON_FIELDS:
        if field not in question:
            raise ValueError(f"{where}: missing required field {field!r}")
    _require_nonempty_str(question, "id", where)
    _require_nonempty_str(question, "pair_id", where)
    _require_nonempty_str(question, "question", where)
    if question["language"] not in _LANGUAGES:
        raise ValueError(f"{where}: language must be one of {_LANGUAGES}")
    if not isinstance(question["answerable"], bool):
        raise ValueError(f"{where}: 'answerable' must be a bool")

    if _normalized(question["question"]) in forbidden_texts:
        raise ValueError(
            f"{where}: question text collides (normalized) with eval_set v1.1.0 or "
            "regression_queries.json — the holdout must not reuse or trivially "
            "rephrase those; author an independent intent"
        )

    if question["answerable"]:
        expected = question.get("expected_chunk_ids")
        if not isinstance(expected, list) or not expected:
            raise ValueError(f"{where}: answerable question needs a non-empty 'expected_chunk_ids'")
        for chunk_id in expected:
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(f"{where}: 'expected_chunk_ids' has an empty entry")
            if chunk_id not in known_chunk_ids:
                raise ValueError(f"{where}: expected_chunk_id {chunk_id!r} is not in chunks.jsonl")
        _require_nonempty_str(question, "expected_answer", where)
    else:
        _require_nonempty_str(question, "absence_note", where)


def _validate_pairs(questions: list[dict[str, Any]]) -> None:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        by_pair[question["pair_id"]].append(question)
    if len(by_pair) != REQUIRED_PAIRS:
        raise ValueError(f"gate holdout must have {REQUIRED_PAIRS} unique pair_ids, got {len(by_pair)}")
    for pair_id, members in by_pair.items():
        if len(members) != 2:
            raise ValueError(f"pair_id {pair_id!r} has {len(members)} members, expected 2 (one EN, one ES)")
        if sorted(m["language"] for m in members) != ["en", "es"]:
            raise ValueError(f"pair_id {pair_id!r} is not one EN + one ES")
        if len({m["answerable"] for m in members}) != 1:
            raise ValueError(f"pair_id {pair_id!r}: EN and ES disagree on 'answerable'")
        if members[0]["answerable"]:
            chunk_id_sets = [frozenset(m.get("expected_chunk_ids", [])) for m in members]
            if chunk_id_sets[0] != chunk_id_sets[1]:
                raise ValueError(
                    f"pair_id {pair_id!r}: the EN and ES answerable halves must target the "
                    f"same expected_chunk_ids (got {sorted(chunk_id_sets[0])} vs {sorted(chunk_id_sets[1])})"
                )


def validate_composition(
    data: dict[str, Any],
    *,
    chunks_path: Path = CHUNKS_FILE,
    eval_set_path: Path = EVAL_SET_FILE,
    regression_set_path: Path = REGRESSION_SET_FILE,
) -> None:
    """Structural gate: a green integrity check must mean a real, frozen,
    balanced 48-question / 24-pair holdout — never an empty draft."""
    if data.get("status") != "frozen":
        raise ValueError(
            "gate holdout status must be 'frozen' before it can be used - "
            "author the questions and run `gate_holdout_integrity --write`"
        )
    if not chunks_path.exists():
        raise ValueError(
            f"{chunks_path} not found - build it (`python -m src.features.ingestion.cli`) "
            "before verifying the holdout; expected_chunk_ids are checked against it"
        )
    questions = data.get("questions")
    if not isinstance(questions, list) or len(questions) != REQUIRED_TOTAL:
        raise ValueError(
            f"gate holdout must have exactly {REQUIRED_TOTAL} questions, "
            f"got {len(questions) if isinstance(questions, list) else 'non-list'}"
        )

    ids = [q.get("id") for q in questions]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"gate holdout has duplicate ids: {duplicates}")

    known_chunk_ids = _known_chunk_ids(chunks_path)
    forbidden_texts = _forbidden_question_texts(eval_set_path, regression_set_path)
    for question in questions:
        _validate_question(question, known_chunk_ids, forbidden_texts)

    normalized_by_text: dict[str, list[str]] = defaultdict(list)
    for question in questions:
        normalized_by_text[_normalized(question["question"])].append(str(question["id"]))
    internal_dupes = {text: ids for text, ids in normalized_by_text.items() if len(ids) > 1}
    if internal_dupes:
        raise ValueError(
            "gate holdout has questions that are identical after normalization "
            f"(NFKC + casefold + whitespace): {internal_dupes}. Each EN and each ES "
            "question must be a distinct intent; a paired EN/ES question is not a duplicate "
            "because the two are in different languages."
        )

    _validate_pairs(questions)

    answerable = sum(1 for q in questions if q["answerable"] is True)
    unanswerable = sum(1 for q in questions if q["answerable"] is False)
    if answerable != REQUIRED_PER_CLASS or unanswerable != REQUIRED_PER_CLASS:
        raise ValueError(
            f"gate holdout must be {REQUIRED_PER_CLASS} answerable / "
            f"{REQUIRED_PER_CLASS} unanswerable, got {answerable}/{unanswerable}"
        )

    by_language = Counter(q["language"] for q in questions)
    if any(by_language.get(lang) != REQUIRED_PER_LANGUAGE for lang in _LANGUAGES):
        raise ValueError(
            f"gate holdout must be {REQUIRED_PER_LANGUAGE} EN / {REQUIRED_PER_LANGUAGE} ES, "
            f"got {dict(by_language)}"
        )


def verify(
    path: Path = GATE_HOLDOUT_FILE,
    *,
    chunks_path: Path = CHUNKS_FILE,
    eval_set_path: Path = EVAL_SET_FILE,
    regression_set_path: Path = REGRESSION_SET_FILE,
) -> None:
    data = load_gate_holdout(path)
    expected = data["sha256"]
    actual = compute_hash(data["questions"])
    if actual != expected:
        raise ValueError(
            f"{path} sha256 mismatch — stored {expected}, computed {actual}. "
            "If this edit was intentional, bump 'version' and re-run "
            "`python -m src.features.evaluation.gate_holdout_integrity --write`."
        )
    validate_composition(
        data,
        chunks_path=chunks_path,
        eval_set_path=eval_set_path,
        regression_set_path=regression_set_path,
    )


def write(path: Path = GATE_HOLDOUT_FILE) -> None:
    """Atomic: a crash mid-write must never leave a truncated frozen dataset.
    Does NOT author or generate questions — it only re-stamps the sha256 over
    whatever `questions` a human has already placed in the file."""
    data = load_gate_holdout(path)
    data["sha256"] = compute_hash(data["questions"])
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true")
    group.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if args.verify:
        verify()
        print(f"{GATE_HOLDOUT_FILE}: hash + composition OK")
    else:
        write()
        print(f"{GATE_HOLDOUT_FILE}: hash regenerated (composition NOT checked; run --verify)")


if __name__ == "__main__":
    main()
