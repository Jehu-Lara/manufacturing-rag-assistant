from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

# eval_set.json stays at its original eval/ location, byte-unchanged — a
# golden dataset is data, not application code, and isn't packaged into the
# installable wheel under src/ (Resolved Decision #12).
EVAL_SET_FILE = Path(__file__).resolve().parent.parent.parent.parent / "eval" / "eval_set.json"


def canonical_questions_bytes(questions: list[dict[str, Any]]) -> bytes:
    return json.dumps(questions, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_hash(questions: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_questions_bytes(questions)).hexdigest()


def load_eval_set(path: Path = EVAL_SET_FILE) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return cast("dict[str, Any]", json.load(f))


def verify(path: Path = EVAL_SET_FILE) -> None:
    data = load_eval_set(path)
    expected = data["sha256"]
    actual = compute_hash(data["questions"])
    if actual != expected:
        raise ValueError(
            f"{path} sha256 mismatch — stored {expected}, computed {actual}. "
            "If this edit was intentional, bump 'version' and re-run "
            "`python -m src.features.evaluation.eval_set_integrity --write`."
        )


def write(path: Path = EVAL_SET_FILE) -> None:
    data = load_eval_set(path)
    data["sha256"] = compute_hash(data["questions"])
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true")
    group.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if args.verify:
        verify()
        print(f"{EVAL_SET_FILE}: hash OK")
    else:
        write()
        print(f"{EVAL_SET_FILE}: hash regenerated")


if __name__ == "__main__":
    main()
