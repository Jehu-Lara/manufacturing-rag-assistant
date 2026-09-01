from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

# regression_queries.json stays at eval/ alongside eval_set.json — a frozen
# regression dataset is data, not application code, and isn't packaged into the
# installable wheel under src/ (mirrors eval_set_integrity).
REGRESSION_SET_FILE = Path(__file__).resolve().parent.parent.parent.parent / "eval" / "regression_queries.json"


def canonical_queries_bytes(queries: list[dict[str, Any]]) -> bytes:
    return json.dumps(queries, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_hash(queries: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_queries_bytes(queries)).hexdigest()


def load_regression_set(path: Path = REGRESSION_SET_FILE) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return cast("dict[str, Any]", json.load(f))


def verify(path: Path = REGRESSION_SET_FILE) -> None:
    data = load_regression_set(path)
    expected = data["sha256"]
    actual = compute_hash(data["queries"])
    if actual != expected:
        raise ValueError(
            f"{path} sha256 mismatch — stored {expected}, computed {actual}. "
            "If this edit was intentional, bump 'version' and re-run "
            "`python -m src.features.evaluation.regression_set_integrity --write`."
        )


def write(path: Path = REGRESSION_SET_FILE) -> None:
    data = load_regression_set(path)
    data["sha256"] = compute_hash(data["queries"])
    with path.open("w", encoding="utf-8", newline="\n") as f:
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
        print(f"{REGRESSION_SET_FILE}: hash OK")
    else:
        write()
        print(f"{REGRESSION_SET_FILE}: hash regenerated")


if __name__ == "__main__":
    main()
