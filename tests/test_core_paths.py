from __future__ import annotations

from pathlib import Path

from src.core import paths

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"


def test_repo_root_is_the_directory_holding_pyproject() -> None:
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()
    assert (paths.REPO_ROOT / "src").is_dir()


def test_derived_paths_hang_off_repo_root() -> None:
    assert paths.CORPUS_DIR == paths.REPO_ROOT / "corpus"
    assert paths.INGESTION_OUTPUT_DIR == paths.REPO_ROOT / "ingestion" / "output"
    assert paths.CHUNKS_FILE == paths.REPO_ROOT / "ingestion" / "output" / "chunks.jsonl"
    assert paths.RETRIEVAL_OUTPUT_DIR == paths.REPO_ROOT / "retrieval" / "output"
    assert paths.EVAL_DIR == paths.REPO_ROOT / "eval"
    assert paths.EVAL_REPORTS_DIR == paths.REPO_ROOT / "eval" / "reports"


def test_paths_module_imports_nothing_but_pathlib() -> None:
    """gate_holdout_integrity.py deliberately redefined CHUNKS_FILE locally so
    the integrity guard would not pull in the embedder import chain. Binding it
    from here is only an improvement if this module stays that cheap."""
    import ast

    module = SRC_ROOT / "core" / "paths.py"
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "pathlib"}, f"paths.py must stay dependency-free, imported: {imported}"


def test_no_other_module_rolls_its_own_repo_root_chain() -> None:
    """`Path(__file__).resolve().parent.parent.parent[.parent]` is off by one the
    moment a module moves between package depths — exactly what the router move
    in Task 1 does. src/core/paths.py is the single place allowed to compute it."""
    offenders: dict[str, int] = {}
    for py_file in SRC_ROOT.rglob("*.py"):
        if py_file == SRC_ROOT / "core" / "paths.py":
            continue
        count = py_file.read_text(encoding="utf-8").count("parent.parent.parent")
        if count:
            offenders[str(py_file.relative_to(SRC_ROOT))] = count
    assert not offenders, f"modules computing their own repo root: {offenders}"
