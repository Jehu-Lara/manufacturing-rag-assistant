from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_MODULES = {"fastapi", "chromadb", "groq", "openai", "streamlit", "torch"}
DOMAIN_ROOT = Path(__file__).resolve().parent.parent / "src" / "domain"


def _imported_top_level_modules(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules


def test_domain_imports_no_framework_or_adapter_libraries() -> None:
    """Static AST parsing, not a runtime sys.modules check: a runtime check
    would be contaminated by other tests in the same pytest process having
    already imported fastapi/chromadb/etc regardless of whether src.domain
    itself imports them, and would miss a violation in a domain module
    nothing else happens to import during the run. AST parsing is
    per-file, hermetic, and needs no process isolation."""
    violations: dict[str, set[str]] = {}
    for py_file in DOMAIN_ROOT.rglob("*.py"):
        found = _imported_top_level_modules(py_file) & FORBIDDEN_MODULES
        if found:
            violations[str(py_file)] = found
    assert not violations, f"src/domain imported forbidden modules: {violations}"
