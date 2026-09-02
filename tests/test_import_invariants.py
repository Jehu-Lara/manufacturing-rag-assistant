from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_MODULES = {"fastapi", "chromadb", "groq", "openai", "streamlit", "torch"}
SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
DOMAIN_ROOT = SRC_ROOT / "domain"
WEB_ROOT = SRC_ROOT / "web"

# src/web/ talks to the backend over HTTP only (CLAUDE.md's module-boundary
# rule). These are the packages it must never reach into directly, even though
# they ship in the same container.
WEB_FORBIDDEN_PACKAGES = ("src.domain", "src.features", "src.adapters")


def _imported_module_paths(py_file: Path) -> set[str]:
    """Full dotted paths, not just the top-level name: `src` alone can't
    distinguish `from src.web.i18n import ...` (allowed) from
    `from src.domain.models import ...` (forbidden). Relative imports are
    resolved against the file's own package so `from ..domain import x`
    cannot slip through as level > 0."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    try:
        package_parts = py_file.resolve().parent.relative_to(SRC_ROOT.parent).parts
    except ValueError:
        package_parts = ()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    modules.add(node.module)
            else:
                base = package_parts[: len(package_parts) - (node.level - 1)]
                suffix = (node.module,) if node.module else ()
                modules.add(".".join(base + suffix))
    return modules


def _imported_top_level_modules(py_file: Path) -> set[str]:
    return {module.split(".")[0] for module in _imported_module_paths(py_file)}


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


def test_web_layer_never_imports_backend_packages_directly() -> None:
    """src/web/ is a separate process reaching the API over HTTP (ADR-005/007).
    A direct import would still "work" in the repo and in the single deploy
    container, and would silently make the API no longer independently
    testable or reachable — exactly the coupling the deploy shape denies."""
    violations: dict[str, set[str]] = {}
    for py_file in WEB_ROOT.rglob("*.py"):
        found = {
            module
            for module in _imported_module_paths(py_file)
            for package in WEB_FORBIDDEN_PACKAGES
            if module == package or module.startswith(package + ".")
        }
        if found:
            violations[str(py_file)] = found
    assert not violations, f"src/web imported backend packages directly: {violations}"


def test_web_import_check_detects_an_injected_violation(tmp_path: Path) -> None:
    """Guards the guard: a checker that silently matched nothing would pass the
    test above forever."""
    offender = tmp_path / "src" / "web" / "offender.py"
    offender.parent.mkdir(parents=True)
    offender.write_text("from src.domain.models import Citation\n", encoding="utf-8")

    assert "src.domain.models" in _imported_module_paths(offender)
