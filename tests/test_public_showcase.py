from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIVE_DEMO_URL = "https://jehulara-manufacturing-rag-assistant-live.hf.space"


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_public_docs_link_the_live_demo_without_stale_pending_claims() -> None:
    readme = _read("README.md")
    spec = _read("SPEC.md")
    space_readme = _read("docs/hf-space/README.md")
    showcase = _read("docs/hf-space/index.html")

    assert LIVE_DEMO_URL in readme
    assert LIVE_DEMO_URL in space_readme
    assert LIVE_DEMO_URL in showcase
    assert "No live interactive deployment exists yet" not in readme
    assert "deployment in progress" not in showcase
    assert "Not yet operational" not in spec


def test_public_showcase_reports_served_retrieval_profile_and_scope() -> None:
    showcase = _read("docs/hf-space/index.html")

    assert "contextual-v1/off" in showcase
    assert "0.887" in showcase
    assert "en 0.917" in showcase
    assert "es 0.844" in showcase
    assert "public functional portfolio demo" in showcase
