from __future__ import annotations

from pathlib import Path

from src.adapters.secondary.embedder import sentence_transformers_embedder as embedder_module

ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_embedder_pins_model_revision(monkeypatch):
    captured: dict[str, str] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, *, revision: str) -> None:
            captured["model_name"] = model_name
            captured["revision"] = revision

    monkeypatch.setattr(embedder_module, "SentenceTransformer", FakeSentenceTransformer)

    embedder = embedder_module.SentenceTransformersEmbedder()
    embedder._get_model()

    assert captured == {
        "model_name": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
    }


def test_dockerfile_reproducibility_and_runtime_contract():
    dockerfile = _read("Dockerfile")

    assert "python:3.11-slim-bookworm@sha256:" in dockerfile
    assert dockerfile.count("pip install --no-cache-dir") == 1
    assert "-r requirements-lock.txt" in dockerfile
    assert "pip check" in dockerfile
    assert "tini" in dockerfile
    assert 'ENTRYPOINT ["/usr/bin/tini", "--"]' in dockerfile
    assert "USER user" in dockerfile
    assert "COPY --chown=1000:1000 . ." in dockerfile
    assert "ENV HF_HOME=/home/user/.cache/huggingface" in dockerfile
    assert "ENV HF_HUB_OFFLINE=1" in dockerfile
    assert "RUN chown -R" not in dockerfile


def test_docker_context_excludes_private_and_nonruntime_content():
    dockerignore = _read(".dockerignore")

    for excluded in (".claude/", ".git/", ".venv/", "docs/", "eval/", "tests/", "CLAUDE.md", "SPEC.md"):
        assert excluded in dockerignore


def test_only_nginx_is_public_and_internal_api_routes_are_hidden():
    nginx = _read("nginx.conf")
    start = _read("start.sh")

    assert "listen 7860;" in nginx
    assert "access_log /dev/stdout;" in nginx
    assert "error_log /dev/stderr warn;" in nginx
    assert "server_tokens off;" in nginx
    assert "location = /query" in nginx and "return 404;" in nginx
    assert "location = /docs" in nginx
    assert "location = /openapi.json" in nginx
    assert "--host 127.0.0.1 --port 8000" in start
    assert "--server.address 127.0.0.1" in start
    assert "wait -n" in start
    assert "trap 'shutdown 143' TERM" in start


def test_manual_oidc_deployment_is_sha_gated_and_allowlisted():
    workflow = _read(".github/workflows/deploy-hf-space.yml")

    assert "workflow_dispatch:" in workflow
    assert "environment: hf-live" in workflow
    assert "id-token: write" in workflow
    assert "actions: read" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "persist-credentials: false" in workflow
    assert "git merge-base --is-ancestor" in workflow
    assert '.conclusion == "success"' in workflow
    assert "HF_OIDC_RESOURCE: spaces/JehuLara/manufacturing-rag-assistant-live" in workflow
    assert "huggingface_hub==1.29.0" in workflow
    assert "cp -R src corpus staging/" in workflow
    assert "--delete '*'" in workflow


def test_space_card_declares_docker_cpu_basic_contract():
    card = _read("deploy/hf-space/README.md")

    assert "sdk: docker" in card
    assert "app_port: 7860" in card
    assert "license: mit" in card
    assert "suggested_hardware: cpu-basic" in card
    assert "BAAI/bge-m3" in card
