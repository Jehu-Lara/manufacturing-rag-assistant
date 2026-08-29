from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from src.adapters.secondary.embedder import sentence_transformers_embedder as embedder_module

ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> str:
    # Normalize CRLF -> LF: on a Windows checkout with core.autocrlf=true the
    # working-tree files carry CRLF, which would break multi-line substring
    # assertions like "permissions:\n  contents: read". CI (Linux) checks out
    # LF, so normalizing here makes the assertions checkout-independent.
    return (ROOT / path).read_text(encoding="utf-8").replace("\r\n", "\n")


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
    directives = "\n".join(
        line for line in dockerfile.splitlines() if not line.lstrip().startswith("#")
    )

    assert "python:3.11-slim-bookworm@sha256:" in dockerfile
    assert dockerfile.count("pip install --no-cache-dir") == 1
    assert "-r requirements-lock.txt" in dockerfile
    assert "--require-hashes" in dockerfile
    # apt packages are intentionally NOT version-pinned: exact Debian pins
    # vanish from the mirror when the security pocket rotates (breaking every
    # future rebuild) and freeze known-old nginx/openssl. The digest-pinned
    # base image + hash-pinned pip set already cover meaningful reproducibility;
    # `apt-get update` on that fixed base pulls current security patches for
    # the three low-risk packages (nginx, tini, ca-certificates).
    assert "apt-get install -y --no-install-recommends" in directives
    for pkg in ("ca-certificates", "nginx", "tini"):
        assert pkg in directives
    assert "nginx=" not in directives
    assert "ca-certificates=" not in directives
    assert "tini=" not in directives
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
    assert "location = /redoc" in nginx
    assert "location = /openapi.json" in nginx
    assert "--host 127.0.0.1 --port 8000" in start
    assert "--server.address 127.0.0.1" in start
    assert "wait -n" in start
    assert "trap 'shutdown 143' TERM" in start
    # Streamlit must launch via `python -m streamlit`, never the bare console
    # script: only `-m` puts the container WORKDIR on sys.path for the app's
    # `from src.web import client`. Behaviourally guarded by
    # test_streamlit_ui_executes_from_repo_root_without_pythonpath.
    assert "python -m streamlit run src/web/app.py" in start
    assert re.search(r"(?<!python -m )streamlit run", start) is None


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
    assert "--require-hashes -r requirements-hf-lock.txt" in workflow
    assert "huggingface-hub==1.29.0" in _read("requirements-hf-lock.txt")
    assert "cp -R src corpus staging/" in workflow
    assert "find staging -type l" in workflow
    assert "gitleaks dir staging" in workflow
    assert "--max-archive-depth 3" in workflow
    assert "--max-decode-depth 3" in workflow
    assert "--delete '*'" in workflow


def test_ci_supply_chain_is_read_only_sha_pinned_and_hash_enforced():
    workflow = _read(".github/workflows/ci.yml")
    lock = _read("requirements-lock.txt")
    ci_lock = _read("requirements-ci-lock.txt")

    assert "permissions:\n  contents: read" in workflow
    assert "--require-hashes" in workflow
    assert "requirements-ci-lock.txt" in workflow
    assert "actions/checkout@v4" not in workflow
    assert "actions/setup-python@v5" not in workflow
    assert "actions/cache@v4" not in workflow
    assert lock.count("--hash=sha256:") == 120
    assert "ruff==0.16.5" in ci_lock
    assert "mypy==2.3.1" in ci_lock


def test_public_proxy_sets_defensive_headers_without_unsafe_streamlit_csp():
    nginx = _read("nginx.conf")
    web_app = _read("src/web/app.py")
    web_render = _read("src/web/render.py")

    nginx_directives = "\n".join(
        line for line in nginx.splitlines() if not line.lstrip().startswith("#")
    )

    assert 'add_header X-Content-Type-Options "nosniff" always;' in nginx
    assert 'add_header Referrer-Policy "no-referrer" always;' in nginx
    # frame-ancestors CSP instead of X-Frame-Options: allows the Hugging Face
    # cross-origin iframe embed while still constraining who may frame the app.
    assert "add_header X-Frame-Options" not in nginx_directives
    assert (
        'add_header Content-Security-Policy '
        '"frame-ancestors \'self\' https://huggingface.co https://*.hf.space" always;'
    ) in nginx_directives
    # Only frame-ancestors — no default-src/script-src that would break the
    # Streamlit frontend's inline and eval'd JS.
    assert "script-src" not in nginx_directives
    assert "default-src" not in nginx_directives
    assert "privacy_warning" in web_app
    assert "unsafe_allow_html=True" not in web_render


def test_space_card_declares_docker_cpu_basic_contract():
    card = _read("deploy/hf-space/README.md")

    assert "sdk: docker" in card
    assert "app_port: 7860" in card
    assert "license: mit" in card
    assert "suggested_hardware: cpu-basic" in card
    assert "BAAI/bge-m3" in card


def test_streamlit_ui_executes_from_repo_root_without_pythonpath(tmp_path):
    """Boot the UI exactly as the container's start.sh does -- `python -m
    streamlit` (so `-m` puts the CWD on sys.path, the whole point of the
    d4c681d fix), CWD = repo root, PYTHONPATH stripped -- then hit
    /_stcore/script-health-check. That endpoint actually runs app.py and
    returns 503 if it raises, so a ModuleNotFoundError, a broken import in the
    src.web chain, or an exception in main() on a no-backend cold start all
    fail here (plain /_stcore/health would not -- it only reports the server
    is up)."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    log_path = tmp_path / "streamlit.log"
    log = log_path.open("wb")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "src/web/app.py",
            "--server.port", str(port),
            "--server.address", "127.0.0.1",
            "--server.headless", "true",
            # Hidden/experimental Streamlit option, pinned via requirements-lock.txt:
            # without it the route 404s to the SPA catch-all (a misleading 200).
            "--server.scriptHealthCheckEnabled", "true",
        ],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    log.close()
    last: tuple[int, str] | None = None
    try:
        url = f"http://127.0.0.1:{port}/_stcore/script-health-check"
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(url, timeout=20) as response:
                    last = (response.status, response.read(120).decode("utf-8", "replace"))
                    break
            except urllib.error.HTTPError as exc:
                # The endpoint responded -- it ran the script. Any status is a
                # verdict (503 "error"/"timeout"), so stop; no point retrying.
                last = (exc.code, exc.read(120).decode("utf-8", "replace"))
                break
            except (urllib.error.URLError, OSError):
                # Server not accepting connections yet -- keep polling.
                time.sleep(0.5)

        output = log_path.read_text(encoding="utf-8", errors="replace")
        assert "ModuleNotFoundError" not in output, output
        assert last is not None, f"script-health-check never responded on :{port}\n{output}"
        assert last == (200, "ok"), f"script-health-check returned {last}\n{output}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)

    assert proc.returncode is not None
