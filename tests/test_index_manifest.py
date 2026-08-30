from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from src.adapters.secondary.embedder.sentence_transformers_embedder import MODEL_NAME, MODEL_REVISION
from src.features.retrieval import index_manifest


def _write_mini_corpus(root: Path) -> None:
    (root / "public").mkdir(parents=True)
    (root / "synthetic").mkdir(parents=True)
    (root / "public" / "alpha.md").write_text("# Alpha\n\nbody a\n", encoding="utf-8")
    (root / "synthetic" / "beta.md").write_text("# Beta\n\nbody b\n", encoding="utf-8")


def test_chunks_sha256_hashes_exact_file_bytes(tmp_path: Path):
    payload = b'{"chunk_id": "c1"}\n{"chunk_id": "c2"}\n'
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_bytes(payload)
    assert index_manifest.chunks_sha256(chunks_file) == hashlib.sha256(payload).hexdigest()


def test_corpus_sha256_is_stable_across_calls(tmp_path: Path):
    _write_mini_corpus(tmp_path)
    assert index_manifest.corpus_sha256(tmp_path) == index_manifest.corpus_sha256(tmp_path)


def test_corpus_sha256_changes_when_a_file_is_renamed(tmp_path: Path):
    _write_mini_corpus(tmp_path)
    before = index_manifest.corpus_sha256(tmp_path)
    (tmp_path / "public" / "alpha.md").rename(tmp_path / "public" / "alpha-renamed.md")
    after = index_manifest.corpus_sha256(tmp_path)
    assert before != after


def test_corpus_sha256_changes_when_content_changes(tmp_path: Path):
    _write_mini_corpus(tmp_path)
    before = index_manifest.corpus_sha256(tmp_path)
    (tmp_path / "public" / "alpha.md").write_text("# Alpha\n\nchanged\n", encoding="utf-8")
    assert before != index_manifest.corpus_sha256(tmp_path)


def test_corpus_sha256_ignores_env_pdf_and_non_md_files(tmp_path: Path):
    _write_mini_corpus(tmp_path)
    before = index_manifest.corpus_sha256(tmp_path)
    (tmp_path / ".env").write_text("GROQ_API_KEY=secret\n", encoding="utf-8")
    (tmp_path / "public" / "source.pdf").write_bytes(b"%PDF-1.7 copyrighted")
    (tmp_path / "public" / "notes.txt").write_text("scratch\n", encoding="utf-8")
    assert index_manifest.corpus_sha256(tmp_path) == before


def test_corpus_sha256_independent_of_mtime(tmp_path: Path):
    _write_mini_corpus(tmp_path)
    before = index_manifest.corpus_sha256(tmp_path)
    import os
    import time

    stamp = time.time() + 10_000
    os.utime(tmp_path / "public" / "alpha.md", (stamp, stamp))
    assert index_manifest.corpus_sha256(tmp_path) == before


def test_resolve_build_commit_prefers_explicit_argument(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEPLOYED_SHA", "env-sha")
    assert index_manifest.resolve_build_commit("explicit-sha") == "explicit-sha"


def test_resolve_build_commit_uses_deployed_sha_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEPLOYED_SHA", "env-sha")
    assert index_manifest.resolve_build_commit() == "env-sha"


def test_resolve_build_commit_falls_back_to_git_head(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DEPLOYED_SHA", raising=False)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=index_manifest.REPO_ROOT, text=True
    ).strip()
    assert index_manifest.resolve_build_commit() == head


def test_resolve_build_commit_returns_unknown_without_git(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DEPLOYED_SHA", raising=False)

    def _boom(*args: object, **kwargs: object) -> str:
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(index_manifest.subprocess, "check_output", _boom)
    assert index_manifest.resolve_build_commit() == "unknown"


def _manifest() -> index_manifest.IndexManifest:
    return index_manifest.IndexManifest(
        index_profile="raw-v1",
        chunks_sha256="a" * 64,
        corpus_sha256="b" * 64,
        embedding_model=MODEL_NAME,
        embedding_revision=MODEL_REVISION,
        build_commit="cafe" * 10,
        chunk_count=228,
    )


def test_manifest_write_then_read_roundtrips(tmp_path: Path):
    path = tmp_path / "index_manifest.json"
    manifest = _manifest()
    index_manifest.write(manifest, path)
    assert index_manifest.read(path) == manifest


def test_manifest_write_uses_lf_newlines(tmp_path: Path):
    path = tmp_path / "index_manifest.json"
    index_manifest.write(_manifest(), path)
    assert b"\r\n" not in path.read_bytes()


def test_manifest_write_holds_the_required_fields(tmp_path: Path):
    path = tmp_path / "index_manifest.json"
    index_manifest.write(_manifest(), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {
        "index_profile",
        "chunks_sha256",
        "corpus_sha256",
        "embedding_model",
        "embedding_revision",
        "build_commit",
        "chunk_count",
    }


def test_manifest_verify_passes_when_stored_hashes_match_live_inputs(tmp_path: Path):
    _write_mini_corpus(tmp_path)
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_bytes(b'{"chunk_id": "c1"}\n')
    path = tmp_path / "index_manifest.json"
    manifest = index_manifest.IndexManifest(
        index_profile="raw-v1",
        chunks_sha256=index_manifest.chunks_sha256(chunks_file),
        corpus_sha256=index_manifest.corpus_sha256(tmp_path),
        embedding_model=MODEL_NAME,
        embedding_revision=MODEL_REVISION,
        build_commit="deadbeef",
        chunk_count=1,
    )
    index_manifest.write(manifest, path)
    index_manifest.verify(path, chunks_path=chunks_file, corpus_dir=tmp_path)


def test_manifest_verify_raises_on_hash_mismatch(tmp_path: Path):
    _write_mini_corpus(tmp_path)
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_bytes(b'{"chunk_id": "c1"}\n')
    path = tmp_path / "index_manifest.json"
    manifest = index_manifest.IndexManifest(
        index_profile="raw-v1",
        chunks_sha256="0" * 64,
        corpus_sha256=index_manifest.corpus_sha256(tmp_path),
        embedding_model=MODEL_NAME,
        embedding_revision=MODEL_REVISION,
        build_commit="deadbeef",
        chunk_count=1,
    )
    index_manifest.write(manifest, path)
    with pytest.raises(ValueError):
        index_manifest.verify(path, chunks_path=chunks_file, corpus_dir=tmp_path)
