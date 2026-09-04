from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from src.adapters.secondary.embedder.sentence_transformers_embedder import MODEL_NAME, MODEL_REVISION
from src.adapters.secondary.lexical.bm25_lexical_index import BM25_SCHEMA_VERSION, LEXICAL_PROFILE
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


def test_corpus_sha256_ignores_top_level_md_outside_embedded_subdirs(tmp_path: Path):
    _write_mini_corpus(tmp_path)
    (tmp_path / "SOURCES.md").write_text("| file | type |\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("scratch notes\n", encoding="utf-8")
    before = index_manifest.corpus_sha256(tmp_path)

    (tmp_path / "SOURCES.md").write_text(
        "| file | type |\n| public/alpha.md | public |\n", encoding="utf-8"
    )
    (tmp_path / "notes.md").write_text("edited scratch notes\n", encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("v2\n", encoding="utf-8")

    assert index_manifest.corpus_sha256(tmp_path) == before


def test_corpus_sha256_changes_when_a_public_md_is_edited(tmp_path: Path):
    _write_mini_corpus(tmp_path)
    (tmp_path / "SOURCES.md").write_text("| file | type |\n", encoding="utf-8")
    before = index_manifest.corpus_sha256(tmp_path)
    (tmp_path / "public" / "alpha.md").write_text("# Alpha\n\nedited body\n", encoding="utf-8")
    assert index_manifest.corpus_sha256(tmp_path) != before


def test_corpus_sha256_ordering_matches_ingestion_discovery(tmp_path: Path):
    _write_mini_corpus(tmp_path)
    (tmp_path / "public" / "gamma.md").write_text("# Gamma\n\nbody g\n", encoding="utf-8")
    (tmp_path / "synthetic" / "delta.md").write_text("# Delta\n\nbody d\n", encoding="utf-8")

    ingestion_order = sorted((tmp_path / "public").glob("*.md")) + sorted(
        (tmp_path / "synthetic").glob("*.md")
    )
    manifest_order = sorted(
        [*(tmp_path / "public").glob("*.md"), *(tmp_path / "synthetic").glob("*.md")],
        key=lambda p: p.relative_to(tmp_path).as_posix(),
    )
    assert manifest_order == ingestion_order


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


def test_resolve_build_commit_reads_deployed_sha_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.delenv("DEPLOYED_SHA", raising=False)
    monkeypatch.setattr(index_manifest, "REPO_ROOT", tmp_path)
    sha = "a1b2c3d4" * 5
    (tmp_path / "DEPLOYED_SHA").write_text(sha + "\n", encoding="utf-8")
    assert index_manifest.resolve_build_commit() == sha


def test_resolve_build_commit_deployed_sha_env_beats_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("DEPLOYED_SHA", "env-sha")
    monkeypatch.setattr(index_manifest, "REPO_ROOT", tmp_path)
    (tmp_path / "DEPLOYED_SHA").write_text("f" * 40 + "\n", encoding="utf-8")
    assert index_manifest.resolve_build_commit() == "env-sha"


def test_resolve_build_commit_ignores_malformed_deployed_sha_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.delenv("DEPLOYED_SHA", raising=False)
    monkeypatch.setattr(index_manifest, "REPO_ROOT", tmp_path)
    (tmp_path / "DEPLOYED_SHA").write_text("not-a-real-sha\n", encoding="utf-8")
    # tmp_path is not a git repo, so the git fallback yields "unknown" — the
    # point is only that the malformed file value is never returned.
    assert index_manifest.resolve_build_commit() == "unknown"


def test_resolve_index_profile_defaults_to_contextual_v1(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("INDEX_PROFILE", raising=False)
    assert index_manifest.resolve_index_profile() == "contextual-v1"


def test_resolve_index_profile_reads_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INDEX_PROFILE", "raw-v1")
    assert index_manifest.resolve_index_profile() == "raw-v1"


def test_resolve_index_profile_rejects_unknown(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INDEX_PROFILE", "bogus-v9")
    with pytest.raises(ValueError, match="INDEX_PROFILE"):
        index_manifest.resolve_index_profile()


def _manifest() -> index_manifest.IndexManifest:
    return index_manifest.IndexManifest(
        index_profile="raw-v1",
        chunks_sha256="a" * 64,
        corpus_sha256="b" * 64,
        embedding_model=MODEL_NAME,
        embedding_revision=MODEL_REVISION,
        build_commit="cafe" * 10,
        chunk_count=228,
        lexical_profile=LEXICAL_PROFILE,
        bm25_schema_version=BM25_SCHEMA_VERSION,
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
        "lexical_profile",
        "bm25_schema_version",
    }


def test_manifest_read_rejects_unknown_index_profile(tmp_path: Path):
    path = tmp_path / "index_manifest.json"
    index_manifest.write(_manifest(), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["index_profile"] = "bogus-v9"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="index_profile"):
        index_manifest.read(path)


def test_manifest_read_rejects_non_int_chunk_count(tmp_path: Path):
    path = tmp_path / "index_manifest.json"
    index_manifest.write(_manifest(), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["chunk_count"] = "228"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="chunk_count"):
        index_manifest.read(path)


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
        lexical_profile=LEXICAL_PROFILE,
        bm25_schema_version=BM25_SCHEMA_VERSION,
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
        lexical_profile=LEXICAL_PROFILE,
        bm25_schema_version=BM25_SCHEMA_VERSION,
    )
    index_manifest.write(manifest, path)
    with pytest.raises(ValueError):
        index_manifest.verify(path, chunks_path=chunks_file, corpus_dir=tmp_path)


def _coherent_manifest_on_disk(
    tmp_path: Path, *, index_profile: str = "raw-v1", chunk_count: int = 2
) -> tuple[Path, Path]:
    _write_mini_corpus(tmp_path)
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text(
        "".join(f'{{"chunk_id": "c{i}"}}\n' for i in range(chunk_count)), encoding="utf-8"
    )
    path = tmp_path / "index_manifest.json"
    index_manifest.write(
        index_manifest.IndexManifest(
            index_profile=index_profile,
            chunks_sha256=index_manifest.chunks_sha256(chunks_file),
            corpus_sha256=index_manifest.corpus_sha256(tmp_path),
            embedding_model=MODEL_NAME,
            embedding_revision=MODEL_REVISION,
            build_commit="deadbeef",
            chunk_count=chunk_count,
            lexical_profile=LEXICAL_PROFILE,
            bm25_schema_version=BM25_SCHEMA_VERSION,
        ),
        path,
    )
    return path, chunks_file


def test_manifest_verify_returns_the_manifest(tmp_path: Path):
    path, chunks_file = _coherent_manifest_on_disk(tmp_path)
    manifest = index_manifest.verify(path, chunks_path=chunks_file, corpus_dir=tmp_path)
    assert isinstance(manifest, index_manifest.IndexManifest)
    assert manifest.index_profile == "raw-v1"


def test_manifest_verify_rejects_profile_mismatch(tmp_path: Path):
    path, chunks_file = _coherent_manifest_on_disk(tmp_path, index_profile="contextual-v1")
    with pytest.raises(ValueError, match="index_profile"):
        index_manifest.verify(
            path, expected_profile="raw-v1", chunks_path=chunks_file, corpus_dir=tmp_path
        )


def test_manifest_verify_accepts_matching_expected_profile(tmp_path: Path):
    path, chunks_file = _coherent_manifest_on_disk(tmp_path, index_profile="contextual-v1")
    index_manifest.verify(
        path, expected_profile="contextual-v1", chunks_path=chunks_file, corpus_dir=tmp_path
    )


def test_manifest_verify_rejects_embedding_model_mismatch(tmp_path: Path):
    path, chunks_file = _coherent_manifest_on_disk(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["embedding_model"] = "some/other-model"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="embedding_model"):
        index_manifest.verify(path, chunks_path=chunks_file, corpus_dir=tmp_path)


def test_manifest_verify_rejects_embedding_revision_mismatch(tmp_path: Path):
    path, chunks_file = _coherent_manifest_on_disk(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["embedding_revision"] = "0" * 40
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="embedding_revision"):
        index_manifest.verify(path, chunks_path=chunks_file, corpus_dir=tmp_path)


def test_manifest_verify_rejects_chunk_count_mismatch(tmp_path: Path):
    path, chunks_file = _coherent_manifest_on_disk(tmp_path, chunk_count=2)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["chunk_count"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="chunk_count"):
        index_manifest.verify(path, chunks_path=chunks_file, corpus_dir=tmp_path)


# --- lexical fields (bucket 3) ---


def test_manifest_records_the_lexical_profile_and_schema_version(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"chunk_id": "d::chunk-0000"}\n', encoding="utf-8")
    corpus = tmp_path / "corpus"
    (corpus / "public").mkdir(parents=True)
    (corpus / "synthetic").mkdir(parents=True)
    (corpus / "public" / "a.md").write_text("# A\n", encoding="utf-8")

    manifest = index_manifest.build_manifest(
        "contextual-v1", 1, build_commit="c" * 40, chunks_path=chunks, corpus_dir=corpus
    )

    assert manifest.lexical_profile == LEXICAL_PROFILE
    assert manifest.bm25_schema_version == BM25_SCHEMA_VERSION


def test_read_rejects_a_manifest_missing_the_lexical_fields(tmp_path):
    """_MANIFEST_FIELDS drives the missing-field check, so a pre-bucket-3
    manifest is unreadable rather than silently assumed compatible."""
    path = tmp_path / "index_manifest.json"
    path.write_text(
        json.dumps(
            {
                "index_profile": "contextual-v1",
                "chunks_sha256": "a" * 64,
                "corpus_sha256": "b" * 64,
                "embedding_model": "m",
                "embedding_revision": "r",
                "build_commit": "c" * 40,
                "chunk_count": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lexical_profile"):
        index_manifest.read(path)
