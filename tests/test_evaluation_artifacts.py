from __future__ import annotations

import dataclasses

import pytest

from src.adapters.secondary.embedder.sentence_transformers_embedder import MODEL_NAME, MODEL_REVISION
from src.core.config import _DEFAULT_REFUSAL_COSINE_THRESHOLD
from src.features.evaluation import artifacts
from src.features.evaluation.eval_set_integrity import load_eval_set
from src.features.evaluation.regression_set_integrity import load_regression_set

_CHUNKS_SHA = "a" * 64
_CORPUS_SHA = "b" * 64
_BUILD_COMMIT = "1111111111111111111111111111111111111111"
_EVAL_COMMIT = "2222222222222222222222222222222222222222"


def test_artifact_suffix_raw_off():
    assert artifacts.artifact_suffix("raw-v1", "off") == "__raw-v1__off"


def test_artifact_suffix_contextual_semantic():
    assert artifacts.artifact_suffix("contextual-v1", "semantic") == "__contextual-v1__semantic"


@pytest.mark.parametrize("expansion_mode", ["off", "semantic", "lexical", "both"])
def test_artifact_suffix_accepts_every_known_expansion_mode(expansion_mode: str):
    assert artifacts.artifact_suffix("raw-v1", expansion_mode) == f"__raw-v1__{expansion_mode}"


@pytest.mark.parametrize(
    ("index_profile", "expansion_mode"),
    [
        ("raw-v2", "off"),
        ("contextual", "off"),
        ("", "off"),
        ("raw-v1", "hybrid"),
        ("raw-v1", ""),
        ("raw-v1", "OFF"),
    ],
)
def test_artifact_suffix_rejects_unknown_values(index_profile: str, expansion_mode: str):
    with pytest.raises(ValueError):
        artifacts.artifact_suffix(index_profile, expansion_mode)


def _header() -> artifacts.ProvenanceHeader:
    return artifacts.provenance_header(
        index_profile="raw-v1",
        expansion_mode="off",
        chunks_sha256=_CHUNKS_SHA,
        corpus_sha256=_CORPUS_SHA,
        index_build_commit=_BUILD_COMMIT,
        evaluation_commit=_EVAL_COMMIT,
    )


def test_provenance_header_is_frozen_dataclass_with_render():
    header = _header()
    rendered = header.render()
    assert isinstance(rendered, str)
    with pytest.raises(dataclasses.FrozenInstanceError):
        header.index_profile = "contextual-v1"  # type: ignore[misc]


def test_provenance_header_is_deterministic():
    assert _header().render() == _header().render()


def test_provenance_header_contains_dataset_versions_and_stored_hashes():
    rendered = _header().render()
    eval_set = load_eval_set()
    regression_set = load_regression_set()
    assert eval_set["version"] in rendered
    assert eval_set["sha256"] in rendered
    assert regression_set["version"] in rendered
    assert regression_set["sha256"] in rendered


def test_provenance_header_contains_axes_and_threshold():
    rendered = _header().render()
    assert "raw-v1" in rendered
    assert "off" in rendered
    assert str(_DEFAULT_REFUSAL_COSINE_THRESHOLD) in rendered


def test_provenance_header_contains_hashes_and_embedding_identity():
    rendered = _header().render()
    assert _CHUNKS_SHA in rendered
    assert _CORPUS_SHA in rendered
    assert MODEL_NAME in rendered
    assert MODEL_REVISION in rendered


def test_provenance_header_reports_build_and_evaluation_commit_separately():
    header = _header()
    assert header.index_build_commit == _BUILD_COMMIT
    assert header.evaluation_commit == _EVAL_COMMIT
    rendered = header.render()
    assert _BUILD_COMMIT in rendered
    assert _EVAL_COMMIT in rendered
    build_line = next(line for line in rendered.splitlines() if _BUILD_COMMIT in line)
    eval_line = next(line for line in rendered.splitlines() if _EVAL_COMMIT in line)
    assert build_line != eval_line


def test_provenance_header_rejects_unknown_axes():
    with pytest.raises(ValueError):
        artifacts.provenance_header(
            index_profile="bogus",
            expansion_mode="off",
            chunks_sha256=_CHUNKS_SHA,
            corpus_sha256=_CORPUS_SHA,
            index_build_commit=_BUILD_COMMIT,
            evaluation_commit=_EVAL_COMMIT,
        )


def test_provenance_header_does_not_leak_secrets():
    rendered = _header().render().lower()
    for needle in ("api_key", "api-key", "secret", "password", "bearer", "groq_api", "openai_api"):
        assert needle not in rendered
