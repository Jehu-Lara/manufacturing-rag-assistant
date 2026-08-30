from __future__ import annotations

from dataclasses import dataclass

from src.adapters.secondary.embedder.sentence_transformers_embedder import MODEL_NAME, MODEL_REVISION
from src.core.config import _DEFAULT_REFUSAL_COSINE_THRESHOLD
from src.features.evaluation.eval_set_integrity import load_eval_set
from src.features.evaluation.regression_set_integrity import load_regression_set

INDEX_PROFILES: tuple[str, ...] = ("raw-v1", "contextual-v1")
EXPANSION_MODES: tuple[str, ...] = ("off", "semantic", "lexical", "both")


def artifact_suffix(index_profile: str, expansion_mode: str) -> str:
    if index_profile not in INDEX_PROFILES:
        raise ValueError(f"index_profile must be one of {INDEX_PROFILES}, got {index_profile!r}")
    if expansion_mode not in EXPANSION_MODES:
        raise ValueError(f"expansion_mode must be one of {EXPANSION_MODES}, got {expansion_mode!r}")
    return f"__{index_profile}__{expansion_mode}"


@dataclass(frozen=True)
class ProvenanceHeader:
    eval_set_version: str
    eval_set_sha256: str
    regression_set_version: str
    regression_set_sha256: str
    index_profile: str
    expansion_mode: str
    refusal_cosine_threshold: float
    chunks_sha256: str
    corpus_sha256: str
    embedding_model: str
    embedding_revision: str
    index_build_commit: str
    evaluation_commit: str

    def render(self) -> str:
        lines = [
            "## Provenance",
            f"- eval_set_version: {self.eval_set_version}",
            f"- eval_set_sha256: {self.eval_set_sha256}",
            f"- regression_set_version: {self.regression_set_version}",
            f"- regression_set_sha256: {self.regression_set_sha256}",
            f"- index_profile: {self.index_profile}",
            f"- expansion_mode: {self.expansion_mode}",
            f"- refusal_cosine_threshold: {self.refusal_cosine_threshold}",
            f"- chunks_sha256: {self.chunks_sha256}",
            f"- corpus_sha256: {self.corpus_sha256}",
            f"- embedding_model: {self.embedding_model}",
            f"- embedding_revision: {self.embedding_revision}",
            f"- index_build_commit: {self.index_build_commit}",
            f"- evaluation_commit: {self.evaluation_commit}",
        ]
        return "\n".join(lines)


def provenance_header(
    *,
    index_profile: str,
    expansion_mode: str,
    chunks_sha256: str,
    corpus_sha256: str,
    index_build_commit: str,
    evaluation_commit: str,
) -> ProvenanceHeader:
    artifact_suffix(index_profile, expansion_mode)

    eval_set = load_eval_set()
    regression_set = load_regression_set()

    return ProvenanceHeader(
        eval_set_version=str(eval_set["version"]),
        eval_set_sha256=str(eval_set["sha256"]),
        regression_set_version=str(regression_set["version"]),
        regression_set_sha256=str(regression_set["sha256"]),
        index_profile=index_profile,
        expansion_mode=expansion_mode,
        refusal_cosine_threshold=_DEFAULT_REFUSAL_COSINE_THRESHOLD,
        chunks_sha256=chunks_sha256,
        corpus_sha256=corpus_sha256,
        embedding_model=MODEL_NAME,
        embedding_revision=MODEL_REVISION,
        index_build_commit=index_build_commit,
        evaluation_commit=evaluation_commit,
    )
