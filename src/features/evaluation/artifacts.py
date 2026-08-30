from __future__ import annotations

from dataclasses import dataclass

from src.adapters.secondary.embedder.sentence_transformers_embedder import MODEL_NAME, MODEL_REVISION
from src.core.config import DEFAULT_REFUSAL_COSINE_THRESHOLD
from src.features.evaluation.eval_set_integrity import load_eval_set
from src.features.evaluation.regression_set_integrity import load_regression_set
from src.features.retrieval import index_manifest

INDEX_PROFILES: tuple[str, ...] = ("raw-v1", "contextual-v1")
EXPANSION_MODES: tuple[str, ...] = ("off", "semantic", "lexical", "both")

CANONICAL_ALIAS_PROFILE = "raw-v1"
CANONICAL_ALIAS_MODE = "off"


def artifact_suffix(index_profile: str, expansion_mode: str) -> str:
    if index_profile not in INDEX_PROFILES:
        raise ValueError(f"index_profile must be one of {INDEX_PROFILES}, got {index_profile!r}")
    if expansion_mode not in EXPANSION_MODES:
        raise ValueError(f"expansion_mode must be one of {EXPANSION_MODES}, got {expansion_mode!r}")
    return f"__{index_profile}__{expansion_mode}"


def artifact_filename(stem: str, version: str, index_profile: str, expansion_mode: str, ext: str) -> str:
    """`<stem>_v<dataset-version><__profile__mode>.<ext>` — the dataset version
    never moves; index_profile and expansion_mode are suffix axes, not version
    bumps."""
    return f"{stem}_v{version}{artifact_suffix(index_profile, expansion_mode)}.{ext}"


def ensure_canonical_alias_allowed(index_profile: str, expansion_mode: str) -> None:
    """The unsuffixed canonical name (`retrieval_report_v1.1.0.md` etc.) is the
    frozen baseline; only a raw-v1/off run may write it."""
    if (index_profile, expansion_mode) != (CANONICAL_ALIAS_PROFILE, CANONICAL_ALIAS_MODE):
        raise ValueError(
            "write_canonical_alias=True is only legal for the "
            f"{CANONICAL_ALIAS_PROFILE}/{CANONICAL_ALIAS_MODE} baseline; got "
            f"index_profile={index_profile!r}, expansion_mode={expansion_mode!r}"
        )


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
        refusal_cosine_threshold=DEFAULT_REFUSAL_COSINE_THRESHOLD,
        chunks_sha256=chunks_sha256,
        corpus_sha256=corpus_sha256,
        embedding_model=MODEL_NAME,
        embedding_revision=MODEL_REVISION,
        index_build_commit=index_build_commit,
        evaluation_commit=evaluation_commit,
    )


def resolve_provenance(index_profile: str, expansion_mode: str) -> ProvenanceHeader:
    """Build a provenance header for a runner about to write a report: hashes
    from the on-disk chunk/corpus inputs, index-build commit from the manifest
    when present (else the current checkout), evaluation commit from the
    current checkout."""
    try:
        index_build_commit = index_manifest.read().build_commit
    except (FileNotFoundError, ValueError):
        index_build_commit = index_manifest.resolve_build_commit()
    return provenance_header(
        index_profile=index_profile,
        expansion_mode=expansion_mode,
        chunks_sha256=index_manifest.chunks_sha256(),
        corpus_sha256=index_manifest.corpus_sha256(),
        index_build_commit=index_build_commit,
        evaluation_commit=index_manifest.resolve_build_commit(),
    )
