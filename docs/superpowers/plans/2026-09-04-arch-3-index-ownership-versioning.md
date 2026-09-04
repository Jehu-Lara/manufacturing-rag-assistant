# Bucket 3 — Index Ownership, Embedding-Text Policy and BM25 Versioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the contextual embedding-text policy out of the Chroma adapter and into the domain; version the BM25 JSON with `schema_version`, `lexical_profile` and a chunks hash; make startup validate manifest **plus profile plus content**; and give the build CLI explicit end-to-end ownership — candidates, per-artifact atomic promotion, manifest written last as the commit marker, fail-closed on any error.

**Architecture:** Four ordered tasks. T1 makes embedding-input construction a domain policy and turns `build_collection` into a dumb writer that is told what to embed. T2 versions the BM25 payload and teaches it to fail closed on an unknown or mismatched one. T3 extends the startup and eval-runner coherence check to cover the new BM25 fields. T4 restructures `cli.run()` so the manifest is the last thing written and no partial build can be promoted. T1 → T4 and T2 → T3 → T4.

**Tech Stack:** Python 3.11, chromadb, rank-bm25, hashlib/json (stdlib), pytest.

**Spec:** `docs/superpowers/specs/2026-09-04-architecture-remediation-design.md`

## Global Constraints

- Execution is gated: PR #9 resolved by the owner, then a new branch cut from `master` with the owner's authorization.
- **This bucket requires a reindex to land.** T2 changes the on-disk BM25 payload shape, so the existing `retrieval/output/bm25_index.json` becomes unreadable by design and every app start, eval runner and the app-startup test suite will fail until `python -m src.features.retrieval.cli` is re-run. That reindex is itself an owner-authorized action — schedule it as the last step of the bucket, not mid-way.
- **No corpus content change.** `chunk_id` stays positional and the frozen eval sets stay anchored. Nothing here touches `corpus/`, so no re-anchoring procedure is triggered.
- Byte-stable invariants unchanged (`0.5999`, `0.5500`, RRF `k=60`, `binary`, `off`, `contextual-v1`). The contextual prefix format — `f"{document_title} > {section_heading}\n\n{chunk_text}"` with an ASCII `>` — is itself byte-stable: it defines what `contextual-v1` means, and changing it would silently invalidate the live index and every `__contextual-v1__` report.
- `documents=` and `metadatas=` in Chroma keep storing the **raw** `chunk_text` for both profiles. Only the embedding vectors differ. That is ADR-008 and it does not change here.
- After each task: `pytest tests/test_chroma_vector_store.py tests/test_bm25_lexical_index.py tests/test_retrieval_cli.py -q` green. End of bucket: `pytest -q`, `ruff check src tests`, `mypy src` green, then the reindex, then `python -m src.features.evaluation.retrieval_eval` with `index_profile="contextual-v1"` to confirm the rebuilt index still measures the same.

---

## File Structure

- `src/domain/policies.py` — gains `embedding_input(chunk, profile)` and `embedding_inputs(chunks, profile)`. Framework-free, so the policy is testable without chromadb and reusable by any future vector adapter.
- `src/adapters/secondary/vector/chroma_vector_store.py` — `build_collection(chunks, embedding_inputs)` becomes a writer that is told what to embed. `contextual_embedding_input` is re-exported from here for one release so `tests/test_retrieval_cli.py:9` and any external reader keep resolving.
- `src/domain/ports.py` — `VectorStorePort.build_collection(chunks, embedding_inputs)`.
- `src/adapters/secondary/lexical/bm25_lexical_index.py` — versioned payload, `build_index(chunks, *, chunks_sha256)`, `validate(expected_chunk_ids, *, expected_chunks_sha256, expected_lexical_profile)`, candidate/promote split.
- `src/features/retrieval/index_manifest.py` — `IndexManifest` gains `lexical_profile` and `bm25_schema_version`; `_MANIFEST_FIELDS` and `verify()` follow.
- `src/features/retrieval/cli.py` — `run()` restructured into build-candidates → promote-each → write-manifest-last, with fail-closed cleanup.
- `src/adapters/primary/http/app.py`, `src/features/evaluation/_eval_retriever.py` — pass the new expectations into `lexical_index.validate(...)`.
- Tests: `tests/test_domain_policies.py`, `tests/test_chroma_vector_store.py`, `tests/test_bm25_lexical_index.py`, `tests/test_retrieval_cli.py`, `tests/test_retrieval_index_manifest.py`, `tests/test_http_app_startup.py`, `tests/fakes.py`.

---

### Task 1: Embedding-text policy moves into the domain

**Files:**
- Modify: `src/domain/policies.py`, `src/domain/ports.py:19`, `src/adapters/secondary/vector/chroma_vector_store.py:14-16,53-64`, `src/features/retrieval/cli.py:37`, `tests/fakes.py:78`
- Test: `tests/test_domain_policies.py`, `tests/test_chroma_vector_store.py`, `tests/test_retrieval_cli.py`

**Interfaces:**
- Produces:
  - `src.domain.policies.embedding_input(chunk: ChunkMetadata, profile: IndexProfile) -> str`
  - `src.domain.policies.embedding_inputs(chunks: Sequence[ChunkMetadata], profile: IndexProfile) -> list[str]`
  - `VectorStorePort.build_collection(self, chunks: list[ChunkMetadata], embedding_inputs: list[str]) -> None`
- Consumes: `src.domain.models.{ChunkMetadata, IndexProfile}` — both already live in the domain, so no new dependency is introduced and `tests/test_import_invariants.py`'s domain rule still holds.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_domain_policies.py`:

```python
def test_contextual_embedding_input_is_the_byte_stable_prefix() -> None:
    """This string IS the definition of contextual-v1 (ADR-008). An ASCII '>',
    one space either side, a blank line, then the raw chunk text. Changing it
    silently invalidates the live index and every __contextual-v1__ report."""
    chunk = _chunk(document_title="Doc A", section_heading="4.2 Cleaning", chunk_text="Body text.")

    assert embedding_input(chunk, "contextual-v1") == "Doc A > 4.2 Cleaning\n\nBody text."


def test_raw_profile_embeds_the_bare_body() -> None:
    chunk = _chunk(document_title="Doc A", section_heading="4.2 Cleaning", chunk_text="Body text.")

    assert embedding_input(chunk, "raw-v1") == "Body text."


def test_embedding_inputs_preserves_order() -> None:
    chunks = [_chunk(chunk_text="one"), _chunk(chunk_text="two"), _chunk(chunk_text="three")]

    assert embedding_inputs(chunks, "raw-v1") == ["one", "two", "three"]


def test_unknown_profile_raises_rather_than_defaulting() -> None:
    """No silent fallback: an unrecognised profile must not quietly embed the
    raw body and produce an index that claims to be something else."""
    with pytest.raises(ValueError, match="index profile"):
        embedding_input(_chunk(), "contextual-v2")  # type: ignore[arg-type]
```

Reuse whatever `_chunk(...)` factory `tests/test_domain_policies.py` already has; if it has none, add one built from `ChunkMetadata.__dataclass_fields__` the way Bucket 1's `_chunk_row` is.

And append to `tests/test_chroma_vector_store.py`:

```python
def test_build_collection_embeds_exactly_what_it_is_given() -> None:
    """The adapter is a writer, not a policy holder: it embeds the strings the
    caller computed and stores the raw chunk_text regardless (ADR-008)."""
    embedder = RecordingEmbedder()
    store = _store(tmp_path, embedder, profile="contextual-v1")

    store.build_collection(FIXTURE_CHUNKS, ["FIRST", "SECOND"][: len(FIXTURE_CHUNKS)])

    assert embedder.embed_texts_calls[-1] == ["FIRST", "SECOND"][: len(FIXTURE_CHUNKS)]


def test_build_collection_rejects_a_length_mismatch(tmp_path) -> None:
    store = _store(tmp_path, RecordingEmbedder(), profile="raw-v1")

    with pytest.raises(ValueError, match="embedding_inputs"):
        store.build_collection(FIXTURE_CHUNKS, ["only-one"])
```

Add `tmp_path` to the first test's signature to match the module's existing fixture style.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_domain_policies.py tests/test_chroma_vector_store.py -q -k "embedding_input or embeds_exactly or length_mismatch"`

Expected: FAIL — `NameError: name 'embedding_input' is not defined` and `build_collection() takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Add the policy**

In `src/domain/policies.py`:

```python
def embedding_input(chunk: ChunkMetadata, profile: IndexProfile) -> str:
    """The one place an index profile's embedding text is defined. contextual-v1
    prefixes the heading breadcrumb with an ASCII '>' (ADR-008); raw-v1 embeds
    the bare body and is the tested rollback path. Both store the raw chunk_text
    in the vector store — only the vectors differ."""
    if profile == "contextual-v1":
        return f"{chunk.document_title} > {chunk.section_heading}\n\n{chunk.chunk_text}"
    if profile == "raw-v1":
        return chunk.chunk_text
    raise ValueError(f"unknown index profile: {profile!r}")


def embedding_inputs(chunks: Sequence[ChunkMetadata], profile: IndexProfile) -> list[str]:
    return [embedding_input(chunk, profile) for chunk in chunks]
```

Import `ChunkMetadata` and `IndexProfile` from `src.domain.models` and `Sequence` from `typing` alongside the module's existing imports.

- [ ] **Step 4: Make the adapter a writer**

`src/domain/ports.py:19`:

```python
    def build_collection(self, chunks: list[ChunkMetadata], embedding_inputs: list[str]) -> None: ...
```

`src/adapters/secondary/vector/chroma_vector_store.py` — replace lines 53–64:

```python
    def build_collection(self, chunks: list[ChunkMetadata], embedding_inputs: list[str]) -> None:
        if len(embedding_inputs) != len(chunks):
            raise ValueError(
                f"embedding_inputs has {len(embedding_inputs)} entries, expected {len(chunks)}"
            )
        client = self._client()

        # Validate + embed BEFORE any collection is created/deleted/renamed, so
        # a length or model failure leaves the live collection untouched.
        self._embedder.assert_fits_max_seq_length(embedding_inputs)
        embeddings = self._embedder.embed_texts(embedding_inputs)
        ...  # candidate creation, add(), count check and _promote — unchanged
```

Keep `contextual_embedding_input` in this module as a thin deprecated shim so `tests/test_retrieval_cli.py:9` keeps importing it:

```python
def contextual_embedding_input(chunk: ChunkMetadata) -> str:
    """Deprecated shim — the policy moved to src.domain.policies.embedding_input.
    Kept for one release because tests and any external reader import it here."""
    return embedding_input(chunk, "contextual-v1")
```

`src/features/retrieval/cli.py:37` becomes:

```python
    vector_store.build_collection(chunks, embedding_inputs(chunks, profile))
```

`tests/fakes.py:78` and the three inline `build_collection` stubs (`tests/test_core_telemetry.py:31`, `tests/test_hybrid_retriever_use_case.py:16`, `tests/test_retrieval_cli.py:77`) each grow the second parameter.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_domain_policies.py tests/test_chroma_vector_store.py tests/test_retrieval_cli.py tests/test_import_invariants.py -q`

Expected: PASS. The import-invariants run matters: `embedding_input` must not have dragged anything framework-shaped into `src/domain/`.

---

### Task 2: Version the BM25 payload

**Files:**
- Modify: `src/adapters/secondary/lexical/bm25_lexical_index.py`
- Test: `tests/test_bm25_lexical_index.py`

**Interfaces:**
- Produces:
  - `BM25_SCHEMA_VERSION = 1`, `LEXICAL_PROFILE = "word-lower-v1"` (module constants)
  - `Bm25LexicalIndex.build_index(self, chunks: list[ChunkMetadata], *, chunks_sha256: str, target_path: Path | None = None) -> Path` — writes to `target_path` (default: a `.candidate` sibling of the persist path) and returns the path written, **without** promoting it
  - `Bm25LexicalIndex.promote(self, candidate_path: Path) -> None` — atomic `replace` onto the persist path
  - `Bm25LexicalIndex.validate(self, expected_chunk_ids: list[str], *, expected_chunks_sha256: str | None = None, expected_lexical_profile: str = LEXICAL_PROFILE) -> None`
- On-disk payload: `{"schema_version": 1, "lexical_profile": "word-lower-v1", "chunks_sha256": "<hex>", "chunk_ids": [...], "corpus_tokens": [[...], ...]}`.

`LEXICAL_PROFILE` names what `tokenize` actually does today — `re.findall(r"\w+", text.lower())`. It exists so Bucket 5's Snowball experiment can ship a `"snowball-bilingual-v1"` index that a `word-lower-v1` runtime refuses to load, instead of silently scoring a differently-tokenized corpus.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_bm25_lexical_index.py`:

```python
def test_persisted_payload_is_versioned(tmp_path) -> None:
    index = Bm25LexicalIndex(persist_path=tmp_path / "bm25.json")
    written = index.build_index(FIXTURE_CHUNKS, chunks_sha256="deadbeef")
    index.promote(written)

    payload = json.loads((tmp_path / "bm25.json").read_text(encoding="utf-8"))

    assert payload["schema_version"] == BM25_SCHEMA_VERSION
    assert payload["lexical_profile"] == LEXICAL_PROFILE
    assert payload["chunks_sha256"] == "deadbeef"
    assert payload["chunk_ids"] == [c.chunk_id for c in FIXTURE_CHUNKS]


def test_unversioned_legacy_payload_fails_closed(tmp_path) -> None:
    """A pre-versioning index on disk is not silently readable: it was built by
    an unknown tokenizer against an unknown chunk set."""
    path = tmp_path / "bm25.json"
    path.write_text(json.dumps({"chunk_ids": ["a"], "corpus_tokens": [["a"]]}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        Bm25LexicalIndex(persist_path=path).query("a", 1)


def test_future_schema_version_fails_closed(tmp_path) -> None:
    path = tmp_path / "bm25.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": BM25_SCHEMA_VERSION + 1,
                "lexical_profile": LEXICAL_PROFILE,
                "chunks_sha256": "x",
                "chunk_ids": ["a"],
                "corpus_tokens": [["a"]],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version"):
        Bm25LexicalIndex(persist_path=path).query("a", 1)


def test_validate_rejects_a_chunks_hash_mismatch(tmp_path) -> None:
    index = Bm25LexicalIndex(persist_path=tmp_path / "bm25.json")
    index.promote(index.build_index(FIXTURE_CHUNKS, chunks_sha256="aaaa"))

    with pytest.raises(RuntimeError, match="chunks_sha256"):
        index.validate([c.chunk_id for c in FIXTURE_CHUNKS], expected_chunks_sha256="bbbb")


def test_validate_rejects_a_lexical_profile_mismatch(tmp_path) -> None:
    index = Bm25LexicalIndex(persist_path=tmp_path / "bm25.json")
    index.promote(index.build_index(FIXTURE_CHUNKS, chunks_sha256="aaaa"))

    with pytest.raises(RuntimeError, match="lexical_profile"):
        index.validate(
            [c.chunk_id for c in FIXTURE_CHUNKS],
            expected_chunks_sha256="aaaa",
            expected_lexical_profile="snowball-bilingual-v1",
        )


def test_build_index_does_not_promote_by_itself(tmp_path) -> None:
    """The CLI owns promotion: a built candidate must not become live until
    every artifact in the build has succeeded."""
    path = tmp_path / "bm25.json"
    index = Bm25LexicalIndex(persist_path=path)

    candidate = index.build_index(FIXTURE_CHUNKS, chunks_sha256="aaaa")

    assert candidate.exists()
    assert not path.exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_bm25_lexical_index.py -q`

Expected: FAIL — `ImportError` on `BM25_SCHEMA_VERSION`/`LEXICAL_PROFILE`, `build_index()` rejecting `chunks_sha256`, and the legacy payload loading happily.

- [ ] **Step 3: Implement the versioned payload**

```python
BM25_SCHEMA_VERSION = 1
LEXICAL_PROFILE = "word-lower-v1"  # re.findall(r"\w+", text.lower()) — see tokenize()


    def build_index(
        self,
        chunks: list[ChunkMetadata],
        *,
        chunks_sha256: str,
        target_path: Optional[Path] = None,
    ) -> Path:
        """Writes a CANDIDATE and returns its path; it does not go live. The
        build CLI promotes it only once every artifact in the build succeeded,
        so a half-built index can never be served."""
        payload = {
            "schema_version": BM25_SCHEMA_VERSION,
            "lexical_profile": LEXICAL_PROFILE,
            "chunks_sha256": chunks_sha256,
            "chunk_ids": [chunk.chunk_id for chunk in chunks],
            "corpus_tokens": [tokenize(chunk.chunk_text) for chunk in chunks],
        }
        destination = target_path or self._candidate_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = destination.with_suffix(destination.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(destination)
        self._invalidate()
        return destination

    def _candidate_path(self) -> Path:
        return self._persist_path.with_name(self._persist_path.name + ".candidate")

    def promote(self, candidate_path: Path) -> None:
        candidate_path.replace(self._persist_path)
        self._invalidate()

    def _invalidate(self) -> None:
        self._chunk_ids = None
        self._bm25 = None
        self._meta = None
```

`_load` gains the version and shape checks and caches the metadata:

```python
    def _load(self) -> tuple[list[str], BM25Okapi]:
        if self._chunk_ids is not None and self._bm25 is not None:
            return self._chunk_ids, self._bm25
        if not self._persist_path.exists():
            raise FileNotFoundError(
                f"{self._persist_path} not found — run the retrieval index-build CLI first"
            )
        data = json.loads(self._persist_path.read_text(encoding="utf-8"))
        version = data.get("schema_version")
        if version != BM25_SCHEMA_VERSION:
            raise ValueError(
                f"{self._persist_path} has schema_version {version!r}, expected "
                f"{BM25_SCHEMA_VERSION} — rebuild with `python -m src.features.retrieval.cli`"
            )
        self._meta = {
            "lexical_profile": data.get("lexical_profile"),
            "chunks_sha256": data.get("chunks_sha256"),
        }
        self._chunk_ids = data["chunk_ids"]
        self._bm25 = BM25Okapi(data["corpus_tokens"])
        return self._chunk_ids, self._bm25
```

and `validate` grows the two new checks alongside the existing chunk-id comparison, each raising `RuntimeError` with the mismatching field name in the message (the tests match on `"chunks_sha256"` and `"lexical_profile"`).

Add `self._meta: Optional[dict[str, Any]] = None` to `__init__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_bm25_lexical_index.py -q`

Expected: PASS. `tests/test_http_app_startup.py` and `tests/test_retrieval_cli*.py` are expected to be RED here — Tasks 3 and 4 fix them.

- [ ] **Step 5: Commit-point check (no commit)**

Run: `mypy src/adapters/secondary/lexical/bm25_lexical_index.py`

Expected: green. `--strict` will demand the `Optional[dict[str, Any]]` annotation on `_meta` and a non-`Any` return on `_candidate_path`.

---

### Task 3: Startup validates manifest **and** profile **and** content

**Files:**
- Modify: `src/features/retrieval/index_manifest.py:24-32,88-115,127-137,144-180`, `src/adapters/primary/http/app.py:51`, `src/features/evaluation/_eval_retriever.py:33-39`
- Test: `tests/test_retrieval_index_manifest.py`, `tests/test_http_app_startup.py`

**Interfaces:**
- Produces: `IndexManifest` gains `lexical_profile: str` and `bm25_schema_version: int`; `_MANIFEST_FIELDS` gains both; `build_manifest(..., lexical_profile: str = LEXICAL_PROFILE, bm25_schema_version: int = BM25_SCHEMA_VERSION)`; `read()` rejects a manifest missing either; `verify()` reports them like the other mismatches.
- Consumes: `src.adapters.secondary.lexical.bm25_lexical_index.{BM25_SCHEMA_VERSION, LEXICAL_PROFILE}` — a `features → adapters.secondary` import, which is allowed (Bucket 1's invariant forbids only `fastapi` and `adapters.primary` here), and mirrors this module's existing `MODEL_NAME`/`MODEL_REVISION` import from the embedder adapter.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_retrieval_index_manifest.py`:

```python
def _fixture_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """A minimal chunks.jsonl + corpus dir, so build_manifest/verify can hash
    real bytes without touching the live retrieval/output/."""
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"chunk_id": "d::chunk-0000"}\n', encoding="utf-8")
    corpus = tmp_path / "corpus"
    (corpus / "public").mkdir(parents=True)
    (corpus / "synthetic").mkdir(parents=True)
    (corpus / "public" / "a.md").write_text("# A\n", encoding="utf-8")
    return chunks, corpus


def test_manifest_records_the_lexical_profile_and_schema_version(tmp_path: Path) -> None:
    chunks, corpus = _fixture_inputs(tmp_path)

    manifest = index_manifest.build_manifest(
        "contextual-v1", 1, build_commit="c" * 40, chunks_path=chunks, corpus_dir=corpus
    )

    assert manifest.lexical_profile == LEXICAL_PROFILE
    assert manifest.bm25_schema_version == BM25_SCHEMA_VERSION


def test_read_rejects_a_manifest_missing_the_new_fields(tmp_path: Path) -> None:
    """_MANIFEST_FIELDS drives the missing-field check, so adding the two names
    there is what makes this pass — no new branch needed in read()."""
    path = tmp_path / "index_manifest.json"
    path.write_text(
        json.dumps(
            {
                "index_profile": "contextual-v1",
                "chunks_sha256": "a" * 64,
                "corpus_sha256": "b" * 64,
                "embedding_model": "BAAI/bge-m3",
                "embedding_revision": "rev",
                "build_commit": "c" * 40,
                "chunk_count": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lexical_profile"):
        index_manifest.read(path)


def test_verify_reports_a_lexical_profile_mismatch(tmp_path: Path) -> None:
    chunks, corpus = _fixture_inputs(tmp_path)
    path = tmp_path / "index_manifest.json"
    index_manifest.write(
        index_manifest.build_manifest(
            "contextual-v1", 1, build_commit="c" * 40, chunks_path=chunks, corpus_dir=corpus
        ),
        path,
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["lexical_profile"] = "snowball-bilingual-v1"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="lexical_profile"):
        index_manifest.verify(path, chunks_path=chunks, corpus_dir=corpus)
```

`build_manifest` reads `MODEL_NAME`/`MODEL_REVISION` from the embedder module, so the `embedding_model` literal above must match what that module exports — check with `python -c "from src.adapters.secondary.embedder.sentence_transformers_embedder import MODEL_NAME, MODEL_REVISION; print(MODEL_NAME, MODEL_REVISION)"` and use the real values. The `read()` test does not care (it fails on the missing field first), but keeping them honest costs nothing.

And append to `tests/test_http_app_startup.py`:

```python
def test_startup_rejects_a_bm25_index_built_from_different_chunks(...) -> None:
    """The three artifacts must agree on content, not just on count. A BM25
    index whose chunks_sha256 differs from the manifest's is scoring a
    different corpus than the vector channel."""
```

Model it on whichever existing test in that module already asserts the profile/count startup failures, corrupting `chunks_sha256` in the BM25 payload instead.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_retrieval_index_manifest.py tests/test_http_app_startup.py -q`

Expected: FAIL — `IndexManifest` has no `lexical_profile`, and startup accepts a BM25 index built from other chunks.

- [ ] **Step 3: Extend the manifest**

Add both fields to the `IndexManifest` dataclass and to `_MANIFEST_FIELDS`, default them in `build_manifest` from the BM25 module's constants, and add to `verify()`:

```python
    if manifest.lexical_profile != LEXICAL_PROFILE:
        mismatches.append(
            f"lexical_profile stored {manifest.lexical_profile}, expected {LEXICAL_PROFILE}"
        )
    if manifest.bm25_schema_version != BM25_SCHEMA_VERSION:
        mismatches.append(
            f"bm25_schema_version stored {manifest.bm25_schema_version}, "
            f"expected {BM25_SCHEMA_VERSION}"
        )
```

`read()` needs no new code for the missing-field case: `_MANIFEST_FIELDS` already drives the `missing` check, and adding the two names there is what makes `test_read_rejects_a_manifest_missing_the_new_fields` pass. Add a type check for `bm25_schema_version` mirroring the existing `chunk_count` one.

- [ ] **Step 4: Pass the expectations at both startup paths**

`src/adapters/primary/http/app.py:51`:

```python
    lexical_index.validate(
        [chunk.chunk_id for chunk in chunks],
        expected_chunks_sha256=manifest.chunks_sha256,
        expected_lexical_profile=manifest.lexical_profile,
    )
```

`src/features/evaluation/_eval_retriever.py:39` gets the identical call, using the `manifest` it already reads on line 34. This keeps the promise in CLAUDE.md that `build_retriever()` runs "the same manifest + Chroma-collection + BM25 cross-check as `app.lifespan`" — after this task the two calls are literally identical, which is worth a comment at both sites.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_retrieval_index_manifest.py tests/test_http_app_startup.py -q`

Expected: PASS **only after Task 4's reindex**, because the live `retrieval/output/` still holds a pre-versioning BM25 index. If the suite builds its own fixture index (check for a `tmp_path`-scoped builder), it passes now; if it reads `retrieval/output/`, expect RED until the reindex and note it rather than working around it.

---

### Task 4: The CLI owns the build — candidates, per-artifact promotion, manifest last

**Files:**
- Modify: `src/features/retrieval/cli.py:26-48`
- Test: `tests/test_retrieval_cli.py`

**Interfaces:**
- Consumes: `embedding_inputs` (T1), `Bm25LexicalIndex.build_index`/`promote` (T2), `index_manifest.build_manifest`/`write` (T3).
- Produces: `run()` with the ordering contract below. No new public names.

**Ordering contract:**

1. Resolve profile and settings; load chunks; compute `chunks_sha256` **once** and reuse it for both the BM25 payload and the manifest, so the two cannot disagree.
2. Build both candidates: the Chroma `__candidate` collection (the adapter already does this internally) and the BM25 `.candidate` file. Neither is live yet.
3. Promote each artifact. Chroma's `_promote` already restores the previous collection if the rename fails.
4. Write the manifest **last**. It is the commit marker: a manifest on disk asserts that both artifacts were promoted successfully.
5. On any exception, delete the BM25 candidate and re-raise. A failed build leaves the previous manifest and previous artifacts in place — `verify()` then either passes on the old index or fails loudly; it never half-passes.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_retrieval_cli.py`:

```python
def test_manifest_is_written_after_both_artifacts_are_promoted(monkeypatch, tmp_path) -> None:
    """The manifest is the commit marker. If it were written first, a BM25
    failure would leave a manifest asserting an index that was never built."""
    order: list[str] = []
    ...  # patch vector_store.build_collection, lexical_index.promote and
        # index_manifest.write to append their names to `order`
    assert order.index("manifest_write") == len(order) - 1


def test_a_failed_bm25_build_leaves_the_previous_manifest_untouched(monkeypatch, tmp_path) -> None:
    before = manifest_path.read_bytes()

    with pytest.raises(RuntimeError):
        cli.run()

    assert manifest_path.read_bytes() == before
    assert not list(tmp_path.glob("*.candidate"))


def test_the_same_chunks_sha256_reaches_the_bm25_payload_and_the_manifest(tmp_path) -> None:
    ...  # run the CLI against tests/fixtures/mini_corpus, then assert
        # json.loads(bm25_path)["chunks_sha256"] == index_manifest.read().chunks_sha256
```

Build these on `tests/test_retrieval_cli.py`'s existing fake-vector-store pattern (line 77) so no real model loads; `tests/test_retrieval_cli_smoke.py` is the one that exercises the real path against `tests/fixtures/mini_corpus/`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_retrieval_cli.py -q`

Expected: FAIL — today `build_index` promotes immediately and nothing cleans up a candidate.

- [ ] **Step 3: Restructure `run()`**

```python
def run() -> None:
    settings = load_settings()
    profile = index_manifest.resolve_index_profile(settings)
    chunks = load_chunks()
    digest = index_manifest.chunks_sha256()

    embedder = SentenceTransformersEmbedder()
    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_path, embedder=embedder, index_profile=profile
    )
    lexical_index = Bm25LexicalIndex(persist_path=settings.bm25_path)

    candidate: Path | None = None
    try:
        vector_store.build_collection(chunks, embedding_inputs(chunks, profile))
        candidate = lexical_index.build_index(chunks, chunks_sha256=digest)
        lexical_index.promote(candidate)
        candidate = None
        # The manifest is written LAST, as the commit marker: its presence
        # asserts that both artifacts were built and promoted. Writing it
        # earlier would let a BM25 failure leave a manifest describing an
        # index that does not exist.
        index_manifest.write(index_manifest.build_manifest(profile, len(chunks)))
    except BaseException:
        if candidate is not None:
            candidate.unlink(missing_ok=True)
        raise

    print(f"Index profile: {profile}")
    ...  # the existing print block, plus one line for the lexical profile
```

`index_manifest.resolve_index_profile(settings)` is Bucket 1 Task 4's signature; if that bucket has not landed, call it with no argument.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_retrieval_cli.py tests/test_retrieval_cli_smoke.py -q`

Expected: PASS.

- [ ] **Step 5: Reindex and verify the rebuilt index measures the same**

This step needs the owner's explicit go-ahead — it overwrites the live `retrieval/output/`.

```bash
python -m src.features.retrieval.cli
pytest -q
ruff check src tests
mypy src
python -m src.features.evaluation.eval_set_integrity --verify
python -m src.features.evaluation.regression_set_integrity --verify
python -m src.features.evaluation.gate_holdout_integrity --verify
python -m src.features.evaluation.gate_score_guard
```

Expected: all green, and `gate_score_guard` confirms `r001`/`r002` top1-semantic are still inside `[0.5500, 0.5999)` on the freshly built index. Then run `retrieval_eval` with `index_profile="contextual-v1"` and compare against the current `__contextual-v1__` report: **recall and MRR must be identical**, because nothing in this bucket changes what gets embedded or how it is scored. Any difference is a bug in T1's policy extraction, not a new result. Write the new report to a fresh filename — never over a baseline. Stop at green; committing needs the owner's separate authorization.
