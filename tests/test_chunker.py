from __future__ import annotations

from ingestion.chunker import (
    CHUNK_UPPER_BOUND_TOKENS,
    TARGET_CHUNK_TOKENS,
    chunk_document,
    count_tokens,
    split_into_sections,
)

_LINE = "The quick brown fox jumps over the lazy dog near the old mill pond today."


def _build_body(num_long_section_lines: int) -> str:
    long_section = "\n".join(f"{_LINE} (line {i})" for i in range(num_long_section_lines))
    return (
        "# Sample Manual\n"
        "\n"
        "Intro paragraph before any subsection.\n"
        "\n"
        "## Section One\n"
        "\n"
        "Short content for section one.\n"
        "\n"
        "## Section Two\n"
        "\n"
        "### Subsection Two A\n"
        "\n"
        f"{long_section}\n"
        "\n"
        "## Section Three\n"
        "\n"
        "Short content for section three.\n"
    )


def test_split_into_sections_builds_correct_breadcrumbs():
    sections = split_into_sections(_build_body(num_long_section_lines=5))
    breadcrumbs = [s.breadcrumb for s in sections]

    assert "Sample Manual" in breadcrumbs  # intro block before any H2, falls back to the H1 title
    assert "Section One" in breadcrumbs
    assert "Section Two > Subsection Two A" in breadcrumbs
    assert "Section Three" in breadcrumbs


def test_short_sections_produce_a_single_chunk():
    chunks = chunk_document(_build_body(num_long_section_lines=5))
    section_one_chunks = [c for c in chunks if c.section_breadcrumb == "Section One"]

    assert len(section_one_chunks) == 1
    assert section_one_chunks[0].token_count <= CHUNK_UPPER_BOUND_TOKENS


def test_long_section_is_split_into_target_band_with_overlap():
    # 80 short, roughly equal-sized lines comfortably exceeds the upper bound
    # several times over, so this exercises the sub-splitting path.
    chunks = chunk_document(_build_body(num_long_section_lines=80))
    long_chunks = [c for c in chunks if c.section_breadcrumb == "Section Two > Subsection Two A"]

    assert len(long_chunks) >= 3, "expected the long section to be split into multiple chunks"

    for chunk in long_chunks[:-1]:
        # Documented tolerance: chunking splits on line boundaries only (never
        # mid-line, to avoid breaking mid-sentence), so a chunk can land
        # somewhat outside the exact target band. Only the LAST chunk in a
        # section is allowed to be small (whatever's left over); every other
        # chunk should still land in a band around the target.
        assert TARGET_CHUNK_TOKENS - 100 <= chunk.token_count <= CHUNK_UPPER_BOUND_TOKENS + 50, (
            f"chunk {chunk.start_line}-{chunk.end_line} token_count={chunk.token_count} "
            "outside the expected band"
        )

    for previous, current in zip(long_chunks, long_chunks[1:]):
        assert current.start_line <= previous.end_line, "expected consecutive chunks in a split section to overlap"


def test_chunks_never_cross_section_boundaries():
    body = _build_body(num_long_section_lines=80)
    sections = split_into_sections(body)
    chunks = chunk_document(body)

    for chunk in chunks:
        candidates = [s for s in sections if s.breadcrumb == chunk.section_breadcrumb]
        assert any(s.start_line <= chunk.start_line and chunk.end_line <= s.end_line for s in candidates), (
            f"chunk {chunk.start_line}-{chunk.end_line} ({chunk.section_breadcrumb!r}) "
            "escaped its section's line range"
        )


def test_count_tokens_is_positive_for_nonempty_text():
    assert count_tokens("hello world") > 0
