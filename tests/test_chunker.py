from __future__ import annotations

from ingestion.chunker import (
    CHUNK_UPPER_BOUND_TOKENS,
    MIN_MERGE_TOKENS,
    TARGET_CHUNK_TOKENS,
    _merge_small_sibling_sections,
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
    merged_sections = _merge_small_sibling_sections(split_into_sections(body))
    chunks = chunk_document(body)

    for chunk in chunks:
        candidates = [s for s in merged_sections if s.breadcrumb == chunk.section_breadcrumb]
        assert any(s.start_line <= chunk.start_line and chunk.end_line <= s.end_line for s in candidates), (
            f"chunk {chunk.start_line}-{chunk.end_line} ({chunk.section_breadcrumb!r}) "
            "escaped its (possibly merged) section's line range"
        )


def test_count_tokens_is_positive_for_nonempty_text():
    assert count_tokens("hello world") > 0


def _build_body_with_small_siblings() -> str:
    # Each leaf is well under MIN_MERGE_TOKENS on its own.
    return (
        "# Sample Manual\n"
        "\n"
        "## Parent Section\n"
        "\n"
        "### Leaf A\n"
        "\n"
        "Tiny content A.\n"
        "\n"
        "### Leaf B\n"
        "\n"
        "Tiny content B.\n"
        "\n"
        "### Leaf C\n"
        "\n"
        "Tiny content C.\n"
        "\n"
        "## Unrelated Top-Level Section\n"
        "\n"
        "Also tiny, but not a sibling of the leaves above.\n"
    )


def test_small_sibling_sections_are_merged_into_one_chunk():
    body = _build_body_with_small_siblings()
    for leaf in ("Leaf A", "Leaf B", "Leaf C"):
        assert count_tokens(f"Tiny content {leaf[-1]}.") < MIN_MERGE_TOKENS

    chunks = chunk_document(body)
    merged = [c for c in chunks if c.section_breadcrumb.startswith("Parent Section > ")]

    assert len(merged) == 1, "expected the three small siblings to collapse into a single chunk"
    assert merged[0].section_breadcrumb == "Parent Section > Leaf A; Leaf B; Leaf C"
    assert "Tiny content A." in merged[0].text
    assert "Tiny content B." in merged[0].text
    assert "Tiny content C." in merged[0].text


def test_unrelated_top_level_sections_are_never_merged_with_each_other():
    # "Parent Section" (before merging) and "Unrelated Top-Level Section" are
    # both top-level (no shared parent) — merging them would blend unrelated
    # topics into one citation, so this must never happen even though both
    # are small.
    body = _build_body_with_small_siblings()
    chunks = chunk_document(body)
    breadcrumbs = {c.section_breadcrumb for c in chunks}

    assert "Unrelated Top-Level Section" in breadcrumbs
    assert not any("Unrelated Top-Level Section" in b and "Leaf" in b for b in breadcrumbs)
