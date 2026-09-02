from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import tiktoken

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(\S.*?)\s*$")

TARGET_CHUNK_TOKENS = 500

# 600 was one constant named CHUNK_UPPER_BOUND_TOKENS, which it never was: no
# chunk is bounded by it. A section under it is emitted whole (so a chunk can
# legitimately exceed it after line-boundary splitting), and the real ceiling
# that matters is the embedding model's max_seq_length, checked separately by
# the embedder. Two names for the two distinct decisions it actually drives.
MAX_MERGED_SECTION_TOKENS = 600
SECTION_SPLIT_TRIGGER_TOKENS = 600
OVERLAP_RATIO = 0.15
MIN_MERGE_TOKENS = 150
"""Sections below this size are candidates to merge with a following sibling
section (same immediate parent heading) rather than standing alone as a
near-empty chunk. See `_merge_small_sibling_sections`."""


class TiktokenCounter:
    """cl100k_base token counter. Instance-scoped (not a module-level
    singleton) — the composition root (the ingestion CLI) constructs one and
    threads it through a whole run, rather than every call reaching for a
    shared global."""

    def __init__(self) -> None:
        self._encoding: Optional[tiktoken.Encoding] = None

    def _get_encoding(self) -> tiktoken.Encoding:
        if self._encoding is None:
            try:
                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as exc:
                raise RuntimeError(
                    "tiktoken encoding not cached and no network available — "
                    "run once with network access to populate the local cache"
                ) from exc
        return self._encoding

    def count_tokens(self, text: str) -> int:
        return len(self._get_encoding().encode(text))


@dataclass(frozen=True)
class Section:
    """A run of body text under a given heading path, before the next heading of any level."""

    breadcrumb: str
    start_line: int  # 1-indexed, inclusive, relative to the document body
    end_line: int  # 1-indexed, inclusive
    text: str


@dataclass(frozen=True)
class RawChunk:
    section_breadcrumb: str
    start_line: int  # 1-indexed, inclusive, relative to the document body
    end_line: int  # 1-indexed, inclusive
    text: str
    token_count: int


def split_into_sections(body: str) -> list[Section]:
    lines = body.splitlines()
    heading_stack: list[tuple[int, str]] = []
    sections: list[Section] = []
    content_lines: list[str] = []
    content_start_line = 1

    def flush() -> None:
        # Trim leading/trailing blank lines, but track exactly how many were
        # trimmed from the front so start_line stays aligned with the first
        # REMAINING line — every downstream line-index calculation (sub-split
        # chunk boundaries, md_line_range) assumes text.split("\n")[0]
        # corresponds to body line `start_line`. Getting this wrong doesn't
        # corrupt chunk text (that's always assembled by index, not by
        # re-slicing the file), but it does corrupt the line-range citation,
        # which silently points a line or more short of the true content —
        # for the last chunk of a multi-chunk section, that visibly drops the
        # final line's citation coverage entirely.
        start_offset = 0
        while start_offset < len(content_lines) and not content_lines[start_offset].strip():
            start_offset += 1
        end_offset = len(content_lines)
        while end_offset > start_offset and not content_lines[end_offset - 1].strip():
            end_offset -= 1

        trimmed_lines = content_lines[start_offset:end_offset]
        if not trimmed_lines:
            return

        text = "\n".join(trimmed_lines)
        breadcrumb_parts = [title for level, title in heading_stack if level >= 2]
        breadcrumb = " > ".join(breadcrumb_parts) if breadcrumb_parts else (
            heading_stack[-1][1] if heading_stack else "(document body)"
        )
        actual_start_line = content_start_line + start_offset
        actual_end_line = content_start_line + end_offset - 1
        sections.append(
            Section(breadcrumb=breadcrumb, start_line=actual_start_line, end_line=actual_end_line, text=text)
        )

    for line_number, line in enumerate(lines, start=1):
        match = HEADING_PATTERN.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = [(lvl, txt) for lvl, txt in heading_stack if lvl < level]
            heading_stack.append((level, title))
            content_lines = []
            content_start_line = line_number + 1
        else:
            content_lines.append(line)

    flush()
    return sections


def _parent_breadcrumb(breadcrumb: str) -> str:
    """The breadcrumb one level up, or "" if `breadcrumb` is already top-level.

    Top-level ("") is deliberately never treated as a shared parent by
    `_merge_small_sibling_sections` — two unrelated top-level sections (e.g.
    "DC Sources" and "DC Circuit Terminology") happening to both be small
    doesn't make them siblings worth blending into one citation.
    """
    if " > " in breadcrumb:
        return breadcrumb.rsplit(" > ", 1)[0]
    return ""


def _merge_small_sibling_sections(sections: list[Section], counter: TiktokenCounter) -> list[Section]:
    """Combine a run of consecutive, same-parent sections that are individually
    below MIN_MERGE_TOKENS into one section, growing toward TARGET_CHUNK_TOKENS
    and never exceeding MAX_MERGED_SECTION_TOKENS.

    Without this pass, a document with many short sibling sections (a CFR
    subpart full of one-sentence provisions, a glossary-style block of short
    subsections) produces a long tail of chunks too small to carry useful
    embedding signal. Merging preserves per-provision citability by listing
    every merged leaf title in the combined breadcrumb, rather than
    collapsing to just the shared parent's name.
    """
    merged: list[Section] = []
    i = 0
    n = len(sections)

    while i < n:
        current = sections[i]
        current_tokens = counter.count_tokens(current.text)
        parent = _parent_breadcrumb(current.breadcrumb)

        if current_tokens >= MIN_MERGE_TOKENS or not parent:
            merged.append(current)
            i += 1
            continue

        group = [current]
        group_tokens = current_tokens
        j = i + 1
        while j < n and _parent_breadcrumb(sections[j].breadcrumb) == parent:
            candidate_tokens = counter.count_tokens(sections[j].text)
            if group_tokens + candidate_tokens > MAX_MERGED_SECTION_TOKENS:
                break
            group.append(sections[j])
            group_tokens += candidate_tokens
            j += 1
            if group_tokens >= TARGET_CHUNK_TOKENS:
                break

        if len(group) == 1:
            merged.append(current)
            i += 1
            continue

        leaf_titles = [s.breadcrumb.rsplit(" > ", 1)[-1] for s in group]
        merged.append(
            Section(
                breadcrumb=f"{parent} > {'; '.join(leaf_titles)}",
                start_line=group[0].start_line,
                end_line=group[-1].end_line,
                text="\n\n".join(s.text for s in group),
            )
        )
        i = j

    return merged


def _split_section(section: Section, counter: TiktokenCounter) -> list[RawChunk]:
    lines = section.text.split("\n")
    line_token_counts = [counter.count_tokens(line) for line in lines]
    total_tokens = sum(line_token_counts)

    if total_tokens <= SECTION_SPLIT_TRIGGER_TOKENS:
        return [
            RawChunk(
                section_breadcrumb=section.breadcrumb,
                start_line=section.start_line,
                end_line=section.end_line,
                text=section.text,
                token_count=total_tokens,
            )
        ]

    overlap_tokens = round(TARGET_CHUNK_TOKENS * OVERLAP_RATIO)
    chunks: list[RawChunk] = []
    n = len(lines)
    line_idx = 0

    while line_idx < n:
        start_idx = line_idx
        chunk_lines: list[str] = []
        chunk_token_total = 0
        while line_idx < n and chunk_token_total < TARGET_CHUNK_TOKENS:
            chunk_lines.append(lines[line_idx])
            chunk_token_total += line_token_counts[line_idx]
            line_idx += 1

        chunks.append(
            RawChunk(
                section_breadcrumb=section.breadcrumb,
                start_line=section.start_line + start_idx,
                end_line=section.start_line + line_idx - 1,
                text="\n".join(chunk_lines),
                token_count=chunk_token_total,
            )
        )

        if line_idx >= n:
            break

        back_idx = line_idx
        overlap_accum = 0
        while back_idx > start_idx and overlap_accum < overlap_tokens:
            back_idx -= 1
            overlap_accum += line_token_counts[back_idx]
        # Guarantee forward progress even if a single line's token count alone
        # meets or exceeds the overlap target (which would otherwise regenerate
        # the same chunk boundary forever).
        line_idx = max(back_idx, start_idx + 1)

    return chunks


def chunk_document(body: str, counter: Optional[TiktokenCounter] = None) -> list[RawChunk]:
    counter = counter or TiktokenCounter()
    sections = _merge_small_sibling_sections(split_into_sections(body), counter)
    chunks: list[RawChunk] = []
    for section in sections:
        chunks.extend(_split_section(section, counter))
    return chunks
