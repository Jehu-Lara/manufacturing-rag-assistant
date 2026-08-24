from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(\S.*?)\s*$")

TARGET_CHUNK_TOKENS = 500
CHUNK_UPPER_BOUND_TOKENS = 600
OVERLAP_RATIO = 0.15

_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is None:
        try:
            _encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:
            raise RuntimeError(
                "tiktoken encoding not cached and no network available — "
                "run once with network access to populate the local cache"
            ) from exc
    return _encoding


def count_tokens(text: str) -> int:
    return len(_get_encoding().encode(text))


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

    def flush(end_line: int) -> None:
        text = "\n".join(content_lines).strip("\n")
        if not text.strip():
            return
        breadcrumb_parts = [title for level, title in heading_stack if level >= 2]
        breadcrumb = " > ".join(breadcrumb_parts) if breadcrumb_parts else (
            heading_stack[-1][1] if heading_stack else "(document body)"
        )
        sections.append(Section(breadcrumb=breadcrumb, start_line=content_start_line, end_line=end_line, text=text))

    for line_number, line in enumerate(lines, start=1):
        match = HEADING_PATTERN.match(line)
        if match:
            flush(line_number - 1)
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = [(lvl, txt) for lvl, txt in heading_stack if lvl < level]
            heading_stack.append((level, title))
            content_lines = []
            content_start_line = line_number + 1
        else:
            content_lines.append(line)

    flush(len(lines))
    return sections


def _split_section(section: Section) -> list[RawChunk]:
    lines = section.text.split("\n")
    line_token_counts = [count_tokens(line) for line in lines]
    total_tokens = sum(line_token_counts)

    if total_tokens <= CHUNK_UPPER_BOUND_TOKENS:
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


def chunk_document(body: str) -> list[RawChunk]:
    chunks: list[RawChunk] = []
    for section in split_into_sections(body):
        chunks.extend(_split_section(section))
    return chunks
