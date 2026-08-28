from __future__ import annotations

import json
from pathlib import Path

from src.domain.models import ChunkMetadata
from src.features.ingestion.chunker import TiktokenCounter, chunk_document
from src.features.ingestion.use_cases import Document, load_corpus

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "ingestion" / "output"
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"


def build_chunks_for_document(document: Document, counter: TiktokenCounter) -> list[ChunkMetadata]:
    chunks: list[ChunkMetadata] = []
    for index, raw in enumerate(chunk_document(document.body, counter), start=1):
        chunk = ChunkMetadata(
            chunk_id=f"{document.document_id}::chunk-{index:04d}",
            document_id=document.document_id,
            document_title=document.document_title,
            revision=document.revision,
            section_heading=raw.section_breadcrumb,
            source_type=document.source_type,
            source_url_or_note=document.source_url_or_note,
            source_page_range=document.source_page_range,
            md_line_range=(
                f"{document.body_start_line + raw.start_line - 1}-"
                f"{document.body_start_line + raw.end_line - 1}"
            ),
            chunk_token_count=raw.token_count,
            chunk_text=raw.text,
        )
        chunk.validate()
        chunks.append(chunk)
    return chunks


def run() -> list[ChunkMetadata]:
    documents = load_corpus()
    counter = TiktokenCounter()

    all_chunks: list[ChunkMetadata] = []
    for document in documents:
        all_chunks.extend(build_chunks_for_document(document, counter))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    public_count = sum(1 for d in documents if d.source_type == "public")
    synthetic_count = sum(1 for d in documents if d.source_type == "synthetic")
    total_words = sum(len(d.body.split()) for d in documents)
    total_chars = sum(len(d.body) for d in documents)

    print(f"Documents processed: {len(documents)} ({public_count} public, {synthetic_count} synthetic)")
    print(f"Chunks produced: {len(all_chunks)}")
    print(f"Total corpus size: {total_words} words, {total_chars} characters")
    print(f"Chunks written to: {OUTPUT_FILE}")

    return all_chunks


if __name__ == "__main__":
    run()
