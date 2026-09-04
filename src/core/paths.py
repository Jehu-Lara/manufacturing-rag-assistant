from __future__ import annotations

from pathlib import Path

# The single repo-root authority. This file sits at src/core/paths.py, so
# parent.parent.parent lands on the repo root — one hop shallower than the
# src/features/**/x.py modules that used four. Counting hops from each module's
# own depth is what breaks silently when a module moves between package levels,
# so every other module binds its constants from here.
#
# Deliberately imports nothing but pathlib: the integrity guards bind their
# paths from here and must never pull in the embedder/chromadb import chain.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

CORPUS_DIR = REPO_ROOT / "corpus"
INGESTION_OUTPUT_DIR = REPO_ROOT / "ingestion" / "output"
CHUNKS_FILE = INGESTION_OUTPUT_DIR / "chunks.jsonl"
RETRIEVAL_OUTPUT_DIR = REPO_ROOT / "retrieval" / "output"
EVAL_DIR = REPO_ROOT / "eval"
EVAL_REPORTS_DIR = EVAL_DIR / "reports"
