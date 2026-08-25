"""Build or refresh the Pinecone index from /knowledge.

Safe to re-run any time a markdown file changes: chunk ids are a
deterministic hash of (source file, section, content), so re-ingesting
upserts existing chunks in place instead of creating duplicates.
Chunks whose source content changed get a new id and the old id is
simply orphaned -- run with --prune to remove ids for a source file
that no longer produce any current chunk (e.g. a section was deleted).

Usage:
    python -m src.ingestion.ingest              # upsert all knowledge files
    python -m src.ingestion.ingest --prune       # also delete stale vectors
"""
from __future__ import annotations

import argparse
import sys

from src.ingestion.loader import load_and_chunk
from src.ingestion.vectorstore import get_vector_store
from src.config import settings

BATCH_SIZE = 100


def _existing_ids_for_sources(index, sources: set[str]) -> set[str]:
    """Fetch every vector id currently stored for the given source files.

    Uses metadata filtering + pagination over Pinecone's list/fetch API
    so pruning does not require keeping a separate id manifest on disk.
    """
    existing: set[str] = set()
    for source in sources:
        for ids_page in index.list(prefix=f"{source}::"):
            existing.update(ids_page)
    return existing


def run(prune: bool = False) -> None:
    documents = load_and_chunk()
    if not documents:
        print("No documents found under /knowledge -- nothing to ingest.")
        return

    ids = [doc.metadata["chunk_id"] for doc in documents]
    store = get_vector_store()

    print(f"Upserting {len(documents)} chunks into index '{settings.pinecone_index_name}'...")
    for start in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[start : start + BATCH_SIZE]
        batch_ids = ids[start : start + BATCH_SIZE]
        store.add_documents(documents=batch_docs, ids=batch_ids)
        print(f"  upserted {min(start + BATCH_SIZE, len(documents))}/{len(documents)}")

    if prune:
        sources = {doc.metadata["source"] for doc in documents}
        index = store.index
        existing_ids = _existing_ids_for_sources(index, sources)
        stale_ids = existing_ids - set(ids)
        if stale_ids:
            print(f"Pruning {len(stale_ids)} stale vectors from removed/changed sections...")
            index.delete(ids=list(stale_ids))
        else:
            print("No stale vectors to prune.")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete vectors for a source file that no longer correspond to any current chunk.",
    )
    args = parser.parse_args()
    run(prune=args.prune)


if __name__ == "__main__":
    sys.exit(main())
