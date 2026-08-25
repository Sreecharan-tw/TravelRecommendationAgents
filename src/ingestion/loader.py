"""Load and chunk the /knowledge markdown files.

Splitting is header-aware: each destination file is first split on
'#' / '##' headings (so a chunk never straddles two unrelated
sections), then any oversized section is further split by character
count so embeddings stay within a sane token budget.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.config import KNOWLEDGE_DIR

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
]

MAX_CHUNK_CHARS = 1800
CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class DestinationFile:
    path: Path
    destination: str


_DESTINATION_SECTION_RE = re.compile(r"^#\s+Destination\s*\n+([^\n#]+)", re.MULTILINE)
_TITLE_LINE_RE = re.compile(r"^#{1,2}\s+(.+?)\s+Destination Knowledge Document", re.MULTILINE)


def discover_knowledge_files(knowledge_dir: Path = KNOWLEDGE_DIR) -> list[DestinationFile]:
    files = sorted(knowledge_dir.glob("*.md"))
    return [
        DestinationFile(path=p, destination=_destination_name(p, p.read_text(encoding="utf-8")))
        for p in files
    ]


def _destination_name(path: Path, raw_text: str) -> str:
    """Derive a human-readable destination name.

    Prefers an explicit '# Destination' section or a
    '## <Name> Destination Knowledge Document' title line from the
    document itself, since filenames (e.g. 'andamanandnicobar.md')
    are not reliably splittable into words. Falls back to
    title-casing the filename.
    """
    if match := _DESTINATION_SECTION_RE.search(raw_text):
        return match.group(1).strip()
    if match := _TITLE_LINE_RE.search(raw_text):
        return match.group(1).strip()
    stem = path.stem.replace("_", " ").replace("-", " ")
    return stem.title()


def _chunk_id(source: str, section: str, content: str) -> str:
    """Deterministic id so re-ingestion upserts in place instead of duplicating."""
    digest = hashlib.sha256(f"{source}::{section}::{content}".encode("utf-8")).hexdigest()
    return f"{source}::{digest[:16]}"


def load_and_chunk(knowledge_dir: Path = KNOWLEDGE_DIR) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    for dest_file in discover_knowledge_files(knowledge_dir):
        raw_text = dest_file.path.read_text(encoding="utf-8")
        header_docs = header_splitter.split_text(raw_text)

        for header_doc in header_docs:
            section = header_doc.metadata.get("h2") or header_doc.metadata.get("h1") or "General"
            top_section = header_doc.metadata.get("h1", "General")

            sub_docs = char_splitter.split_documents([header_doc])
            for i, sub_doc in enumerate(sub_docs):
                metadata = {
                    "source": dest_file.path.name,
                    "destination": dest_file.destination,
                    "section": section,
                    "top_section": top_section,
                    "chunk_index": i,
                }
                chunk_id = _chunk_id(dest_file.path.name, section, sub_doc.page_content)
                metadata["chunk_id"] = chunk_id
                documents.append(
                    Document(page_content=sub_doc.page_content, metadata=metadata, id=chunk_id)
                )

    return documents
