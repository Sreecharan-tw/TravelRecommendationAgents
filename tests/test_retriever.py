"""Tests for markdown chunking and the retriever tool.

These run without any live Google/Pinecone credentials: chunking is
pure-Python, and the retriever tool's vector store is monkeypatched.
"""
from __future__ import annotations

from langchain_core.documents import Document

from src.ingestion.loader import load_and_chunk


def test_load_and_chunk_produces_documents_with_required_metadata():
    docs = load_and_chunk()

    assert len(docs) > 0
    for doc in docs:
        assert doc.page_content.strip()
        assert doc.metadata["source"].endswith(".md")
        assert doc.metadata["destination"]
        assert doc.metadata["section"]
        assert doc.metadata["chunk_id"].startswith(doc.metadata["source"])


def test_chunk_ids_are_deterministic_across_runs():
    first_pass = {d.metadata["chunk_id"] for d in load_and_chunk()}
    second_pass = {d.metadata["chunk_id"] for d in load_and_chunk()}

    assert first_pass == second_pass


def test_chunk_ids_are_unique():
    ids = [d.metadata["chunk_id"] for d in load_and_chunk()]
    assert len(ids) == len(set(ids))


def test_known_destinations_are_present():
    docs = load_and_chunk()
    destinations = {d.metadata["destination"] for d in docs}
    assert "Kashmir" in destinations
    assert "Kerala" in destinations
    assert "Meghalaya" in destinations
    assert "Andaman And Nicobar" in destinations


def test_search_travel_knowledge_returns_serializable_chunks_from_store(monkeypatch):
    from src.tools import retriever_tool

    fake_docs = [
        Document(
            page_content="Gulmarg has India's only ski resort.",
            metadata={"source": "kashmir.md", "destination": "Kashmir", "section": "Overview"},
        )
    ]

    class FakeStore:
        def similarity_search(self, query, **kwargs):
            self.last_query = query
            self.last_kwargs = kwargs
            return fake_docs

    fake_store = FakeStore()
    monkeypatch.setattr(retriever_tool, "_store", lambda: fake_store)

    result = retriever_tool.search_travel_knowledge.invoke(
        {"query": "skiing in Kashmir", "destination": "Kashmir", "k": 3}
    )

    assert result == [
        {
            "content": "Gulmarg has India's only ski resort.",
            "source": "kashmir.md",
            "destination": "Kashmir",
            "section": "Overview",
        }
    ]
    assert fake_store.last_kwargs["filter"] == {"destination": {"$eq": "Kashmir"}}
    assert fake_store.last_kwargs["k"] == 3
