"""Agentic RAG retriever tool.

This is a LangChain @tool, not a hardcoded similarity call baked into
a chain -- each agent decides for itself whether and how to call it
(what query to use, how many results, whether to call it again with a
refined query). Every call returns the raw chunks plus their source
metadata so answers stay traceable back to specific markdown sections.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.ingestion.vectorstore import get_vector_store

DEFAULT_K = 5


class RetrievedChunk(BaseModel):
    content: str
    source: str
    destination: str
    section: str


@lru_cache(maxsize=1)
def _store():
    return get_vector_store()


class KnowledgeSearchInput(BaseModel):
    query: str = Field(description="Natural-language search query about a destination.")
    destination: str | None = Field(
        default=None,
        description=(
            "Optional destination name to restrict the search to "
            "(e.g. 'Kashmir', 'Kerala'). Leave empty to search all destinations."
        ),
    )
    k: int = Field(default=DEFAULT_K, description="Number of chunks to retrieve.")


@tool("search_travel_knowledge", args_schema=KnowledgeSearchInput)
def search_travel_knowledge(
    query: str, destination: str | None = None, k: int = DEFAULT_K
) -> list[dict]:
    """Search the /knowledge markdown base for travel information.

    Use this tool to look up anything about a destination -- attractions,
    costs, best time to visit, transport, food, safety, etc. Always call
    this before stating any fact, price, or recommendation. If the results
    don't contain what you need, you may call it again with a different
    query, but never invent information that isn't returned here.
    """
    search_kwargs: dict = {"k": k}
    if destination:
        search_kwargs["filter"] = {"destination": {"$eq": destination}}

    results = _store().similarity_search(query, **search_kwargs)
    return [
        RetrievedChunk(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            destination=doc.metadata.get("destination", "unknown"),
            section=doc.metadata.get("section", "unknown"),
        ).model_dump()
        for doc in results
    ]
