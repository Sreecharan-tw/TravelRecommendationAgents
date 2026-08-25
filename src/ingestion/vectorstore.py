"""Pinecone index management and embeddings.

Provides one function to get an initialized Pinecone-backed LangChain
vector store, creating the serverless index on first use if it does
not already exist.
"""
from __future__ import annotations

from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import settings


class _FixedDimensionEmbeddings(GoogleGenerativeAIEmbeddings):
    """Pins every embed call to settings.embedding_dimension.

    gemini-embedding-001 natively outputs 3072-dim vectors but supports
    Matryoshka-style truncation via output_dimensionality. The vector
    store callers (langchain_pinecone) call embed_documents/embed_query
    with no way to pass that kwarg through, so we default it here to
    match whatever dimension the Pinecone index was created with.
    """

    def embed_documents(self, texts: List[str], **kwargs) -> List[List[float]]:
        kwargs.setdefault("output_dimensionality", settings.embedding_dimension)
        return super().embed_documents(texts, **kwargs)

    def embed_query(self, text: str, **kwargs) -> List[float]:
        kwargs.setdefault("output_dimensionality", settings.embedding_dimension)
        return super().embed_query(text, **kwargs)

    async def aembed_documents(self, texts: List[str], **kwargs) -> List[List[float]]:
        kwargs.setdefault("output_dimensionality", settings.embedding_dimension)
        return await super().aembed_documents(texts, **kwargs)

    async def aembed_query(self, text: str, **kwargs) -> List[float]:
        kwargs.setdefault("output_dimensionality", settings.embedding_dimension)
        return await super().aembed_query(text, **kwargs)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return _FixedDimensionEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )


def ensure_index(pc: Pinecone) -> None:
    index_name = settings.pinecone_index_name
    existing = {idx["name"] for idx in pc.list_indexes()}
    if index_name in existing:
        return
    pc.create_index(
        name=index_name,
        dimension=settings.embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
    )


def get_vector_store() -> PineconeVectorStore:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    ensure_index(pc)
    index = pc.Index(settings.pinecone_index_name)
    return PineconeVectorStore(index=index, embedding=get_embeddings())
