"""Centralized configuration loaded from .env.

Every other module imports settings from here instead of calling
os.getenv directly, so there is exactly one place that knows the
environment variable names.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = REPO_ROOT / "knowledge"

load_dotenv(REPO_ROOT / ".env")


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


class Settings:
    """Lazy accessors so importing this module never fails by itself.

    Values are only validated when actually read, which lets read-only
    tooling (like listing agents) import the package without a full .env.
    """

    @property
    def google_api_key(self) -> str:
        return _require("GOOGLE_API_KEY")

    @property
    def pinecone_api_key(self) -> str:
        return _require("PINECONE_API_KEY")

    @property
    def pinecone_index_name(self) -> str:
        return _require("INDEX_NAME")

    @property
    def embedding_model(self) -> str:
        # text-embedding-004 is being phased out; gemini-embedding-001 is the
        # current recommended model and supports truncating its native
        # 3072-dim output down to embedding_dimension via output_dimensionality.
        return os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

    @property
    def embedding_dimension(self) -> int:
        return int(os.getenv("EMBEDDING_DIMENSION", "768"))

    @property
    def chat_model(self) -> str:
        return os.getenv("CHAT_MODEL", "gemini-2.5-flash")

    @property
    def pinecone_cloud(self) -> str:
        return os.getenv("PINECONE_CLOUD", "aws")

    @property
    def pinecone_region(self) -> str:
        return os.getenv("PINECONE_REGION", "us-east-1")

    @property
    def langsmith_project(self) -> str:
        return os.getenv("LANGSMITH_PROJECT", "travel-recommendation-agents")

    def configure_langsmith(self) -> None:
        """Turn on LangSmith tracing for every LLM/tool/graph call.

        langchain-core reads the LANGSMITH_* names directly in recent
        versions, but we also mirror them onto the legacy LANGCHAIN_*
        names for compatibility with older tooling that only looks there.
        """
        tracing_enabled = os.getenv("LANGSMITH_TRACING", "false")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", tracing_enabled)
        if os.getenv("LANGSMITH_API_KEY"):
            os.environ.setdefault("LANGCHAIN_API_KEY", os.environ["LANGSMITH_API_KEY"])
        if os.getenv("LANGSMITH_ENDPOINT"):
            os.environ.setdefault("LANGCHAIN_ENDPOINT", os.environ["LANGSMITH_ENDPOINT"])
        os.environ.setdefault("LANGCHAIN_PROJECT", self.langsmith_project)
        os.environ.setdefault("LANGSMITH_PROJECT", self.langsmith_project)


settings = Settings()
