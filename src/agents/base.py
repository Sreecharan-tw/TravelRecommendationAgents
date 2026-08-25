"""Shared machinery for building/running a specialist ReAct agent node."""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from src.config import settings
from src.graph.state import AgentOutput
from src.tools.grounding import check_grounding
from src.tools.retriever_tool import search_travel_knowledge


def get_chat_model(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
    )


def build_specialist_agent(system_prompt: str, name: str):
    """A single-purpose ReAct agent bound to the knowledge retriever tool.

    Because it's a ReAct loop (not a one-shot prompt with pre-fetched
    context), the agent decides for itself how many times to call the
    tool and with what query -- this is what makes retrieval "agentic"
    rather than a hardcoded similarity call.
    """
    model = get_chat_model()
    return create_react_agent(
        model=model,
        tools=[search_travel_knowledge],
        prompt=system_prompt,
        name=name,
    )


def run_specialist(agent, task_message: str, run_name: str, tags: list[str]) -> AgentOutput:
    """Invoke a specialist agent and collect its answer + traceable sources."""
    result = agent.invoke(
        {"messages": [HumanMessage(content=task_message)]},
        config={"run_name": run_name, "tags": tags},
    )
    messages = result["messages"]

    final_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    #  This is the traceability piece — nothing here calls the retriever again;
    #  it's just replaying what the agent already retrieved during its own loop.
    sources: list[dict[str, str]] = []
    retrieved_texts: list[str] = []
    seen = set()
    for msg in messages:
        if not isinstance(msg, ToolMessage) or msg.name != "search_travel_knowledge":
            continue
        if isinstance(msg.content, list):
            chunks = msg.content
        elif isinstance(msg.content, str):
            try:
                chunks = json.loads(msg.content)
            except json.JSONDecodeError:
                chunks = []
        else:
            chunks = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            retrieved_texts.append(chunk.get("content", ""))
            key = (chunk.get("source"), chunk.get("section"))
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "source": chunk.get("source", "unknown"),
                        "destination": chunk.get("destination", "unknown"),
                        "section": chunk.get("section", "unknown"),
                    }
                )

    grounding = check_grounding(final_text, retrieved_texts)

    return AgentOutput(
        text=final_text,
        sources=sources,
        flagged_sentences=grounding.flagged_sentences,
    )
