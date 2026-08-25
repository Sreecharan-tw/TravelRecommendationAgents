"""Tests for agent graph nodes.

Nodes are only skipped or exercised in ways that don't require a live
GOOGLE_API_KEY: the "not routed this turn" skip path (a real no-op used
in production to avoid a wasted LLM call), and the merge node's pure
combination logic. The LLM-calling path itself is exercised manually /
in the Streamlit app, since mocking Gemini's structured-output wire
format would test the mock rather than the integration.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.agents.budget_agent import budget_planning_node
from src.agents.destination_agent import _build_task, destination_research_node
from src.graph.coordinator import merge_node
from src.graph.state import AgentOutput


def _base_state(route_overrides: dict) -> dict:
    route = {
        "need_destination_research": False,
        "need_budget_planning": False,
        "need_itinerary_planning": False,
        "reasoning": "test",
        **route_overrides,
    }
    return {
        "messages": [HumanMessage(content="Tell me about Kashmir")],
        "trip_params": {"destination": "Kashmir"},
        "route": route,
    }


def test_destination_research_node_skips_when_not_routed():
    state = _base_state({"need_destination_research": False})
    result = destination_research_node(state)
    assert result == {}


def test_budget_planning_node_skips_when_not_routed():
    state = _base_state({"need_budget_planning": False})
    result = budget_planning_node(state)
    assert result == {}


def test_destination_research_task_includes_trip_params():
    state = _base_state({"need_destination_research": True})
    state["trip_params"]["interests"] = "trekking"
    task = _build_task(state)
    assert "Kashmir" in task
    assert "trekking" in task
    assert "Tell me about Kashmir" in task


def _fake_output(text: str) -> AgentOutput:
    return AgentOutput(
        text=text,
        sources=[{"source": "kashmir.md", "destination": "Kashmir", "section": "Overview"}],
        flagged_sentences=[],
    )


def test_merge_node_passes_through_single_agent_output_verbatim():
    state = {
        "destination_research": _fake_output("Kashmir is known for Dal Lake."),
        "budget_planning": None,
        "itinerary_planning": None,
    }
    result = merge_node(state)
    assert result["final_response"] == "Kashmir is known for Dal Lake."
    assert result["all_sources"] == [
        {"source": "kashmir.md", "destination": "Kashmir", "section": "Overview"}
    ]


def test_merge_node_handles_no_agent_output():
    state = {"destination_research": None, "budget_planning": None, "itinerary_planning": None}
    result = merge_node(state)
    assert "clarify" in result["final_response"].lower()
    assert result["all_sources"] == []
