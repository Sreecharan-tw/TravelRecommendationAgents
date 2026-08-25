"""Shared state schema for the LangGraph coordinator graph."""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class TripParams(TypedDict, total=False):
    destination: str
    days: int
    travelers: int
    budget_inr: int
    interests: str
    pace: str


class AgentOutput(TypedDict):
    """One specialist agent's answer plus everything needed to trace it."""

    text: str
    sources: list[dict[str, str]]
    flagged_sentences: list[str]


class RouteDecision(TypedDict):
    need_destination_research: bool
    need_budget_planning: bool
    need_itinerary_planning: bool
    reasoning: str


class TripState(TypedDict):
    messages: Annotated[list, add_messages]
    trip_params: TripParams
    route: RouteDecision
    destination_research: AgentOutput | None
    budget_planning: AgentOutput | None
    itinerary_planning: AgentOutput | None
    final_response: str
    all_sources: list[dict[str, str]]
