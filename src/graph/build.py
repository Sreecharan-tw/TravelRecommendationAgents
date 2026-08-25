"""Assemble the LangGraph coordinator graph.

Flow: route -> destination_research -> budget_planning -> itinerary_planning -> merge

The three specialist nodes run in a fixed linear order (rather than
parallel fan-out) so the Itinerary agent can see the Budget agent's
already-computed cost breakdown, per the requirement that itinerary
planning respects budget constraints. Each specialist node is a no-op
pass-through when the router decided it isn't needed for this turn, so
skipped agents cost no extra LLM call.
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.agents.budget_agent import budget_planning_node
from src.agents.destination_agent import destination_research_node
from src.agents.itinerary_agent import itinerary_planning_node
from src.graph.coordinator import merge_node, route_node
from src.graph.state import TripState


def build_graph():
    graph = StateGraph(TripState)

    graph.add_node("route", route_node)
    graph.add_node("destination_research", destination_research_node)
    graph.add_node("budget_planning", budget_planning_node)
    graph.add_node("itinerary_planning", itinerary_planning_node)
    graph.add_node("merge", merge_node)

    graph.add_edge(START, "route")
    graph.add_edge("route", "destination_research")
    graph.add_edge("destination_research", "budget_planning")
    graph.add_edge("budget_planning", "itinerary_planning")
    graph.add_edge("itinerary_planning", "merge")
    graph.add_edge("merge", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_compiled_graph():
    return build_graph()
