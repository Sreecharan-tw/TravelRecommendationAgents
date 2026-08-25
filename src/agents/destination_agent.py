"""Destination Research Agent: general destination info from the knowledge base."""
from __future__ import annotations

from functools import lru_cache

from src.agents.base import build_specialist_agent, run_specialist
from src.agents.prompts import DESTINATION_RESEARCH_PROMPT
from src.graph.state import TripState

AGENT_NAME = "destination_research"


@lru_cache(maxsize=1)
def _get_agent():
    # Built lazily (not at import time) so importing this module -- e.g. in
    # tests that only exercise the "not needed this turn" skip path -- does
    # not require a live GOOGLE_API_KEY.
    return build_specialist_agent(DESTINATION_RESEARCH_PROMPT, name=AGENT_NAME)


def _build_task(state: TripState) -> str:
    params = state.get("trip_params", {})
    user_message = _last_user_text(state)

    lines = [f"User request: {user_message}"]
    if params.get("destination"):
        lines.append(f"Destination of interest: {params['destination']}")
    if params.get("interests"):
        lines.append(f"User interests: {params['interests']}")
    lines.append(
        "Research this destination (overview, best time to visit, attractions, "
        "hidden gems, culture, safety, transport, food -- whatever is relevant "
        "to the request) using the knowledge base."
    )
    return "\n".join(lines)


def _last_user_text(state: TripState) -> str:
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", None) == "human":
            return msg.content
    return ""


def destination_research_node(state: TripState) -> dict:
    if not state["route"].get("need_destination_research"):
        return {}
    output = run_specialist(
        _get_agent(),
        task_message=_build_task(state),
        run_name="destination_research_agent",
        tags=["agent:destination_research"],
    )
    return {"destination_research": output}
