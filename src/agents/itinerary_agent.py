"""Itinerary Planning Agent: day-by-day plan, aware of the Budget Agent's output."""
from __future__ import annotations

from functools import lru_cache

from src.agents.base import build_specialist_agent, run_specialist
from src.agents.prompts import ITINERARY_PLANNING_PROMPT
from src.graph.state import TripState

AGENT_NAME = "itinerary_planning"


@lru_cache(maxsize=1)
def _get_agent():
    return build_specialist_agent(ITINERARY_PLANNING_PROMPT, name=AGENT_NAME)


def _last_user_text(state: TripState) -> str:
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", None) == "human":
            return msg.content
    return ""


def _build_task(state: TripState) -> str:
    params = state.get("trip_params", {})
    user_message = _last_user_text(state)

    lines = [f"User request: {user_message}"]
    if params.get("destination"):
        lines.append(f"Destination: {params['destination']}")
    if params.get("days"):
        lines.append(f"Number of days: {params['days']}")
    if params.get("pace"):
        lines.append(f"Preferred pace: {params['pace']}")
    if params.get("interests"):
        lines.append(f"Interests: {params['interests']}")

    budget_output = state.get("budget_planning")
    if budget_output and budget_output.get("text"):
        lines.append(
            "Budget Planning Agent's cost breakdown for this trip "
            "(respect these constraints when choosing activities):\n"
            f"{budget_output['text']}"
        )

    lines.append(
        "Build a day-by-day itinerary using attractions/activities from the "
        "knowledge base, respecting the pace, interests, and budget above."
    )
    return "\n".join(lines)


def itinerary_planning_node(state: TripState) -> dict:
    if not state["route"].get("need_itinerary_planning"):
        return {}
    output = run_specialist(
        _get_agent(),
        task_message=_build_task(state),
        run_name="itinerary_planning_agent",
        tags=["agent:itinerary_planning"],
    )
    return {"itinerary_planning": output}
