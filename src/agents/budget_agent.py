"""Budget Planning Agent: cost breakdown built strictly from knowledge-base figures."""
from __future__ import annotations

from functools import lru_cache

from src.agents.base import build_specialist_agent, run_specialist
from src.agents.prompts import BUDGET_PLANNING_PROMPT
from src.graph.state import TripState

AGENT_NAME = "budget_planning"


@lru_cache(maxsize=1)
def _get_agent():
    return build_specialist_agent(BUDGET_PLANNING_PROMPT, name=AGENT_NAME)


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
    if params.get("travelers"):
        lines.append(f"Number of travelers: {params['travelers']}")
    if params.get("budget_inr"):
        lines.append(f"User's total stated budget: INR {params['budget_inr']}")
    lines.append(
        "Build a cost breakdown (stay, food, transport, activities) for this "
        "trip using only figures from the knowledge base's Budget Planning "
        "section (or other cost mentions in the document). Show your "
        "day/traveler multiplication. Flag any category with missing data."
    )
    return "\n".join(lines)


def budget_planning_node(state: TripState) -> dict:
    if not state["route"].get("need_budget_planning"):
        return {}
    output = run_specialist(
        _get_agent(),
        task_message=_build_task(state),
        run_name="budget_planning_agent",
        tags=["agent:budget_planning"],
    )
    return {"budget_planning": output}
