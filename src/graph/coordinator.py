"""Coordinator nodes: route the request to specialists, then merge their outputs."""
from __future__ import annotations

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from src.agents.base import get_chat_model
from src.agents.prompts import COORDINATOR_MERGE_PROMPT, COORDINATOR_ROUTING_PROMPT
from src.graph.state import RouteDecision, TripState


class _RouteSchema(BaseModel):
    need_destination_research: bool = Field(
        description="True if the user wants general destination info (attractions, "
        "best time to visit, culture, safety, transport, food, etc.)"
    )
    need_budget_planning: bool = Field(
        description="True if the user wants a cost breakdown or is asking about budget."
    )
    need_itinerary_planning: bool = Field(
        description="True if the user wants a day-by-day plan or schedule."
    )
    reasoning: str = Field(description="One sentence explaining the routing decision.")


def _last_user_text(state: TripState) -> str:
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", None) == "human":
            return msg.content
    return ""


def route_node(state: TripState) -> dict:
    model = get_chat_model(temperature=0)
    router = model.with_structured_output(_RouteSchema)

    params = state.get("trip_params", {})
    context_lines = [f"User message: {_last_user_text(state)}"]
    if params:
        context_lines.append(f"Trip parameters provided: {params}")

    decision: _RouteSchema = router.invoke(
        [
            ("system", COORDINATOR_ROUTING_PROMPT),
            ("human", "\n".join(context_lines)),
        ],
        config={"run_name": "coordinator_route", "tags": ["agent:coordinator", "node:route"]},
    )

    route: RouteDecision = {
        "need_destination_research": decision.need_destination_research,
        "need_budget_planning": decision.need_budget_planning,
        "need_itinerary_planning": decision.need_itinerary_planning,
        "reasoning": decision.reasoning,
    }
    return {"route": route}


def merge_node(state: TripState) -> dict:
    sections = []
    all_sources: list[dict[str, str]] = []
    seen_sources = set()

    for label, key in [
        ("Destination Research", "destination_research"),
        ("Budget Planning", "budget_planning"),
        ("Itinerary Planning", "itinerary_planning"),
    ]:
        output = state.get(key)
        if not output:
            continue
        sections.append(f"### {label}\n{output['text']}")
        for src in output["sources"]:
            src_key = (src["source"], src["section"])
            if src_key not in seen_sources:
                seen_sources.add(src_key)
                all_sources.append(src)

    if not sections:
        return {
            "final_response": (
                "I couldn't determine what you're asking for. Could you clarify whether "
                "you want destination info, a budget breakdown, or an itinerary?"
            ),
            "all_sources": [],
        }

    if len(sections) == 1:
        # Nothing to merge/rephrase -- pass the single specialist's answer through
        # verbatim so we don't risk the merge LLM call introducing ungrounded text.
        final_text = sections[0].split("\n", 1)[1]
    else:
        model = get_chat_model(temperature=0.1)
        response: AIMessage = model.invoke(
            [
                ("system", COORDINATOR_MERGE_PROMPT),
                ("human", "\n\n".join(sections)),
            ],
            config={"run_name": "coordinator_merge", "tags": ["agent:coordinator", "node:merge"]},
        )
        final_text = response.content

    return {"final_response": final_text, "all_sources": all_sources}
