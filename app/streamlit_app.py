"""Streamlit chat UI for the multi-agent travel recommendation system."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.graph.build import get_compiled_graph  # noqa: E402

settings.configure_langsmith()

st.set_page_config(page_title="Travel Recommendation Agents", page_icon="\U0001f9f3", layout="wide")

AGENT_LABELS = {
    "destination_research": "\U0001f30d Destination Research Agent",
    "budget_planning": "\U0001f4b0 Budget Planning Agent",
    "itinerary_planning": "\U0001f5fa️ Itinerary Planning Agent",
}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content, extras|None)

if "graph_messages" not in st.session_state:
    st.session_state.graph_messages = []  # LangChain messages fed into the graph state


def render_sources_and_grounding(result: dict) -> None:
    used_agents = [
        key for key in ("destination_research", "budget_planning", "itinerary_planning")
        if result.get(key)
    ]
    if not used_agents:
        return

    labels = " · ".join(AGENT_LABELS[a] for a in used_agents)
    st.caption(f"Handled by: {labels}")

    with st.expander("Sources & grounding check", expanded=False):
        for key in used_agents:
            output = result[key]
            st.markdown(f"**{AGENT_LABELS[key]}**")
            if output["sources"]:
                for src in output["sources"]:
                    st.markdown(
                        f"- Retrieved from: `{src['source']}` › *{src['section']}* "
                        f"({src['destination']})"
                    )
            else:
                st.markdown("- No knowledge-base chunks retrieved.")

            if output["flagged_sentences"]:
                st.markdown(
                    ":warning: **Possibly ungrounded statements** "
                    "(low overlap with retrieved text -- double-check these):"
                )
                for sentence in output["flagged_sentences"]:
                    st.markdown(f"  - _{sentence}_")
            st.markdown("---")

        route = result.get("route", {})
        if route.get("reasoning"):
            st.caption(f"Routing reasoning: {route['reasoning']}")


def run_turn(user_text: str, trip_params: dict) -> dict:
    st.session_state.graph_messages.append(HumanMessage(content=user_text))

    graph = get_compiled_graph()
    result = graph.invoke(
        {
            "messages": st.session_state.graph_messages,
            "trip_params": {k: v for k, v in trip_params.items() if v},
        },
        config={"run_name": "coordinator_graph", "tags": ["app:streamlit"]},
    )

    st.session_state.graph_messages.append(AIMessage(content=result["final_response"]))
    return result


with st.sidebar:
    st.header("Trip Parameters")
    destination = st.text_input(
        "Destination", placeholder="e.g. Kashmir, Kerala, Meghalaya, Andaman and Nicobar"
    )
    days = st.number_input("Number of days", min_value=1, max_value=60, value=5, step=1)
    travelers = st.number_input("Number of travelers", min_value=1, max_value=20, value=2, step=1)
    budget_inr = st.number_input(
        "Total budget (INR, optional)", min_value=0, value=0, step=1000,
        help="Leave at 0 if you don't want to specify a budget cap.",
    )
    interests = st.text_input(
        "Interests", placeholder="e.g. trekking, food, photography, wildlife"
    )
    pace = st.selectbox("Pace", ["No preference", "Relaxed", "Moderate", "Packed"])

    st.divider()
    st.caption(f"LangSmith project: `{settings.langsmith_project}`")
    st.caption(f"Chat model: `{settings.chat_model}`")

    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.graph_messages = []
        st.rerun()

st.title("Travel Recommendation Agents")
st.caption(
    "Answers are grounded strictly in the markdown knowledge base under /knowledge. "
    "If something isn't in there, the agents will say so instead of guessing."
)

for role, content, extras in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)
        if extras:
            render_sources_and_grounding(extras)

user_input = st.chat_input("Ask about a destination, budget, or itinerary...")

if user_input:
    st.session_state.chat_history.append(("user", user_input, None))
    with st.chat_message("user"):
        st.markdown(user_input)

    trip_params = {
        "destination": destination,
        "days": int(days),
        "travelers": int(travelers),
        "budget_inr": int(budget_inr) if budget_inr else None,
        "interests": interests,
        "pace": pace if pace != "No preference" else None,
    }

    with st.chat_message("assistant"):
        with st.spinner("Consulting the specialist agents..."):
            try:
                result = run_turn(user_input, trip_params)
            except Exception as exc:  # surfaced directly -- no silent failure
                st.error(f"Something went wrong: {exc}")
                st.stop()
        st.markdown(result["final_response"])
        render_sources_and_grounding(result)

    st.session_state.chat_history.append(("assistant", result["final_response"], result))
