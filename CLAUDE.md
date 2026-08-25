# Travel Recommendation Agents

Multi-agent travel recommendation system: LangChain/LangGraph agents doing
agentic RAG over `/knowledge` markdown files, backed by Pinecone + Gemini,
traced in LangSmith, served through a Streamlit chat UI. Full architecture
and setup instructions are in [README.md](README.md) — this file is about
*why* things are built the way they are, for whoever (human or Claude)
touches this code next.

## Tech stack (fixed, do not substitute)

LangChain + LangGraph · Google Gemini (`gemini-2.5-flash` chat,
`gemini-embedding-001` embeddings) via `langchain-google-genai` · Pinecone
via `langchain-pinecone` · LangSmith tracing · Streamlit UI. Python 3.12,
managed with `uv` (see below for why not the system Python).

## Commands

```bash
make setup      # create .venv (Python 3.12 via uv) + install requirements.txt
make ingest     # embed + upsert /knowledge into Pinecone (idempotent)
make reingest   # same, plus prunes vectors for removed/changed sections
make app        # run the Streamlit chat UI
make test       # pytest tests/
make graph      # print the compiled graph as Mermaid + save docs/graph.png
make clean      # remove __pycache__ / .pytest_cache
```

## Key decisions and why

**Python pinned to 3.12 via `uv`, not the system Python.** The machine's
only system Python was 3.14 (very new at the time this was built) — the
LangChain/Pinecone/Streamlit ecosystem is best-tested on 3.11–3.12. `uv venv
--python 3.12` sidesteps compatibility risk without touching system Python.

**Agentic RAG, not naive RAG.** `search_travel_knowledge`
(`src/tools/retriever_tool.py`) is a LangChain `@tool`, and each specialist
is its own `create_react_agent` (`langgraph.prebuilt`) bound to that tool —
the agent decides itself whether/how many times to call it and with what
query, rather than a fixed chunk of context spliced into a prompt ahead of
time. Note the naming trap: `langchain.agents.create_react_agent` is a
*different*, older function (returns a `Runnable` for `AgentExecutor`, not a
graph) — always import `create_react_agent` from `langgraph.prebuilt` here.

**Specialist agents are built lazily**, not at module import time
(`functools.lru_cache` around a `_get_agent()` per agent module). Building
a `ChatGoogleGenerativeAI` client eagerly at import time would mean
importing `src.agents.destination_agent` requires a live `GOOGLE_API_KEY`
even just to test the "not routed this turn" skip path. Lazy construction
keeps that path (and `tests/test_agent_node.py`) credential-free.

**Graph is a fixed linear chain**
(`route → destination_research → budget_planning → itinerary_planning →
merge`), not a parallel fan-out, specifically so the Itinerary agent can
read the Budget agent's already-computed cost breakdown as context
(`src/agents/itinerary_agent.py`). A specialist not selected by the router
is a true no-op pass-through — zero LLM calls — controlled by
`state["route"]` flags from a structured-output routing call in
`src/graph/coordinator.py`.

**Merge node never adds facts.** If only one specialist ran, its text is
passed through verbatim — no LLM rephrasing step that could introduce
something ungrounded. If multiple ran, an LLM merge call combines them but
is explicitly instructed only to reorganize/rephrase, never add new
content.

**Grounding check is lexical, not a second LLM call**
(`src/tools/grounding.py`) — cheap word-overlap between the answer and the
actually-retrieved chunk text, flagging low-overlap sentences for the UI.
It's a hint for the user, not a hard gate; deliberately biased toward
false-positives over missing a real hallucination.

**`ToolMessage.content` from a `list[dict]`-returning tool gets
JSON-serialized to a string** by LangGraph's `ToolNode` before it lands in
the message history (confirmed by reading `langgraph/prebuilt/tool_node.py`
directly — `msg_content_output()` does `json.dumps` on anything that isn't
already a string or content-block list). `run_specialist()`
(`src/agents/base.py`) has to `json.loads()` it back out when reconstructing
sources — don't assume `ToolMessage.content` is already a list.

**Deterministic chunk ids for idempotent upserts.** Each chunk's Pinecone id
is `sha256(source_file + section + content)` (`src/ingestion/loader.py`).
Re-running `make ingest` after editing a markdown file updates that chunk in
place instead of duplicating it; `make reingest` additionally prunes ids
that no longer correspond to any current chunk.

**Destination name isn't derived from the filename alone.**
`andamanandnicobar.md` can't be reliably split into words. The loader
first looks for a `# Destination` section or a
`## <Name> Destination Knowledge Document` title line inside the file
content itself, and only falls back to title-casing the filename
(`src/ingestion/loader.py:_destination_name`). The four existing knowledge
files are inconsistent about which pattern (if either) they use.

**Embedding model: `gemini-embedding-001`, not `text-embedding-004`.**
`text-embedding-004` is being deprecated and returned a 404 against the
project's API key. `gemini-embedding-001` natively outputs 3072 dims, so
`src/ingestion/vectorstore.py` wraps it in `_FixedDimensionEmbeddings` to
force `output_dimensionality=768` on every call (matching the Pinecone
index's dimension) — `langchain_pinecone` calls `embed_documents`/
`embed_query` with no way to pass that kwarg through otherwise.

**Compiled graph is cached per-process**, not per-request.
`get_compiled_graph()` (`src/graph/build.py`) is `@lru_cache(maxsize=1)`.
Streamlit reruns the whole script on every interaction, but that's a script
rerun inside the same process — `src.graph.build` stays in `sys.modules`,
so `StateGraph(...).compile()` only actually runs once per running app, not
once per chat message.

## Environment / secrets notes

- `.env` is git-ignored; `.env.example` documents every variable actually
  read (see `src/config.py`).
- The original `.env` had `INDEX_NAME="medium-blog-index"` (a different,
  1024-dim project — confirmed via `pinecone.list_indexes()`) and
  `LANGSMITH_PROJECT="LangchainTest"` (a generic shared project). Both were
  changed, with the user's confirmation, to a dedicated
  `travel-recommendation-agents` index/project so this app's data and
  traces stay isolated from other work.
- `GOOGLE_API_KEY` may be exported in the shell profile rather than sitting
  in `.env` on some machines this was tested on — `src/config.py` reads via
  `os.getenv` after `load_dotenv()`, and `load_dotenv()` does not override
  an already-exported shell variable. If `GOOGLE_API_KEY` looks unset when
  running a command directly (not through an interactive shell), check
  `.env` first, then the shell profile.
- Never print `.env` contents (even masked) further than confirming a key
  is *present*; when debugging env issues, check length/prefix, not the
  value.

## Testing approach

Tests avoid needing live credentials: `tests/test_retriever.py` exercises
the pure chunking logic and monkeypatches the vector store for the tool
test; `tests/test_agent_node.py` exercises the no-LLM-call "not routed this
turn" skip path and the merge node's pure combination logic rather than
mocking Gemini's structured-output wire format; `tests/test_grounding.py`
tests the lexical overlap heuristic directly. The LLM-calling paths
themselves are verified by actually running `make ingest` + a live graph
invocation + the Streamlit UI, not by mocking the model.

## Known limitations

See the "Known limitations" section in [README.md](README.md) (grounding
check is lexical not semantic, no cross-session conversation persistence,
single routing LLM call per turn, no cross-destination budget comparison,
serverless-only Pinecone index, uneven knowledge base coverage across
destinations).
