.PHONY: setup ingest reingest app test graph clean

VENV := .venv
PYTHON := $(VENV)/bin/python

# Create the project virtual environment and install pinned dependencies.
setup:
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(PYTHON) -r requirements.txt

# Build the Pinecone index from /knowledge (idempotent -- safe to re-run).
ingest:
	$(PYTHON) -m src.ingestion.ingest

# Same as ingest, but also deletes vectors for sections that were removed
# or changed since the last run (use after editing/deleting knowledge files).
reingest:
	$(PYTHON) -m src.ingestion.ingest --prune

# Run the Streamlit chat app.
app:
	$(PYTHON) -m streamlit run app/streamlit_app.py

# Run the test suite.
test:
	$(PYTHON) -m pytest tests/ -v

# Print the compiled LangGraph's structure as Mermaid syntax (no credentials
# needed) and also save a PNG to docs/graph.png.
graph:
	$(PYTHON) -m src.graph.visualize --png docs/graph.png

# Remove caches and bytecode (does not touch .venv or the Pinecone index).
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
