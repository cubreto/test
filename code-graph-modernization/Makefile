# Code Graph Modernization sandbox — one-command pipeline.
# The core path (extract -> cluster -> spec) is stdlib-only and needs no DB.

GRAPH ?= graph.json
SRC   ?= sample/cobol

.PHONY: all extract cluster spec spec-llm pipeline up down load load-dry inspect clean

all: pipeline

# --- offline pipeline (no database required) -------------------------------
extract:
	cd extractor && python3 extract.py --src ../$(SRC) --out ../$(GRAPH)

cluster:
	cd analysis && python3 cluster.py --graph ../$(GRAPH)

spec:
	cd specs && python3 generate_spec.py --graph ../$(GRAPH) --out out

spec-llm:
	cd specs && python3 generate_spec.py --graph ../$(GRAPH) --out out --llm

pipeline: extract cluster spec
	@echo "Pipeline complete. Specs in specs/out/  (graph in $(GRAPH))"

# --- Neo4j interactive path ------------------------------------------------
up:
	docker compose up -d
	@echo "Neo4j Browser: http://localhost:7474  (neo4j / neo4jtest)"

down:
	docker compose down

load-dry:
	cd loader && python3 load_neo4j.py --graph ../$(GRAPH) --dry-run

load:
	cd loader && python3 load_neo4j.py --graph ../$(GRAPH)

clean:
	rm -f $(GRAPH)
	rm -rf specs/out
