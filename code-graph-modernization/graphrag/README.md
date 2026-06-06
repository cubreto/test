# GraphRAG path — LLM-built code knowledge graph

This is the **GraphRAG-builds-the-graph** path (as opposed to the deterministic
parser in `../extractor/`). Microsoft GraphRAG 3.1 reads the legacy source as
text and uses an LLM to extract entities and relationships, detect communities,
and summarize them — then you can query it.

```
sample COBOL -> prepare_input.py -> input/*.txt
   -> graphrag index  (LLM: extract entities+relationships -> communities -> community reports)
   -> graphrag query  (global search over community reports)
```

## What's tuned for code

- `settings.yaml` → `extract_graph.entity_types: [program, function, screen, copybook, datastore, job, handler]`
  (the KDM-lite types — steers the LLM toward a *code* graph).
- `prompts/extract_graph.txt` → rewritten with COBOL examples so the LLM emits
  `CALLS` / `READS_FROM` / `WRITES_TO` / `USES_COPYBOOK` / `JOB_RUNS_PROGRAM` etc.
- `snapshots.graphml: true` → emits `output/graph.graphml` to inspect or load into Neo4j.

## Two ways to run

### A. In the Claude-on-the-web sandbox (Anthropic, completion-only)
Only Anthropic is reachable here, and no embeddings provider is — so `settings.yaml`
uses `model_provider: anthropic` and a workflow list **without** `generate_text_embeddings`.
This still produces the full graph (entities + relationships), communities, and
community reports; **global search** works (it uses reports, not embeddings).

```bash
pip install graphrag
cp .env.example .env        # set GRAPHRAG_API_KEY = your Anthropic key
python prepare_input.py --src ../sample/cobol --out input
graphrag index --root . --skip-validation
graphrag query --root . --method global --query "What are the main functional areas and what data does each touch?"
```

### B. Locally, full pipeline (OpenAI / Together, with embeddings)
Where `api.openai.com` / `api.together.xyz` are reachable:

1. In `settings.yaml`, set `model_provider`/`model` for completion **and** embeddings
   to your provider, and **delete the `workflows:` block** (restores the standard
   pipeline incl. `generate_text_embeddings`).
2. `cp .env.example .env` and set `GRAPHRAG_API_KEY`.
3. `graphrag index --root .` then `graphrag query ... --method local` also works.

## Status / verification

Verified end-to-end up to the LLM boundary: with a dummy key the pipeline runs
`load_input_documents → create_base_text_units → create_final_documents →
extract_graph` and dispatches one Claude extraction call per document — no
host-allowlist or embeddings errors. A valid key completes the run.

## GraphRAG vs the deterministic extractor

GraphRAG's relationships are **untyped** (a description + strength), and entities
are whatever the LLM infers — powerful for discovery, but non-deterministic and
needing validation. The `../extractor/` path produces **typed, evidence-stamped**
edges deterministically (the role MasterCraft plays in TCS). Running both on the
same COBOL sample is the intended comparison: reliability vs. coverage.
