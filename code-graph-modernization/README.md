# Code Graph Modernization — Learning Sandbox

A small, end-to-end sandbox for building a **code knowledge graph for
application modernization**: take legacy source, extract a *typed,
evidence-backed* dependency graph, cluster it into **functional areas**, and
generate **grounded functional specifications** from it.

This is the **open-source learning sandbox**. It deliberately uses an open
schema and a sample COBOL estate so we can prove the pipeline *before* moving to
the TCS environment. The schema is a subset of the **OMG KDM** standard that
MasterCraft aligns to, so the move to TCS is a **swap of the extractor**, not a
redesign — see [Transition to TCS / MasterCraft](#transition-to-tcs--mastercraft).

> **Design principle:** structure (programs, calls, reads/writes) is derived
> *deterministically* from a parser/inventory — **not** inferred by an LLM from
> raw code. GraphRAG/LLM techniques sit *on top* of a trustworthy graph to
> cluster, explain, and draft specs, and every claim cites source evidence.

## Architecture

```
Legacy source (COBOL + copybooks + JCL)
        │
        ▼
[1] Extractor                 extractor/         (heuristic; → MasterCraft later)
        │   typed nodes + edges + source-line evidence
        ▼
[2] KDM-lite graph            graph.json         schema/kdm_lite.{md,json}
        │
        ├─▶ [3a] Neo4j load + inspect     loader/    (interactive: Cypher, GDS)
        │
        ▼
[3b] Clustering               analysis/          functional-area detection
        │
        ▼
[4] Grounded spec generation  specs/             evidence-cited markdown specs
        │
        ▼
Functional specs + integration points  →  modernization backlog (next phase)
```

## Layout

| Path | What it is |
|---|---|
| `schema/` | KDM-lite node/edge definitions + KDM/MasterCraft mapping |
| `sample/cobol/` | Sample legacy estate: 4 programs, 3 copybooks, 1 JCL job, 2 SQL tables |
| `extractor/` | Heuristic COBOL/JCL → KDM-lite graph (stdlib only) |
| `analysis/` | Label-propagation clustering (offline stand-in for GDS Louvain) |
| `specs/` | Grounded functional-spec generator (optional Claude narrative) |
| `loader/` | Neo4j loader + Cypher for constraints, inspection, GDS clustering |
| `docker-compose.yml` | Neo4j 5 + APOC + GDS |
| `Makefile` | One-command pipeline |

## Quick start

The core pipeline is **stdlib-only — no database, no pip install needed**:

```bash
make pipeline          # extract → cluster → generate specs
# specs land in specs/out/ ; the graph in graph.json
```

Or step by step:

```bash
make extract           # sample/cobol → graph.json   (typed graph + evidence)
make cluster           # adds functional-area cluster ids to graph.json
make spec              # specs/out/functional_area_*.md
```

### Neo4j (interactive inspection + GDS clustering)

```bash
make up                # start Neo4j  (http://localhost:7474  neo4j/neo4jtest)
pip install -r extractor/requirements.txt
make load              # load graph.json into Neo4j
# then run loader/cypher/{constraints,inspect,cluster}.cypher in the Browser
make down
```

`make load-dry` prints the Cypher without needing a running database.

### Optional: Claude-generated narratives

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...        # CLAUDE_MODEL optional, defaults to a current model
make spec-llm
```

Without a key/SDK, spec generation falls back to a deterministic narrative, so
the sandbox always runs offline.

## What the sample demonstrates

The extractor produces **16 nodes / 19 edges** across the sample estate, and
clustering finds **3 functional areas**:

- **Customer management & daily reporting** — `CUSTMGMT` (screen-driven),
  `DAILYRPT` (batch via `DAILYJOB`), over `CUSTOMER-FILE` / `REPORT-FILE`.
- **Account posting** — `ACCTPOST`, over `ACCOUNT` / `TRANSACTION` (DB2).
- **Parts ordering** — `PARTORDR` (screen-driven), over `PARTS` / `ORDERS`.

Crucially, the generated specs flag **integration points** — e.g. `CUSTMGMT`
`CALLS` `ACCTPOST` and `DAILYRPT` `READS_FROM ACCOUNT`, both crossing into the
account-posting area. Those cross-area edges are the contracts/risks that drive
modernization sequencing.

## Validation posture

Generated specs are explicitly marked **heuristic, requires SME validation**.
Every assertion links to `source_artifact:lines` with an `extraction_method` and
`confidence`, so reviewers can check claims against:
dependency paths (`CALLS`), data flow (`READS_FROM`/`WRITES_TO`), and source
evidence. Low-confidence (e.g. inferred screen) facts are queryable separately.

## Transition to TCS / MasterCraft

Only **one box changes**. Replace the heuristic `extractor/` with a thin adapter
that maps the **MasterCraft inventory export** onto the same KDM-lite node/edge
types (see the mapping table in [`schema/kdm_lite.md`](schema/kdm_lite.md)) and
stamps facts as deterministic (`confidence` ≈ 1.0). Everything downstream —
`graph.json` format, Neo4j loader, clustering, spec generation — is reused
unchanged. That is the whole point of choosing a KDM-aligned schema for the
sandbox.

### Roadmap

| Phase | Goal |
|---|---|
| **0 — Schema alignment** | Confirm MasterCraft export maps onto KDM-lite nodes/edges |
| **1 — This sandbox** | Prove extract → graph → cluster → spec on open COBOL ✔ |
| **2 — GraphRAG enrichment** | Semantic retrieval + richer narratives; broaden node/edge coverage |
| **3 — Target mapping** | `MAPS_TO_TARGET` edges, capability-level specs, modernization backlog |
