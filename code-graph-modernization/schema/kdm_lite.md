# KDM-lite Schema

A pragmatic property-graph subset of the **OMG Knowledge Discovery Metamodel
(KDM, ISO/IEC 19506)** — the open, vendor-neutral standard at the core of OMG's
Architecture-Driven Modernization (ADM). We use it as an **open-source stand-in
for the TCS MasterCraft inventory schema** so the sandbox learning transfers:
moving to TCS is a *remapping of sources*, not a redesign of the graph.

The machine-readable definition lives in [`kdm_lite.json`](./kdm_lite.json).

## Why KDM (and not a generic code-intelligence schema)

MasterCraft is aligned to OMG ADM, whose foundation is KDM. By modelling our
sandbox graph as a KDM subset we keep the **same node/edge concepts** MasterCraft
emits — programs, data stores, screens, jobs, calls/reads/writes — so the
extractor is the only piece that gets swapped at the TCS boundary.

## Node types

| KDM-lite | KDM origin | MasterCraft counterpart |
|---|---|---|
| `Program` | `code:CompilationUnit` / `code:Module` | Program / Class |
| `Function` | `code:CallableUnit` | Paragraph / Method |
| `Screen` | `ui:Screen` | Screen / Map |
| `Copybook` | `code:RecordType` | Copybook / Shared Structure |
| `DataStore` | `data:RelationalTable` / `code:StorableUnit` | File / Table / Queue |
| `Job` | `platform:DeployedComponent` / build | Batch Job / JCL Step |
| `Handler` | `event:Event` | Transaction / Handler |

## Edge types

| KDM-lite | KDM origin | MasterCraft counterpart |
|---|---|---|
| `CALLS` | `action:Calls` | CALLS |
| `READS_FROM` | `action:Reads` | READS_FROM |
| `WRITES_TO` | `action:Writes` | WRITES_TO |
| `USES_COPYBOOK` | `code:Imports` | USES_COPYBOOK |
| `SCREEN_CALLS_PROGRAM` | `ui:UIFlow` | SCREEN_CALLS_PROGRAM |
| `JOB_RUNS_PROGRAM` | `platform:executes` | JOB_RUNS_PROGRAM |
| `HANDLES_EVENT` | `event:produces/consumes` | HANDLES_EVENT |
| `MAPS_TO_TARGET` | `structure` (conceptual) | MAPS_TO_TARGET |

## Evidence contract

Every node and edge carries `source_artifact`, `source_lines`,
`extraction_method`, and `confidence`. This is non-negotiable: generated
functional specs must trace every assertion back to source, and deterministic
facts must be distinguishable from inferred (lower-confidence) ones. The sandbox
extractor stamps all facts `heuristic-cobol-v0` / `0.7` (screens `0.5`); a
MasterCraft import would stamp them as deterministic.
