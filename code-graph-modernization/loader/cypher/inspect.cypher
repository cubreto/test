// ---------------------------------------------------------------------------
// Inspection queries for the KDM-lite code graph. Run individually in the
// Neo4j Browser (http://localhost:7474) or via cypher-shell.
// ---------------------------------------------------------------------------

// 1. Inventory: how many of each node type?
MATCH (n:KdmNode)
RETURN labels(n)[1] AS type, count(*) AS count
ORDER BY count DESC;

// 2. Relationship inventory.
MATCH ()-[r]->()
RETURN type(r) AS edge, count(*) AS count
ORDER BY count DESC;

// 3. Impact analysis: everything a program touches (calls, data, copybooks).
MATCH (p:Program {name: 'CUSTMGMT'})-[r]->(t)
RETURN p.name AS program, type(r) AS rel, labels(t)[1] AS target_type, t.name AS target;

// 4. Data lineage: which programs read or write a given store?
MATCH (p)-[r:READS_FROM|WRITES_TO]->(d:DataStore {name: 'ACCOUNT'})
RETURN p.name AS program, type(r) AS access, d.name AS store;

// 5. Shared data stores = coupling hot-spots (touched by >1 program).
MATCH (p:Program)-[:READS_FROM|WRITES_TO]->(d:DataStore)
WITH d, collect(DISTINCT p.name) AS programs
WHERE size(programs) > 1
RETURN d.name AS shared_store, programs;

// 6. Batch reachability: programs reachable from a job (direct + via CALLS).
MATCH path = (j:Job {name: 'DAILYJOB'})-[:JOB_RUNS_PROGRAM]->(:Program)-[:CALLS*0..]->(p:Program)
RETURN DISTINCT p.name AS reachable_program;

// 7. Low-confidence (heuristic) facts that need SME validation first.
MATCH (n:KdmNode)
WHERE n.confidence < 0.7
RETURN labels(n)[1] AS type, n.name, n.confidence, n.source_artifact;
