// ---------------------------------------------------------------------------
// Functional-area detection with Neo4j GDS (Louvain community detection).
// This is the "real" clustering path; analysis/cluster.py is the offline
// label-propagation stand-in. Requires the graph-data-science plugin
// (included in docker-compose.yml). Run the blocks top to bottom.
// ---------------------------------------------------------------------------

// 1. Project an in-memory, UNDIRECTED graph over all KDM nodes/relationships.
//    (Community detection ignores edge direction; coupling is symmetric.)
CALL gds.graph.project(
  'codegraph',
  'KdmNode',
  {ALL: {type: '*', orientation: 'UNDIRECTED'}}
);

// 2. Run Louvain and write a communityId back onto each node.
CALL gds.louvain.write('codegraph', {writeProperty: 'communityId'})
YIELD communityCount, modularity
RETURN communityCount, modularity;

// 3. Review the functional areas that emerged.
MATCH (n:KdmNode)
RETURN n.communityId AS functional_area,
       collect(labels(n)[1] + ':' + n.name) AS components
ORDER BY functional_area;

// 4. Integration points: edges that cross a community boundary
//    (the contracts/risks for modernization sequencing).
MATCH (a:KdmNode)-[r]->(b:KdmNode)
WHERE a.communityId <> b.communityId
RETURN a.communityId AS from_area, type(r) AS edge,
       a.name AS source, b.name AS target, b.communityId AS to_area
ORDER BY from_area;

// 5. Clean up the in-memory projection when done.
CALL gds.graph.drop('codegraph');
