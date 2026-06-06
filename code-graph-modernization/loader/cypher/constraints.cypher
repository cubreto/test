// Run once before loading. Ensures node identity and speeds up MERGE/lookup.
CREATE CONSTRAINT kdm_node_id IF NOT EXISTS
FOR (n:KdmNode) REQUIRE n.id IS UNIQUE;

CREATE INDEX kdm_node_name IF NOT EXISTS
FOR (n:KdmNode) ON (n.name);
