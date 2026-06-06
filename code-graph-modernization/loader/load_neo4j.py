"""Load a KDM-lite graph.json into Neo4j.

Every node gets a shared :KdmNode label plus its type label (e.g. :Program),
so constraints and GDS projections can target one label while queries can
still filter by type. Relationship types come from the fixed KDM-lite
whitelist, so interpolating them into Cypher is safe.

Connection via env: NEO4J_URI (bolt://localhost:7687), NEO4J_USER (neo4j),
NEO4J_PASSWORD (neo4jtest).

Usage:
    python load_neo4j.py --graph ../graph.json
    python load_neo4j.py --graph ../graph.json --dry-run   # print Cypher, no DB
"""
from __future__ import annotations

import argparse
import json
import os
import sys

NODE_TYPES = {"Program", "Function", "Screen", "Copybook", "DataStore", "Job", "Handler"}
EDGE_TYPES = {"CALLS", "READS_FROM", "WRITES_TO", "USES_COPYBOOK",
              "SCREEN_CALLS_PROGRAM", "JOB_RUNS_PROGRAM", "HANDLES_EVENT", "MAPS_TO_TARGET"}


def node_cypher(node: dict):
    if node["type"] not in NODE_TYPES:
        raise ValueError(f"Illegal node type {node['type']}")
    props = {"id": node["id"], "name": node["name"], **node["attrs"]}
    q = f"MERGE (n:KdmNode:{node['type']} {{id: $id}}) SET n += $props"
    return q, {"id": node["id"], "props": props}


def edge_cypher(edge: dict):
    if edge["type"] not in EDGE_TYPES:
        raise ValueError(f"Illegal edge type {edge['type']}")
    q = (f"MATCH (a:KdmNode {{id: $src}}), (b:KdmNode {{id: $dst}}) "
         f"MERGE (a)-[r:{edge['type']}]->(b) SET r += $props")
    return q, {"src": edge["src"], "dst": edge["dst"], "props": edge.get("attrs", {})}


def main() -> None:
    ap = argparse.ArgumentParser(description="Load KDM-lite graph into Neo4j.")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--dry-run", action="store_true", help="Print Cypher instead of connecting.")
    args = ap.parse_args()

    with open(args.graph) as fh:
        g = json.load(fh)

    statements = [node_cypher(n) for n in g["nodes"]] + [edge_cypher(e) for e in g["edges"]]

    if args.dry_run:
        for q, params in statements:
            print(q, "  -- ", json.dumps(params))
        print(f"\n-- {len(g['nodes'])} nodes, {len(g['edges'])} edges (dry run, nothing written)")
        return

    try:
        from neo4j import GraphDatabase
    except ImportError:
        sys.exit("neo4j driver not installed. `pip install -r requirements.txt` or use --dry-run.")

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "neo4jtest"))
    with GraphDatabase.driver(uri, auth=auth) as driver:
        with driver.session() as session:
            for q, params in statements:
                session.run(q, **params)
    print(f"Loaded {len(g['nodes'])} nodes, {len(g['edges'])} edges into {uri}")


if __name__ == "__main__":
    main()
