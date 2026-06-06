"""Cluster the KDM-lite graph into functional areas.

Offline path (stdlib only): asynchronous label propagation over the
undirected projection of the graph. This is a lightweight stand-in for the
Neo4j GDS Louvain step in loader/cypher/cluster.cypher -- same intent
(community detection -> functional areas), runnable without a database so the
full extract -> cluster -> spec pipeline works in the sandbox.

Writes the cluster id back onto each node (attrs.cluster) in-place.

Usage:
    python cluster.py --graph ../graph.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict


def label_propagation(nodes, edges, seed: int = 7, max_iter: int = 50):
    adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:                      # treat edges as undirected for community detection
        adj[e["src"]].add(e["dst"])
        adj[e["dst"]].add(e["src"])

    labels = {n["id"]: n["id"] for n in nodes}
    order = [n["id"] for n in nodes]
    rng = random.Random(seed)

    for _ in range(max_iter):
        rng.shuffle(order)
        changed = False
        for nid in order:
            if not adj[nid]:
                continue
            counts = Counter(labels[nb] for nb in adj[nid])
            top = max(counts.values())
            # deterministic tie-break: smallest label id among the winners
            winner = min(lbl for lbl, c in counts.items() if c == top)
            if labels[nid] != winner:
                labels[nid] = winner
                changed = True
        if not changed:
            break

    # normalise raw labels to compact integer cluster ids
    remap, cid = {}, 0
    for nid in (n["id"] for n in nodes):
        lbl = labels[nid]
        if lbl not in remap:
            remap[lbl] = cid
            cid += 1
    return {nid: remap[labels[nid]] for nid in labels}


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect functional-area clusters in a KDM-lite graph.")
    ap.add_argument("--graph", required=True)
    args = ap.parse_args()

    with open(args.graph) as fh:
        g = json.load(fh)

    assignment = label_propagation(g["nodes"], g["edges"])
    for n in g["nodes"]:
        n["attrs"]["cluster"] = assignment[n["id"]]

    with open(args.graph, "w") as fh:
        json.dump(g, fh, indent=2)

    by_cluster: dict[int, list[str]] = defaultdict(list)
    for n in g["nodes"]:
        by_cluster[n["attrs"]["cluster"]].append(f"{n['type']}:{n['name']}")

    print(f"Found {len(by_cluster)} functional area(s):")
    for c in sorted(by_cluster):
        print(f"  cluster {c}: {', '.join(sorted(by_cluster[c]))}")


if __name__ == "__main__":
    main()
