"""CLI: scan a legacy source tree -> KDM-lite graph.json.

Pass 1 parses copybooks (so record->file resolution works), pass 2 parses
COBOL programs, pass 3 parses JCL. All fragments merge into one graph.

Usage:
    python extract.py --src ../sample/cobol --out ../graph.json
"""
from __future__ import annotations

import argparse
import os

from model import Graph
import cobol_parser as cp


def _merge(dst: Graph, src: Graph) -> None:
    for n in src.nodes.values():
        dst.add_node(n.type, n.name, **n.attrs)
    for e in src.edges.values():
        dst.add_edge(e.src, e.dst, e.type, **e.attrs)


def _walk(src_dir: str):
    for root, _dirs, files in os.walk(src_dir):
        for f in files:
            yield os.path.join(root, f)


def build_graph(src_dir: str) -> Graph:
    paths = list(_walk(src_dir))
    graph = Graph()

    # Pass 1: copybooks -> copybook_name -> record_name
    copybook_records: dict[str, str] = {}
    for p in paths:
        if p.upper().endswith((".CPY", ".CBY")):
            name = os.path.basename(p).rsplit(".", 1)[0].upper()
            with open(p) as fh:
                frag, rec = cp.parse_copybook(p, name, fh.read())
            if rec:
                copybook_records[name] = rec
            _merge(graph, frag)

    # Pass 2: COBOL programs
    for p in paths:
        if p.upper().endswith((".CBL", ".COB", ".CBLLE")):
            with open(p) as fh:
                _merge(graph, cp.parse_cobol(p, fh.read(), copybook_records))

    # Pass 3: JCL
    for p in paths:
        if p.upper().endswith((".JCL", ".JOB")):
            with open(p) as fh:
                _merge(graph, cp.parse_jcl(p, fh.read()))

    return graph


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract a KDM-lite code graph from legacy source.")
    ap.add_argument("--src", required=True, help="Source tree to scan.")
    ap.add_argument("--out", default="graph.json", help="Output graph JSON.")
    args = ap.parse_args()

    graph = build_graph(args.src)
    graph.to_json(args.out)
    stats = graph.to_dict()["stats"]
    print(f"Extracted {stats['node_count']} nodes, {stats['edge_count']} edges -> {args.out}")
    print("  nodes:", stats["node_types"])
    print("  edges:", stats["edge_types"])


if __name__ == "__main__":
    main()
