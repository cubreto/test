"""KDM-lite graph model.

Node/edge types mirror schema/kdm_lite.json. Every element carries the
evidence contract (source_artifact, source_lines, extraction_method,
confidence) so downstream specs can be traced back to source.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


NODE_TYPES = {
    "Program", "Function", "Screen", "Copybook", "DataStore", "Job", "Handler",
}
EDGE_TYPES = {
    "CALLS", "READS_FROM", "WRITES_TO", "USES_COPYBOOK",
    "SCREEN_CALLS_PROGRAM", "JOB_RUNS_PROGRAM", "HANDLES_EVENT", "MAPS_TO_TARGET",
}


def node_id(node_type: str, name: str) -> str:
    return f"{node_type}:{name.upper()}"


@dataclass
class Node:
    id: str
    type: str
    name: str
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in NODE_TYPES:
            raise ValueError(f"Unknown node type: {self.type}")


@dataclass
class Edge:
    src: str
    dst: str
    type: str
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in EDGE_TYPES:
            raise ValueError(f"Unknown edge type: {self.type}")


class Graph:
    """Deduplicating container. Re-adding a node merges its attrs and
    accumulates source evidence; re-adding an edge accumulates evidence."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[tuple[str, str, str], Edge] = {}

    def add_node(self, node_type: str, name: str, **attrs: Any) -> str:
        nid = node_id(node_type, name)
        if nid in self.nodes:
            self._merge_attrs(self.nodes[nid].attrs, attrs)
        else:
            self.nodes[nid] = Node(id=nid, type=node_type, name=name.upper(), attrs=attrs)
        return nid

    def add_edge(self, src: str, dst: str, edge_type: str, **attrs: Any) -> None:
        key = (src, dst, edge_type)
        if key in self.edges:
            self._merge_attrs(self.edges[key].attrs, attrs)
        else:
            self.edges[key] = Edge(src=src, dst=dst, type=edge_type, attrs=attrs)

    @staticmethod
    def _merge_attrs(target: dict, incoming: dict) -> None:
        for k, v in incoming.items():
            if k == "source_lines" and k in target:
                merged = sorted(set(target[k]) | set(v))
                target[k] = merged
            elif k not in target:
                target[k] = v

    def to_dict(self) -> dict:
        return {
            "schema": "kdm-lite",
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
            "stats": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "node_types": _count_by(self.nodes.values(), lambda n: n.type),
                "edge_types": _count_by(self.edges.values(), lambda e: e.type),
            },
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)


def _count_by(items, key) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        out[key(it)] = out.get(key(it), 0) + 1
    return dict(sorted(out.items()))
