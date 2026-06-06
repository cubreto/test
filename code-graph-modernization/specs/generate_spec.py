"""Generate grounded functional specifications from the clustered KDM-lite graph.

For each functional area (cluster) it emits a markdown spec that is:
  * structured   - components, entry points, data flow, integration points;
  * grounded     - every assertion cites source_artifact:lines + confidence;
  * validatable  - integration points (cross-cluster edges) and low-confidence
                   (heuristic) facts are flagged for SME review.

An optional LLM narrative (Anthropic) can be layered on top with --llm; without
it a deterministic narrative is produced so the sandbox runs fully offline.

Usage:
    python generate_spec.py --graph ../graph.json --out out [--llm]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


def _short(path: str) -> str:
    return path.split("/")[-1]


def _evidence(attrs: dict) -> str:
    lines = attrs.get("source_lines", [])
    ln = ",".join(str(x) for x in lines) if lines else "?"
    return f"{_short(attrs.get('source_artifact', '?'))}:{ln}"


def build_specs(graph: dict):
    nodes = {n["id"]: n for n in graph["nodes"]}
    cluster_of = {n["id"]: n["attrs"].get("cluster", 0) for n in graph["nodes"]}
    clusters: dict[int, list[str]] = defaultdict(list)
    for nid, c in cluster_of.items():
        clusters[c].append(nid)

    specs = {}
    for c, nids in sorted(clusters.items()):
        members = set(nids)
        by_type: dict[str, list[str]] = defaultdict(list)
        for nid in nids:
            by_type[nodes[nid]["type"]].append(nid)

        entry, internal_flow, integration = [], [], []
        for e in graph["edges"]:
            s_in, d_in = e["src"] in members, e["dst"] in members
            if not (s_in or d_in):
                continue
            if e["type"] in ("SCREEN_CALLS_PROGRAM", "JOB_RUNS_PROGRAM") and d_in:
                entry.append(e)
            elif e["type"] in ("READS_FROM", "WRITES_TO") and s_in and d_in:
                internal_flow.append(e)
            elif s_in != d_in:                      # crosses a cluster boundary
                integration.append(e)
        specs[c] = {
            "members": members, "by_type": by_type,
            "entry": entry, "internal_flow": internal_flow, "integration": integration,
        }
    return nodes, cluster_of, specs


def render_markdown(c: int, spec: dict, nodes: dict, cluster_of: dict, narrative: str) -> str:
    name = lambda nid: nodes[nid]["name"]
    out: list[str] = []
    out.append(f"# Functional Area {c} — Draft Functional Specification\n")
    out.append("> Auto-generated from the KDM-lite code graph. **Heuristic extraction — "
               "requires SME validation.** Every claim links to source evidence.\n")

    out.append(f"\n## Narrative\n\n{narrative}\n")

    out.append("\n## Components\n")
    for t in ("Program", "Screen", "Job", "DataStore", "Copybook", "Handler", "Function"):
        ids = spec["by_type"].get(t)
        if ids:
            out.append(f"- **{t}**: " + ", ".join(sorted(name(i) for i in ids)))

    out.append("\n\n## Entry points\n")
    if spec["entry"]:
        for e in spec["entry"]:
            verb = "screen drives" if e["type"] == "SCREEN_CALLS_PROGRAM" else "batch job runs"
            out.append(f"- {name(e['src'])} *{verb}* {name(e['dst'])}  "
                       f"(`{_evidence(e['attrs'])}`)")
    else:
        out.append("- _none detected_")

    out.append("\n\n## Internal data flow\n")
    if spec["internal_flow"]:
        for e in spec["internal_flow"]:
            arrow = "reads" if e["type"] == "READS_FROM" else "writes"
            out.append(f"- {name(e['src'])} {arrow} `{name(e['dst'])}`  "
                       f"(`{_evidence(e['attrs'])}`)")
    else:
        out.append("- _none detected_")

    out.append("\n\n## Integration points (cross-area dependencies)\n")
    if spec["integration"]:
        out.append("These edges leave this functional area — the key risks/contracts for "
                   "modernization sequencing:\n")
        for e in spec["integration"]:
            other = e["dst"] if e["src"] in spec["members"] else e["src"]
            out.append(f"- `{e['type']}` {name(e['src'])} → {name(e['dst'])} "
                       f"(other area: cluster {cluster_of[other]})  (`{_evidence(e['attrs'])}`)")
    else:
        out.append("- _self-contained: no cross-area dependencies_")

    out.append("\n\n## Evidence & confidence\n")
    out.append("| Element | Source | Method | Confidence |")
    out.append("|---|---|---|---|")
    seen = set()
    for nid in sorted(spec["members"]):
        a = nodes[nid]["attrs"]
        key = nodes[nid]["type"] + nodes[nid]["name"]
        if key in seen:
            continue
        seen.add(key)
        out.append(f"| {nodes[nid]['type']}:{nodes[nid]['name']} | `{_evidence(a)}` | "
                   f"{a.get('extraction_method', '?')} | {a.get('confidence', '?')} |")
    return "\n".join(out) + "\n"


def deterministic_narrative(c: int, spec: dict, nodes: dict) -> str:
    progs = [nodes[i]["name"] for i in spec["by_type"].get("Program", [])]
    stores = [nodes[i]["name"] for i in spec["by_type"].get("DataStore", [])]
    drivers = []
    for e in spec["entry"]:
        drivers.append(nodes[e["src"]]["name"])
    driver_txt = f" It is entered via {', '.join(sorted(set(drivers)))}." if drivers else ""
    integ = len(spec["integration"])
    integ_txt = (f" It has {integ} cross-area dependency/dependencies that must be treated as "
                 f"integration contracts during modernization.") if integ else \
                " It is self-contained with no cross-area dependencies."
    return (f"This functional area is implemented by {', '.join(sorted(progs)) or 'no programs'} "
            f"operating over the data stores {', '.join(sorted(stores)) or 'none'}."
            f"{driver_txt}{integ_txt}")


def llm_narrative(c: int, spec: dict, nodes: dict) -> str:
    """Optional Claude-generated narrative. Falls back to deterministic on any error."""
    try:
        import anthropic  # imported lazily so offline runs need no dependency
    except ImportError:
        return deterministic_narrative(c, spec, nodes)
    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
    facts = {
        "programs": [nodes[i]["name"] for i in spec["by_type"].get("Program", [])],
        "data_stores": [nodes[i]["name"] for i in spec["by_type"].get("DataStore", [])],
        "entry_points": [f"{nodes[e['src']]['name']}->{nodes[e['dst']]['name']}" for e in spec["entry"]],
        "integration": [f"{e['type']} {nodes[e['src']]['name']}->{nodes[e['dst']]['name']}"
                        for e in spec["integration"]],
    }
    prompt = ("You are documenting a legacy functional area for a modernization effort. "
              "Using ONLY these graph facts, write a 3-4 sentence functional summary. "
              "Do not invent behaviour not implied by the facts.\n\n"
              f"{json.dumps(facts, indent=2)}")
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model, max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:  # network/credentials/etc -> stay offline-safe
        return deterministic_narrative(c, spec, nodes) + f"\n\n_(LLM narrative skipped: {exc})_"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate grounded functional specs from the code graph.")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--llm", action="store_true", help="Layer a Claude-generated narrative on top.")
    args = ap.parse_args()

    with open(args.graph) as fh:
        graph = json.load(fh)
    nodes, cluster_of, specs = build_specs(graph)

    os.makedirs(args.out, exist_ok=True)
    index = ["# Functional Specification Index\n",
             f"Generated from {graph['stats']['node_count']} nodes / "
             f"{graph['stats']['edge_count']} edges across {len(specs)} functional area(s).\n"]
    for c, spec in specs.items():
        narrative = (llm_narrative if args.llm else lambda *a: deterministic_narrative(*a))(c, spec, nodes)
        md = render_markdown(c, spec, nodes, cluster_of, narrative)
        fname = f"functional_area_{c}.md"
        with open(os.path.join(args.out, fname), "w") as fh:
            fh.write(md)
        progs = ", ".join(sorted(nodes[i]["name"] for i in spec["by_type"].get("Program", [])))
        index.append(f"- [Functional Area {c}]({fname}) — {progs}")
        print(f"wrote {os.path.join(args.out, fname)}")
    with open(os.path.join(args.out, "index.md"), "w") as fh:
        fh.write("\n".join(index) + "\n")


if __name__ == "__main__":
    main()
