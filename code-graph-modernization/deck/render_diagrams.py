#!/usr/bin/env python3
"""Render real diagrams for the deck from graph.json.

    python deck/render_diagrams.py     # -> deck/img/{code_graph,architecture}.png

Outputs high-res PNGs embedded by build_deck.py.
"""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Patch
from matplotlib.lines import Line2D
import networkx as nx

HERE = os.path.dirname(__file__)
IMG = os.path.join(HERE, "img"); os.makedirs(IMG, exist_ok=True)
ROOT = os.path.dirname(HERE)

# palette
NAVY="#0B2447"; TEAL="#149EC8"; AMBER="#E89B1D"; VIOLET="#7C5CBF"
SLATE="#3A4A5A"; LIGHT="#F2F5F8"; GREEN="#2E8B57"; MUTED="#8A97A3"; RED="#C0392B"
CLUSTER_COL = {0: TEAL, 1: AMBER, 2: VIOLET}
CLUSTER_NAME = {0: "Customer mgmt & reporting", 1: "Parts ordering", 2: "Account posting"}
TYPE_MARKER = {"Program":"s", "DataStore":"o", "Screen":"^", "Copybook":"D", "Job":"h"}
TYPE_SIZE   = {"Program":2000, "DataStore":1350, "Screen":1250, "Copybook":1150, "Job":1300}
EDGE_COL = {"CALLS":RED, "READS_FROM":"#2D6CDF", "WRITES_TO":GREEN,
            "USES_COPYBOOK":MUTED, "SCREEN_CALLS_PROGRAM":VIOLET, "JOB_RUNS_PROGRAM":AMBER}


# =====================================================================
def render_code_graph():
    g = json.load(open(os.path.join(ROOT, "graph.json")))
    nodes, edges = g["nodes"], g["edges"]
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n["id"], ntype=n["type"], label=n["name"],
                   cluster=n["attrs"].get("cluster", 0))
    for e in edges:
        G.add_edge(e["src"], e["dst"], etype=e["type"])

    clusters = sorted({d["cluster"] for _, d in G.nodes(data=True)})
    # separated per-cluster layout: spring within cluster, translate to centroid
    pos = {}
    R = 4.7
    for i, c in enumerate(clusters):
        ang = 2 * math.pi * i / len(clusters) + math.pi / 2
        cx, cy = R * math.cos(ang), R * math.sin(ang)
        members = [n for n, d in G.nodes(data=True) if d["cluster"] == c]
        sub = G.subgraph(members)
        sp = nx.spring_layout(sub, seed=11, k=4.6, iterations=600)
        scale = 1.7 + 0.30 * len(members)        # bigger clusters spread more
        for n, (x, y) in sp.items():
            pos[n] = np.array([cx + x * scale, cy + y * scale])

    fig, ax = plt.subplots(figsize=(13.2, 6.4), dpi=200)
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    # cluster halo blobs
    for c in clusters:
        pts = np.array([pos[n] for n, d in G.nodes(data=True) if d["cluster"] == c])
        ctr = pts.mean(axis=0)
        rad = max(np.linalg.norm(pts - ctr, axis=1).max() + 0.95, 1.3)
        ax.add_patch(Circle(ctr, rad, facecolor=CLUSTER_COL[c], alpha=0.09, zorder=0))
        ax.add_patch(Circle(ctr, rad, facecolor="none", edgecolor=CLUSTER_COL[c],
                            lw=1.6, ls=(0, (6, 4)), alpha=0.5, zorder=0))
        # title placed radially outward from the graph center, clear of cross-edges
        outward = ctr / (np.linalg.norm(ctr) or 1.0)
        tp = ctr + outward * (rad + 0.45)
        ax.text(tp[0], tp[1], CLUSTER_NAME[c].upper(),
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="white", zorder=8,
                bbox=dict(boxstyle="round,pad=0.3", fc=CLUSTER_COL[c], ec="white", lw=1.2))

    # edges
    def is_cross(u, v):
        return G.nodes[u]["cluster"] != G.nodes[v]["cluster"]
    cross_i = 0
    for u, v, d in G.edges(data=True):
        col = EDGE_COL.get(d["etype"], MUTED)
        cross = is_cross(u, v)
        if cross:  # halo + thick = integration contract
            ax.add_patch(FancyArrowPatch(pos[u], pos[v], arrowstyle="-|>",
                mutation_scale=26, lw=6.5, color="black", alpha=0.85,
                connectionstyle="arc3,rad=0.16", zorder=2,
                shrinkA=26, shrinkB=26))
            ax.add_patch(FancyArrowPatch(pos[u], pos[v], arrowstyle="-|>",
                mutation_scale=22, lw=3.2, color=col,
                connectionstyle="arc3,rad=0.16", zorder=3,
                shrinkA=26, shrinkB=26))
            frac = 0.38 if cross_i % 2 == 0 else 0.66      # stagger along the edge
            lab = pos[u] + frac * (pos[v] - pos[u]) + np.array([0.55, 0.0])
            cross_i += 1
            ax.text(lab[0], lab[1], d["etype"], fontsize=9.5, fontweight="bold",
                    color="white", ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.24", fc=RED, ec="white", lw=1.0))
        else:
            ax.add_patch(FancyArrowPatch(pos[u], pos[v], arrowstyle="-|>",
                mutation_scale=15, lw=1.8, color=col, alpha=0.75,
                connectionstyle="arc3,rad=0.08", zorder=2,
                shrinkA=22, shrinkB=22))

    # nodes by type
    for t, mk in TYPE_MARKER.items():
        ns = [n for n, d in G.nodes(data=True) if d["ntype"] == t]
        if not ns: continue
        nx.draw_networkx_nodes(G, pos, nodelist=ns, node_shape=mk,
            node_size=TYPE_SIZE[t], ax=ax,
            node_color=[CLUSTER_COL[G.nodes[n]["cluster"]] for n in ns],
            edgecolors="white", linewidths=2.0)
    # labels — placed just below each node, dark text with white halo for readability
    import matplotlib.patheffects as pe
    for n, d in G.nodes(data=True):
        x, y = pos[n]
        off = 0.40 if d["ntype"] == "Program" else 0.34
        ax.text(x, y - off, d["label"], ha="center", va="top",
                fontsize=8.5, fontweight="bold", color=NAVY, zorder=7,
                path_effects=[pe.withStroke(linewidth=3.2, foreground="white")])

    # legends
    type_handles = [Line2D([0],[0], marker=mk, color="w", markerfacecolor=SLATE,
                    markeredgecolor="white", markersize=13, label=t)
                    for t, mk in TYPE_MARKER.items()]
    edge_handles = [Line2D([0],[0], color=c, lw=3, label=t) for t, c in EDGE_COL.items()]
    leg1 = ax.legend(handles=type_handles, title="Node types", loc="upper left",
                     fontsize=8.5, title_fontsize=9.5, framealpha=0.97,
                     bbox_to_anchor=(-0.01, 1.02))
    leg1.get_title().set_fontweight("bold"); ax.add_artist(leg1)
    leg2 = ax.legend(handles=edge_handles, title="Edge types", loc="upper right",
                     fontsize=8.5, title_fontsize=9.5, framealpha=0.97,
                     bbox_to_anchor=(1.01, 0.90))
    leg2.get_title().set_fontweight("bold"); ax.add_artist(leg2)
    # cross-edge callout
    ax.text(0.5, -0.01,
        "Bold red edges = cross-area integration contracts  —  the dependencies that drive modernization sequencing",
        transform=ax.transAxes, ha="center", va="top", fontsize=10.5,
        fontweight="bold", color=RED)

    ax.set_axis_off(); ax.margins(0.10)
    ax.set_aspect("equal")
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    out = os.path.join(IMG, "code_graph.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white"); plt.close()
    print("wrote", out)


# =====================================================================
def _rbox(ax, x, y, w, h, label, sub, fc, ec, tc="white", fs=12, subfs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=2.2, zorder=3))
    ax.text(x + w/2, y + h*0.66, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=tc, zorder=4)
    if sub:
        ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center",
                fontsize=subfs, color=tc, zorder=4)


def _arrow(ax, p0, p1, col=SLATE, lw=2.6, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=20,
        lw=lw, color=col, connectionstyle=f"arc3,rad={rad}", zorder=2))


def render_architecture():
    fig, ax = plt.subplots(figsize=(13.2, 6.4), dpi=200)
    ax.set_xlim(0, 13.2); ax.set_ylim(0, 6.4); ax.set_axis_off()
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    y = 4.25; h = 1.15; w = 2.05
    stages = [
        ("Legacy source", "COBOL · copybooks\nJCL · SQL", SLATE, SLATE),
        ("[1] Extractor", "deterministic\nparser/inventory", TEAL, TEAL),
        ("[2] KDM-lite\ngraph", "typed nodes+edges\n+ evidence", NAVY, NAVY),
        ("[3] Cluster", "functional-area\ndetection", TEAL, TEAL),
        ("[4] Grounded\nspecs", "evidence-cited\nmarkdown", NAVY, NAVY),
    ]
    xs = []
    x = 0.35
    gap = 0.42
    for i, (lab, sub, fc, ec) in enumerate(stages):
        _rbox(ax, x, y, w, h, lab, sub, fc=fc, ec=ec, fs=12, subfs=8.5)
        xs.append(x)
        if i < len(stages) - 1:
            _arrow(ax, (x + w, y + h/2), (x + w + gap, y + h/2), col=AMBER, lw=3.0)
        x += w + gap

    # source feeds extractor (top label)
    ax.text(xs[0] + w/2, y + h + 0.18, "input", ha="center", fontsize=9,
            style="italic", color=MUTED)

    # evidence ribbon under the pipeline
    ax.add_patch(FancyBboxPatch((0.35, 3.25), x - 0.42 - 0.35 + w, 0.5,
        boxstyle="round,pad=0.02,rounding_size=0.05", fc="#10243f", ec=TEAL, lw=1.4, zorder=1))
    ax.text((0.35 + x - 0.42)/2 + 0.2, 3.5,
            "Evidence contract carried on every node & edge:  source_artifact · source_lines · extraction_method · confidence",
            ha="center", va="center", fontsize=10, color="white", fontweight="bold", zorder=2)

    # branch from KDM-lite graph (stage index 2) downward
    gx = xs[2] + w/2
    _arrow(ax, (gx, y), (gx, 2.55), col=SLATE, lw=2.4)
    ax.text(gx + 0.12, 2.78, "branch at the graph", fontsize=9, color=MUTED, style="italic")

    by = 1.4; bh = 1.05; bw = 3.1
    branches = [
        (0.6,  "Neo4j + GDS", "interactive Cypher,\nLouvain clustering", TEAL),
        (4.6,  "GraphRAG (LLM)", "entities → communities\n→ reports (discovery)", VIOLET),
        (8.6,  "Simple RAG", "retrieve-then-generate\nQ&A baseline", AMBER),
    ]
    for bx, lab, sub, col in branches:
        _rbox(ax, bx, by, bw, bh, lab, sub, fc="white", ec=col, tc=col, fs=12, subfs=8.5)
        _arrow(ax, (gx, 2.5), (bx + bw/2, by + bh), col=col, lw=2.0, rad=0.0)

    # outcome bar
    ax.add_patch(FancyBboxPatch((0.6, 0.35), 12.0, 0.62,
        boxstyle="round,pad=0.02,rounding_size=0.05", fc=GREEN, ec="none", zorder=2))
    ax.text(6.6, 0.66, "Output  →  functional specs + integration points  →  modernization backlog",
            ha="center", va="center", fontsize=12.5, color="white", fontweight="bold", zorder=3)

    # title strip
    ax.text(0.35, 6.05, "Pipeline architecture", fontsize=18, fontweight="bold", color=NAVY)
    ax.text(0.35, 5.68, "Deterministic structure first; LLM/GraphRAG layered on top; evidence end-to-end",
            fontsize=11.5, color=SLATE)
    # TCS note
    ax.text(xs[1] + w/2, y - 0.18, "↑ only this box swaps for MasterCraft", ha="center",
            fontsize=9, color=AMBER, fontweight="bold")

    out = os.path.join(IMG, "architecture.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white"); plt.close()
    print("wrote", out)


if __name__ == "__main__":
    render_code_graph()
    render_architecture()
