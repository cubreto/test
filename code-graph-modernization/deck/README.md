# Deck — Code Graph Modernization (technical review)

Editable PowerPoint for a **technical-lead** audience (Brajesh).

- **Source of truth:** [`build_deck.py`](build_deck.py) — regenerate, don't hand-edit if you want reproducibility.
- **Diagrams:** [`render_diagrams.py`](render_diagrams.py) renders real PNGs into `img/` from `graph.json`
  (the code knowledge graph) and a matplotlib pipeline diagram.
- **Output:** `code-graph-modernization.pptx` (12 slides, 16:9, editable in PowerPoint/Keynote/Google Slides).

## Regenerate

```bash
pip install python-pptx matplotlib networkx
python deck/render_diagrams.py   # -> deck/img/{code_graph,architecture}.png  (from graph.json)
python deck/build_deck.py        # -> deck/code-graph-modernization.pptx
```

> Run clustering first if `graph.json` has no `attrs.cluster`:
> `python analysis/cluster.py --graph graph.json`

## Slide outline

1. Title
2. The problem — modernization needs a graph you can trust (design principle)
3. **Architecture** — rendered pipeline diagram (extract → KDM-lite graph → cluster → specs; branch to Neo4j/GraphRAG/RAG)
4. Schema — KDM-lite as an open stand-in for the MasterCraft inventory + evidence contract
5. The extractor — deterministic structure, every fact stamped (provenance example)
6. Proof on a sample estate — 16 nodes / 19 edges / 3 functional areas + cross-area integration points
7. **The code knowledge graph** — rendered from `graph.json`: nodes coloured by functional area, typed edges, cross-area contracts in bold red
8. Three paths — deterministic graph vs GraphRAG vs simple RAG (trust profiles)
9. GraphRAG path — MS GraphRAG 3.1 tuned for code; sandbox (Anthropic) vs local runs
10. Transition to TCS/MasterCraft — one box changes, everything downstream reused
11. Roadmap — phases 0–3
12. Closing — what this gives us + asks/next steps

> Rendering note: this sandbox's LibreOffice is non-functional, so the PPTX
> wasn't auto-exported to PDF here. To make a PDF locally:
> `soffice --headless --convert-to pdf code-graph-modernization.pptx`
