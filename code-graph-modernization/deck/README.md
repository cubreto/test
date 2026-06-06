# Deck — Code Graph Modernization (technical review)

Editable PowerPoint for a **technical-lead** audience (Brajesh).

- **Source of truth:** [`build_deck.py`](build_deck.py) — regenerate, don't hand-edit if you want reproducibility.
- **Output:** `code-graph-modernization.pptx` (11 slides, 16:9, editable in PowerPoint/Keynote/Google Slides).

## Regenerate

```bash
pip install python-pptx
python deck/build_deck.py        # -> deck/code-graph-modernization.pptx
```

## Slide outline

1. Title
2. The problem — modernization needs a graph you can trust (design principle)
3. Architecture — extract → KDM-lite graph → cluster → grounded specs
4. Schema — KDM-lite as an open stand-in for the MasterCraft inventory + evidence contract
5. The extractor — deterministic structure, every fact stamped (provenance example)
6. Proof on a sample estate — 16 nodes / 19 edges / 3 functional areas + cross-area integration points
7. Three paths — deterministic graph vs GraphRAG vs simple RAG (trust profiles)
8. GraphRAG path — MS GraphRAG 3.1 tuned for code; sandbox (Anthropic) vs local runs
9. Transition to TCS/MasterCraft — one box changes, everything downstream reused
10. Roadmap — phases 0–3
11. Closing — what this gives us + asks/next steps

> Rendering note: this sandbox's LibreOffice is non-functional, so the PPTX
> wasn't auto-exported to PDF here. To make a PDF locally:
> `soffice --headless --convert-to pdf code-graph-modernization.pptx`
