# Simple RAG over the legacy code corpus

A minimal retrieve-then-generate RAG that follows the standard example pattern,
over the same COBOL sample used by the rest of this sandbox. It's the lightweight
baseline to compare against the GraphRAG path (`../graphrag/`).

```
docs (graphrag/input/*.txt) -> chunk -> retrieve top-k -> stuff into prompt -> LLM answer
```

## Why two retrievers / three providers

This repo runs in two very different places, so the script is deliberately flexible:

| | Retriever | Generator | Runs where |
|---|---|---|---|
| Offline demo | `tfidf` (scikit-learn, no key) | `--no-generate` | **anywhere**, incl. locked-down sandboxes |
| Local, with your keys | `embed` | `together` / `openai` | your machine (those API hosts reachable) |
| Inside Claude-on-the-web | `tfidf` | `anthropic` | the web sandbox (only Anthropic is allowlisted) |

> **Network note:** the Claude-on-the-web environment's policy allowlists only a
> few hosts. `api.openai.com` and `api.together.xyz` are **blocked** there, so the
> Together/OpenAI paths must be run **locally**. Anthropic is reachable in-sandbox.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then add your key(s)
```

## Run

```bash
# Retrieval only — works with no key, anywhere:
python simple_rag.py --retriever tfidf --no-generate \
    --query "Which program posts account transactions?"

# Full RAG with Together AI (run locally):
python simple_rag.py --provider together \
    --query "What does the nightly settlement job do, and what data does it touch?"

# Full RAG with Claude (works inside the web sandbox):
python simple_rag.py --provider anthropic --retriever tfidf \
    --query "What does the nightly settlement job do, and what data does it touch?"
```

Flags: `--corpus <dir>` (default `../graphrag/input`), `--k <n>` retrieved chunks,
`--retriever {embed,tfidf}`, `--provider {together,openai,anthropic}`.

## How this differs from the GraphRAG path

This simple RAG retrieves **raw text chunks** by similarity. It has no notion of
programs, calls, or data flow — it can answer "where is X mentioned?" but not
"what is the blast radius if I change ACCOUNT?". The `../graphrag/` path builds a
**typed entity/relationship graph + community summaries**, which is what supports
structural questions and functional-area specs. Running both on the same corpus
is the clearest way to show why a code graph beats plain RAG for modernization.
