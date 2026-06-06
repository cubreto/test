"""A minimal, provider-flexible RAG over the legacy code corpus.

Follows the standard simple-RAG pattern:
    load docs -> chunk -> index (retriever) -> retrieve top-k -> generate answer.

Two retrievers:
  * embed  - real embeddings via an OpenAI-compatible provider (Together AI by
             default; also OpenAI). Cosine similarity over numpy vectors.
  * tfidf  - offline lexical retrieval (scikit-learn). No network/key needed,
             so it runs anywhere (used to demo retrieval inside locked-down envs).

Two generators:
  * an OpenAI-compatible chat model (Together AI default, or OpenAI)
  * Anthropic Claude (the only provider reachable from the Claude-on-the-web sandbox)

Keys come from a .env file or the environment. Nothing is hard-coded.

Examples:
    # offline retrieval only (no LLM call) -- works anywhere:
    python simple_rag.py --query "Which program posts account transactions?" \
        --retriever tfidf --no-generate

    # full RAG with Together AI (run where api.together.xyz is reachable):
    python simple_rag.py --query "..." --provider together

    # full RAG with Claude (works inside the web sandbox):
    python simple_rag.py --query "..." --provider anthropic --retriever tfidf
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import textwrap

# --- provider config -------------------------------------------------------
PROVIDERS = {
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "chat_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "embed_model": "BAAI/bge-large-en-v1.5",
        "key_env": "TOGETHER_API_KEY",
    },
    "openai": {
        "base_url": None,  # default OpenAI endpoint
        "chat_model": "gpt-4o-mini",
        "embed_model": "text-embedding-3-small",
        "key_env": "OPENAI_API_KEY",
    },
    "anthropic": {  # chat only; Anthropic has no embeddings API -> use --retriever tfidf
        "chat_model": os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5"),
        "key_env": "ANTHROPIC_API_KEY",
    },
}


def load_dotenv(path: str = ".env") -> None:
    """Tiny .env loader (no dependency). Does not overwrite existing env vars."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# --- corpus + chunking -----------------------------------------------------
def load_corpus(corpus_dir: str):
    docs = []
    for p in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        with open(p) as fh:
            docs.append((os.path.basename(p), fh.read()))
    return docs


def chunk_docs(docs, size: int = 900, overlap: int = 150):
    """Naive character chunking with overlap; each chunk keeps its source id."""
    chunks = []
    for doc_id, text in docs:
        if len(text) <= size:
            chunks.append((doc_id, text))
            continue
        start = 0
        while start < len(text):
            chunks.append((doc_id, text[start:start + size]))
            start += size - overlap
    return chunks


# --- retrievers ------------------------------------------------------------
class TfidfRetriever:
    def __init__(self, chunks):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks = chunks
        self.vec = TfidfVectorizer(lowercase=True, token_pattern=r"[A-Za-z0-9_-]+")
        self.matrix = self.vec.fit_transform([c[1] for c in chunks])

    def search(self, query: str, k: int):
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [(self.chunks[i][0], self.chunks[i][1], float(sims[i])) for i in order]


class EmbeddingRetriever:
    def __init__(self, chunks, provider: str):
        import numpy as np
        from openai import OpenAI
        cfg = PROVIDERS[provider]
        key = os.environ.get(cfg["key_env"])
        if not key:
            sys.exit(f"Missing {cfg['key_env']} (set it in .env). Or use --retriever tfidf.")
        self.np = np
        self.client = OpenAI(api_key=key, base_url=cfg["base_url"])
        self.model = cfg["embed_model"]
        self.chunks = chunks
        self.matrix = self._embed([c[1] for c in chunks])

    def _embed(self, texts):
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return self.np.array([d.embedding for d in resp.data], dtype="float32")

    def search(self, query: str, k: int):
        q = self._embed([query])[0]
        m = self.matrix
        sims = (m @ q) / (self.np.linalg.norm(m, axis=1) * self.np.linalg.norm(q) + 1e-9)
        order = sims.argsort()[::-1][:k]
        return [(self.chunks[i][0], self.chunks[i][1], float(sims[i])) for i in order]


# --- generation ------------------------------------------------------------
PROMPT = """You are a legacy-modernization assistant. Answer the question using ONLY the \
provided source excerpts from a COBOL codebase. Cite the artifact names you used. \
If the excerpts do not contain the answer, say so.

Question: {query}

Source excerpts:
{context}

Answer:"""


def generate(query: str, hits, provider: str) -> str:
    context = "\n\n".join(f"[{src}]\n{txt}" for src, txt, _ in hits)
    prompt = PROMPT.format(query=query, context=context)
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["key_env"])
    if not key:
        sys.exit(f"Missing {cfg['key_env']} (set it in .env).")

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(model=cfg["chat_model"], max_tokens=600,
                                     messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip()

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=cfg["base_url"])
    resp = client.chat.completions.create(
        model=cfg["chat_model"], max_tokens=600,
        messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple RAG over the legacy code corpus.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--corpus", default="../graphrag/input")
    ap.add_argument("--provider", choices=list(PROVIDERS), default="together")
    ap.add_argument("--retriever", choices=["embed", "tfidf"], default="embed")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--no-generate", action="store_true", help="Retrieve only; skip the LLM call.")
    args = ap.parse_args()

    load_dotenv()
    chunks = chunk_docs(load_corpus(args.corpus))
    if not chunks:
        sys.exit(f"No .txt documents found in {args.corpus}")

    if args.retriever == "tfidf":
        retriever = TfidfRetriever(chunks)
    else:
        retriever = EmbeddingRetriever(chunks, args.provider)

    hits = retriever.search(args.query, args.k)
    print(f"\n# Query\n{args.query}\n")
    print(f"# Retrieved ({args.retriever}, top {args.k})")
    for src, txt, score in hits:
        preview = " ".join(txt.split())[:120]
        print(f"  [{score:.3f}] {src}: {preview}...")

    if args.no_generate:
        return
    print(f"\n# Answer ({args.provider}: {PROVIDERS[args.provider]['chat_model']})\n")
    print(textwrap.fill(generate(args.query, hits, args.provider), width=100))


if __name__ == "__main__":
    main()
