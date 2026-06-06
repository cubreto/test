"""Prepare the legacy sample as GraphRAG text input.

Microsoft GraphRAG ingests natural-language *text* documents and uses an LLM to
extract entities/relationships. Source code is not prose, so we wrap each
artifact with a short context header (artifact name + kind) to give the
extraction LLM the framing it needs, then write one .txt document per source
file into graphrag/input/.

Usage:
    python prepare_input.py --src ../sample/cobol --out input
"""
from __future__ import annotations

import argparse
import os

KIND = {
    ".cbl": "COBOL program", ".cob": "COBOL program",
    ".cpy": "COBOL copybook (shared data structure)",
    ".jcl": "JCL batch job", ".job": "JCL batch job",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="input")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    count = 0
    for root, _dirs, files in os.walk(args.src):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in KIND:
                continue
            with open(os.path.join(root, f)) as fh:
                body = fh.read()
            header = (f"Artifact: {f}\n"
                      f"Artifact type: {KIND[ext]}\n"
                      f"---\n")
            out_name = os.path.splitext(f)[0] + ".txt"
            with open(os.path.join(args.out, out_name), "w") as fh:
                fh.write(header + body)
            count += 1
    print(f"Wrote {count} input documents to {args.out}/")


if __name__ == "__main__":
    main()
