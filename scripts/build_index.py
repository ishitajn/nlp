import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from topicbank.embed import encode
from topicbank.indexer import build_index

def load_jsonl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8")]

def main(in_path="data/curated/topics.jsonl", out_dir="data/index"):
    print(f"Loading curated data from {in_path}...")
    meta = load_jsonl(in_path)
    if not meta:
        print("No data to index. Exiting.")
        return

    texts = [m["text"] for m in meta]

    print(f"Encoding {len(texts)} texts...")
    vecs = encode(texts)

    print(f"Building and writing FAISS index to {out_dir}...")
    build_index(texts, meta, vecs, out_dir)
    print(f"Index built with {len(texts)} items at {out_dir}")

if __name__ == "__main__":
    main()
