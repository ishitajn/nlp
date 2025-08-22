import faiss, numpy as np, json, os

def build_index(texts, meta, vecs, out_dir):
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, f"{out_dir}/topics.faiss")
    np.save(f"{out_dir}/embeddings.npy", vecs)
    with open(f"{out_dir}/meta.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

def load_index(out_dir):
    index = faiss.read_index(f"{out_dir}/topics.faiss")
    vecs = np.load(f"{out_dir}/embeddings.npy")
    meta = [json.loads(x) for x in open(f"{out_dir}/meta.jsonl", encoding="utf-8")]
    return index, vecs, meta
