import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import numpy as np
from topicbank.embed import encode
from topicbank.indexer import load_index
from topicbank.rank import rerank
from topicbank.transition import transition_score
from topicbank.tagging import tag_topics

def suggest(
    convo_texts: list,
    profile_tags: set = set(),
    used_ids: set = set(),
    k: int = 50,
    allow_explicit: bool = False,
    index_dir: str = "data/index"
):
    """
    Retrieves and re-ranks topic suggestions based on conversation context.
    """
    try:
        index, vecs, meta = load_index(index_dir)
    except FileNotFoundError:
        print(f"Error: Index not found at {index_dir}. Please run scripts/build_index.py first.")
        return []

    if not convo_texts:
        # Handle empty conversation: maybe return popular/generic starters
        # For now, returning empty list
        return []

    # Use last 3 messages for query vector
    q = " ".join(convo_texts[-3:])
    qvec = encode([q])

    # FAISS search
    D, I = index.search(qvec, k)
    # Guard against no results
    if I.size == 0:
        return []
    pairs = list(zip(I[0], D[0]))

    # Get tags from recent conversation for re-ranking
    recent_tags = set()
    for t in convo_texts[-5:]:
        recent_tags.update(tag_topics(t))

    # Re-rank the retrieved candidates
    ranked = rerank(pairs, meta, convo_texts, used_ids, profile_tags, recent_tags)

    # Filter and format the final output
    out = []
    for score, idx in ranked:
        m = meta[idx]
        if not allow_explicit and m["explicit_level"] >= 2:
            continue
        if m["id"] in used_ids:
            continue

        out.append({
            "id": m["id"],
            "text": m["text"],
            "category": m["category"],
            "explicit_level": m["explicit_level"],
            "tags": m.get("tags", []),
            "score": round(float(score), 4)
        })
        # Return top 2 suggestions
        if len(out) == 2:
            break

    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--convo", nargs="+", required=True, help="Recent messages (last 3–5 lines)")
    ap.add_argument("--profile_tags", nargs="*", default=[], help="Anchors like: fitness romance career")
    ap.add_argument("--used_jsonl", default=None, help="Path to a JSONL file of used topic IDs")
    ap.add_argument("--allow_explicit", action="store_true")
    args = ap.parse_args()

    used_ids = set()
    if args.used_jsonl:
        try:
            used_ids = {json.loads(x)["id"] for x in open(args.used_jsonl, encoding="utf-8")}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load used IDs from {args.used_jsonl}. Reason: {e}")


    res = suggest(
        convo_texts=args.convo,
        profile_tags=set(args.profile_tags),
        used_ids=used_ids,
        allow_explicit=args.allow_explicit
    )

    print(json.dumps(res, ensure_ascii=False, indent=2))
