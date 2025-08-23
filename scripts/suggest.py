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

from sentence_transformers.util import cos_sim

def suggest(
    convo_texts: list,
    profile_tags: set = set(),
    used_ids: set = set(),
    k: int = 100, # Increased k to get more candidates
    allow_explicit: bool = False,
    index_dir: str = "data/index",
    sentence_model = None,
    focus_topics: list = []
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
    enriched_candidates = []
    if sentence_model and convo_texts:
        last_turn_vec = sentence_model.encode([convo_texts[-1]])
        focus_topics_vecs = sentence_model.encode(focus_topics) if focus_topics else []

        for idx, sim in pairs:
            candidate_vec = vecs[idx:idx+1]

            immediate_relevance = cos_sim(last_turn_vec, candidate_vec).item()

            anti_repetition = 0
            if len(focus_topics_vecs) > 0:
                max_rep_sim = cos_sim(candidate_vec, focus_topics_vecs).max().item()
                if max_rep_sim > 0.8:
                    anti_repetition = -0.3

            enriched_candidates.append({
                "idx": idx, "sim": sim,
                "immediate_relevance": immediate_relevance,
                "anti_repetition": anti_repetition
            })
    else:
        enriched_candidates = [
            {"idx": idx, "sim": sim, "immediate_relevance": 0, "anti_repetition": 0}
            for idx, sim in pairs
        ]

    ranked = rerank(enriched_candidates, meta, convo_texts, used_ids, profile_tags, recent_tags)

    # Filter and format the final output, grouping by category
    categorized_out = {}
    SUGGESTIONS_PER_CATEGORY = 2
    for score, idx in ranked:
        m = meta[idx]
        # --- Filtering ---
        if not allow_explicit and m["explicit_level"] >= 2:
            continue
        if m["id"] in used_ids:
            continue
        if len(m["text"]) > 250: # Filter out long suggestions
            continue

        category = m.get("category", "unknown")

        # --- Grouping ---
        if category not in categorized_out:
            categorized_out[category] = []

        if len(categorized_out[category]) < SUGGESTIONS_PER_CATEGORY:
            categorized_out[category].append({
                "id": m["id"],
                "text": m["text"],
                "category": category,
                "explicit_level": m["explicit_level"],
                "tags": m.get("tags", []),
                "score": round(float(score), 4)
            })

    return categorized_out

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
