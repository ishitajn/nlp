# In scoring_engine.py
import re
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from embedder import embedder_service
from semantic_concepts import CONCEPT_EMBEDDINGS

# --- Constants and Lexicons for Categorization ---

AVOID_KEYWORDS = [r'\b(politics|religion|government|election|vote|biden|trump|conservative|liberal)\b']
SENSITIVE_KEYWORDS = [r'\b(autism|adhd|ocd|bpd|trauma|disability|mental health|therapy|depression|anxiety|grief|loss)\b']
FETISH_KEYWORDS = [r'\b(kink|fetish|bdsm|dom|sub|daddy|kitten|foot|feet|choke|spank)\b']

# Category priority for conflict resolution (higher number = higher priority)
CATEGORY_PRIORITY = {
    "sensitive": 5,
    "fetish": 4,
    "sexual": 3,
    "focus": 2,
    "avoid": 1,
    "neutral": 0
}

def _calculate_keyword_centrality(keywords: List[str], cluster_centroid: np.ndarray) -> float:
    """Calculates the average cosine similarity of keywords to the cluster centroid."""
    if not keywords or cluster_centroid is None:
        return 0.0
    keyword_embeddings = embedder_service.encode_cached(keywords)
    if not keyword_embeddings.any():
        return 0.0
    similarities = cosine_similarity(keyword_embeddings, cluster_centroid.reshape(1, -1))
    return float(np.mean(similarities)) if similarities.any() else 0.0

def score_and_categorize_topics(
    topic_clusters: List[Dict[str, Any]],
    context: Dict[str, Any]
) -> Dict[str, List[str]]:
    """
    Scores and categorizes topic clusters based on salience, keywords, and semantic similarity.
    """
    final_categorized_topics = defaultdict(list)
    if not topic_clusters:
        return dict(final_categorized_topics)

    max_cluster_size = max(len(c.get("messages", [])) for c in topic_clusters) or 1
    keyword_embedding_cache = {}

    for cluster in topic_clusters:
        topic_name = cluster.get("canonical_name", "Unknown Topic")
        keywords_str = " ".join(cluster.get("keywords", []))

        # --- Score Calculation ---
        scores = defaultdict(float)

        # Salience Score (used for focus)
        cluster_size_normalized = len(cluster.get("messages", [])) / max_cluster_size
        keyword_centrality = _calculate_keyword_centrality(cluster.get("keywords", []), cluster.get("centroid"))
        recency_rank = context.get("topic_recency", {}).get(topic_name, 10)
        recency_weight = 1 / recency_rank
        scores['focus'] = (cluster_size_normalized + keyword_centrality + recency_weight)

        # Keyword-based scores
        if any(re.search(p, keywords_str, re.IGNORECASE) for p in AVOID_KEYWORDS):
            scores['avoid'] = 1.0
        if any(re.search(p, keywords_str, re.IGNORECASE) for p in SENSITIVE_KEYWORDS):
            scores['sensitive'] = 1.0
        if any(re.search(p, keywords_str, re.IGNORECASE) for p in FETISH_KEYWORDS):
            scores['fetish'] = 1.0

        # Semantic similarity score for 'sexual'
        if keywords_str:
            if keywords_str not in keyword_embedding_cache:
                keyword_embedding_cache[keywords_str] = embedder_service.encode_cached([keywords_str])[0]
            topic_embedding = keyword_embedding_cache[keywords_str]

            flirt_embedding = CONCEPT_EMBEDDINGS.get("FLIRTATION")
            if flirt_embedding is not None and topic_embedding is not None:
                sexual_similarity = cosine_similarity(topic_embedding.reshape(1, -1), flirt_embedding.reshape(1, -1))[0][0]
                scores['sexual'] = sexual_similarity

        # --- Final Assignment based on Priority ---
        # Find the highest-priority category that has a score above a certain threshold
        assigned = False
        for category, _ in sorted(CATEGORY_PRIORITY.items(), key=lambda item: item[1], reverse=True):
            # Using a simple threshold for now. This can be tuned.
            if scores.get(category, 0) > 0.4:
                final_categorized_topics[category].append(topic_name)
                assigned = True
                break # Assign to the highest priority category only

        if not assigned:
            final_categorized_topics["neutral"].append(topic_name)

    return dict(final_categorized_topics)
