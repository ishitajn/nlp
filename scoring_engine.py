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
FETISH_KEYWORDS = [r'\b(kink|fetish|bdsm|dom|sub|foot|feet|choke|spank)\b'] # Removed 'daddy', 'kitten'

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
    if keyword_embeddings is None or keyword_embeddings.size == 0:
        return 0.0
    similarities = cosine_similarity(keyword_embeddings, cluster_centroid.reshape(1, -1))
    return float(np.mean(similarities)) if similarities.size > 0 else 0.0

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

    # Cache for topic embeddings to avoid re-computation
    topic_embedding_cache = {}

    for cluster in topic_clusters:
        topic_name = cluster.get("canonical_name", "Unknown Topic")
        keywords_str = " ".join(cluster.get("keywords", []))

        # --- Score Calculation ---
        scores = defaultdict(float)

        # Salience Score (used for focus)
        cluster_size_normalized = len(cluster.get("messages", [])) / max_cluster_size
        keyword_centrality = _calculate_keyword_centrality(cluster.get("keywords", []), cluster.get("centroid"))
        recency_rank = context.get("topic_recency", {}).get(topic_name, 10) # Default to a low rank
        recency_weight = 1 / (recency_rank + 1) # Avoid division by zero, smooth out effect
        scores['focus'] = (cluster_size_normalized + keyword_centrality + recency_weight) / 3

        # Keyword-based scores for clear-cut categories
        if any(re.search(p, keywords_str, re.IGNORECASE) for p in AVOID_KEYWORDS):
            scores['avoid'] = 1.0
        if any(re.search(p, keywords_str, re.IGNORECASE) for p in SENSITIVE_KEYWORDS):
            scores['sensitive'] = 1.0
        if any(re.search(p, keywords_str, re.IGNORECASE) for p in FETISH_KEYWORDS):
            scores['fetish'] = 1.0

        # Semantic similarity score for 'sexual' using multiple concept embeddings
        if keywords_str:
            if keywords_str not in topic_embedding_cache:
                # Use the cluster's centroid for a more stable representation than concatenated keywords
                topic_embedding_cache[keywords_str] = cluster.get("centroid")

            topic_embedding = topic_embedding_cache[keywords_str]

            if topic_embedding is not None:
                sexual_concept_embeddings = [
                    CONCEPT_EMBEDDINGS.get("FLIRTATION"),
                    CONCEPT_EMBEDDINGS.get("ROMANTIC"),
                    CONCEPT_EMBEDDINGS.get("SEXUAL_ADVANCE")
                ]

                max_similarity = 0.0
                for concept_embedding in sexual_concept_embeddings:
                    if concept_embedding is not None:
                        similarity = cosine_similarity(topic_embedding.reshape(1, -1), concept_embedding.reshape(1, -1))[0][0]
                        if similarity > max_similarity:
                            max_similarity = similarity

                scores['sexual'] = float(max_similarity)

        # --- Final Assignment based on Priority and Thresholds ---
        assigned = False
        # Sort categories by priority to handle overlaps correctly
        for category, _ in sorted(CATEGORY_PRIORITY.items(), key=lambda item: item[1], reverse=True):
            score = scores.get(category, 0)

            # Define thresholds for each category to allow for more nuanced control
            threshold = 0.5 if category == 'sexual' else 0.6 # Stricter threshold for keyword matches

            if score >= threshold:
                final_categorized_topics[category].append(topic_name)
                assigned = True
                break # Assign to the highest priority category only

        if not assigned:
            # Only assign to focus if it has a reasonably high score, otherwise neutral
            if scores.get('focus', 0) > 0.4:
                 final_categorized_topics["focus"].append(topic_name)
            else:
                 final_categorized_topics["neutral"].append(topic_name)

    return dict(final_categorized_topics)
