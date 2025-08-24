# In topic_engine.py
import re
import numpy as np
import hdbscan
import yake
from typing import List, Dict, Any
from collections import defaultdict
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

from embedder import embedder_service
from preprocessor import preprocess_text
from semantic_concepts import CONCEPT_EMBEDDINGS


def _get_canonical_name(keywords: List[str], centroid: np.ndarray) -> str:
    """Chooses the most central keyword as the canonical name for a topic."""
    if not keywords:
        return "Unknown Topic"
    # If there's no centroid, fall back to the first keyword
    if centroid is None:
        return keywords[0]

    keyword_embeddings = embedder_service.encode_cached(keywords)
    # Handle cases where embedding fails or returns an empty array
    if keyword_embeddings is None or keyword_embeddings.size == 0:
        return keywords[0]

    similarities = cosine_similarity(keyword_embeddings, centroid.reshape(1, -1))
    most_central_keyword_index = int(np.argmax(similarities))
    return keywords[most_central_keyword_index]

def _deduplicate_and_merge_topics(
    topics: List[Dict[str, Any]],
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    string_threshold=90,
    embedding_threshold=0.95
) -> List[Dict[str, Any]]:
    """
    Deduplicates topics based on keyword and semantic similarity.
    """
    # First, calculate the semantic centroid for each topic
    topic_centroids = {}
    for topic in topics:
        topic_id = topic["topic_id"]
        indices = np.where(cluster_labels == topic_id)[0]
        if len(indices) > 0:
            topic_centroids[topic_id] = np.mean(embeddings[indices], axis=0)

    merged_topics = []
    merged_ids = set()

    for i in range(len(topics)):
        if topics[i]["topic_id"] in merged_ids:
            continue

        current_topic = topics[i]

        for j in range(i + 1, len(topics)):
            if topics[j]["topic_id"] in merged_ids:
                continue

            other_topic = topics[j]

            # 1. Keyword similarity
            keyword_sim = fuzz.token_set_ratio(" ".join(current_topic["keywords"]), " ".join(other_topic["keywords"]))

            # 2. Semantic similarity
            centroid1 = topic_centroids.get(current_topic["topic_id"])
            centroid2 = topic_centroids.get(other_topic["topic_id"])

            semantic_sim = 0
            if centroid1 is not None and centroid2 is not None:
                semantic_sim = cosine_similarity(centroid1.reshape(1, -1), centroid2.reshape(1, -1))[0][0]

            # If both are very similar, merge them
            if keyword_sim > string_threshold and semantic_sim > embedding_threshold:
                # Merge 'other_topic' into 'current_topic'
                current_topic["messages"].extend(other_topic["messages"])
                # Simple keyword union
                current_topic["keywords"] = list(set(current_topic["keywords"]) | set(other_topic["keywords"]))
                merged_ids.add(other_topic["topic_id"])

        merged_topics.append(current_topic)

    return merged_topics

def identify_topics(conversation_turns: List[Dict[str, Any]], their_profile: str = "") -> List[Dict[str, Any]]:
    """
    Runs the full topic analysis pipeline.
    """
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=5, features=None)
    initial_topics = []

    # --- Process Profile Topics ---
    if their_profile:
        processed_profile = preprocess_text(their_profile)
        profile_keywords = [kw for kw, score in kw_extractor.extract_keywords(processed_profile)]
        if profile_keywords:
            profile_embedding = embedder_service.encode_cached([processed_profile])[0]
            initial_topics.append({
                "topic_id": -2, # Special ID for profile topics
                "keywords": profile_keywords,
                "messages": ["From Profile"],
                "message_turns": [], # No turns from conversation
                "centroid": profile_embedding
            })


    if not conversation_turns:
        # If only profile topics exist, finalize and return them
        if initial_topics:
            topic = initial_topics[0]
            canonical_name = _get_canonical_name(topic["keywords"], topic.get("centroid"))
            return [{
                "canonical_name": canonical_name,
                "keywords": topic["keywords"],
                "category": "Uncategorized",
                "message_count": 1, # Represents the profile itself
                "messages": topic["messages"],
                "message_turns": topic["message_turns"],
                "centroid": topic.get("centroid")
            }]
        return []

    # --- 1. Preprocess text for each turn ---
    for turn in conversation_turns:
        if "content" in turn:
            turn["processed_content"] = preprocess_text(turn["content"])

    # Use the 'processed_content' from the preprocessor
    processed_texts = [turn.get('processed_content', '') for turn in conversation_turns]
    # Keep a map to the original turn object
    text_to_turn_map = {proc: turn for proc, turn in zip(processed_texts, conversation_turns) if proc}

    if not text_to_turn_map:
        return []

    valid_processed_texts = list(text_to_turn_map.keys())
    embeddings = embedder_service.encode_cached(valid_processed_texts)

    distance_matrix = cosine_distances(embeddings)
    if distance_matrix.dtype != np.float64:
        distance_matrix = distance_matrix.astype(np.float64)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='precomputed', cluster_selection_method='leaf', allow_single_cluster=True)
    cluster_labels = clusterer.fit_predict(distance_matrix)

    # Group texts by cluster label
    clustered_texts = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        if label == -1: continue # Skip outliers
        clustered_texts[label].append(valid_processed_texts[i])

    # Extract keywords and form initial topics
    # Get the embeddings for the texts that were actually clustered
    clustered_indices = np.where(cluster_labels != -1)[0]
    clustered_embeddings = embeddings[clustered_indices]
    clustered_labels = cluster_labels[clustered_indices]

    for label, texts in clustered_texts.items():
        if not texts: continue
        full_cluster_text = " ".join(texts)
        keywords = [kw for kw, score in kw_extractor.extract_keywords(full_cluster_text)]

        # Calculate centroid for the current cluster
        indices_in_cluster = np.where(clustered_labels == label)[0]
        centroid = np.mean(clustered_embeddings[indices_in_cluster], axis=0)

        # Map back to original messages
        messages = [text_to_turn_map[text]['content'] for text in texts]
        initial_topics.append({
            "topic_id": int(label),
            "keywords": keywords,
            "messages": messages,
            "message_turns": [text_to_turn_map[text] for text in texts], # Keep original turn objects
            "centroid": centroid
        })

    # Deduplicate topics
    if len(initial_topics) > 1:
        # We need the embeddings of the processed texts that were actually clustered
        clustered_embeddings = embeddings[cluster_labels != -1]
        # And the corresponding labels
        labels_for_dedup = cluster_labels[cluster_labels != -1]
        deduplicated_topics = _deduplicate_and_merge_topics(initial_topics, clustered_embeddings, labels_for_dedup)
    else:
        deduplicated_topics = initial_topics

    # Select canonical name and finalize structure
    final_topics = []
    for topic in deduplicated_topics:
        if not topic["keywords"]: continue

        # Use the new function to get a better canonical name
        canonical_name = _get_canonical_name(topic["keywords"], topic.get("centroid"))

        final_topics.append({
            "canonical_name": canonical_name,
            "keywords": topic["keywords"],
            "category": "Uncategorized", # will be filled by taxonomy mapping
            "message_count": len(topic["messages"]),
            "messages": topic["messages"],
            "message_turns": topic["message_turns"],
            "centroid": topic.get("centroid") # Pass the centroid along
        })

    # Sort by message count
    sorted_topics = sorted(final_topics, key=lambda x: x["message_count"], reverse=True)

    return sorted_topics


# --- From topic_categorizer.py ---

# --- Constants and Lexicons for Categorization ---

AVOID_KEYWORDS = [r'\b(politics|religion|government|election|vote|biden|trump|conservative|liberal)\b']
SENSITIVE_KEYWORDS = [r'\b(autism|adhd|ocd|bpd|trauma|disability|mental health|therapy|depression|anxiety|grief|loss)\b']
FETISH_KEYWORDS = [r'\b(kink|fetish|bdsm|dom|sub|foot|feet|choke|spank)\b'] # Removed 'daddy', 'kitten'

# Category priority for conflict resolution (higher number = higher priority)
CATEGORY_PRIORITY = {
    "sensitive": 5,
    "fetish": 4,
    "sexual": 4, # Highest non-sensitive priority
    "romantic": 3,
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

        # Semantic similarity scores for 'romantic' and 'sexual'
        if keywords_str:
            if keywords_str not in topic_embedding_cache:
                topic_embedding_cache[keywords_str] = cluster.get("centroid")

            topic_embedding = topic_embedding_cache[keywords_str]

            if topic_embedding is not None:
                # Sexual score is based only on direct advances
                sexual_embedding = CONCEPT_EMBEDDINGS.get("SEXUAL_ADVANCE")
                if sexual_embedding is not None:
                    scores['sexual'] = float(cosine_similarity(topic_embedding.reshape(1, -1), sexual_embedding.reshape(1, -1))[0][0])

                # Romantic score is based on romance and flirtation
                romantic_embeddings = [CONCEPT_EMBEDDINGS.get("ROMANTIC"), CONCEPT_EMBEDDINGS.get("FLIRTATION")]
                max_romantic_sim = 0.0
                for emb in romantic_embeddings:
                    if emb is not None:
                        sim = cosine_similarity(topic_embedding.reshape(1, -1), emb.reshape(1, -1))[0][0]
                        if sim > max_romantic_sim:
                            max_romantic_sim = sim
                scores['romantic'] = float(max_romantic_sim)

        # --- Final Assignment based on Priority and Thresholds ---
        assigned = False
        for category, _ in sorted(CATEGORY_PRIORITY.items(), key=lambda item: item[1], reverse=True):
            # Neutral is the fallback, so we don't check it in the loop
            if category == 'neutral':
                continue

            score = scores.get(category, 0.0)

            # Set a specific threshold for each category
            threshold = 0.5 # Default threshold
            if category == 'sexual':
                threshold = 0.6
            elif category == 'romantic':
                threshold = 0.5
            elif category in ['sensitive', 'fetish', 'avoid']:
                # Keyword-based scores are 1.0, so this ensures they are only triggered if matched
                threshold = 0.9
            elif category == 'focus':
                threshold = 0.4

            if score >= threshold:
                final_categorized_topics[category].append(topic_name)
                assigned = True
                break # Assign to the highest-priority category only

        if not assigned:
            final_categorized_topics["neutral"].append(topic_name)

    return dict(final_categorized_topics)
