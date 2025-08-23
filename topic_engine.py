# In topic_engine.py
import numpy as np
import hdbscan
import yake
from typing import List, Dict, Any
from collections import defaultdict
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity

from embedder import embedder_service

def _deduplicate_and_merge_topics(
    topics: List[Dict[str, Any]],
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    string_threshold=85,
    embedding_threshold=0.9
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

DATING_TAXONOMY = {
    "Logistics": ["plan", "meet", "dinner", "coffee", "drinks", "when", "where", "time", "number", "schedule"],
    "Flirting": ["cute", "hot", "sexy", "beautiful", "gorgeous", "haha", "lol", "omg", "wow", "feel", "vibe", "kiss", "date"],
    "Hobbies & Interests": ["hobbies", "interests", "music", "movie", "book", "show", "art", "sport", "game", "hike", "outdoors", "travel", "food"],
    "Work & Ambition": ["work", "job", "career", "company", "office", "finance", "project", "ambition", "goal"],
    "Deeper Connection": ["family", "friends", "relationship", "values", "feelings", "life", "story", "dream", "connect"],
}

taxonomy_embeddings = {
    category: embedder_service.encode_cached([" ".join(keywords)])[0]
    for category, keywords in DATING_TAXONOMY.items()
}
taxonomy_categories = list(DATING_TAXONOMY.keys())

def _map_topics_to_taxonomy(topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Maps each topic to a category from the DATING_TAXONOMY using embedding similarity.
    """
    if not topics:
        return []

    for topic in topics:
        keyword_str = " ".join(topic["keywords"])
        if not keyword_str.strip():
            topic["category"] = "Uncategorized"
            continue

        topic_embedding = embedder_service.encode_cached([keyword_str])[0]
        similarities = [cosine_similarity(topic_embedding.reshape(1, -1), tax_emb.reshape(1, -1))[0][0] for tax_emb in taxonomy_embeddings.values()]

        if similarities:
            best_match_index = np.argmax(similarities)
            topic["category"] = taxonomy_categories[best_match_index]
        else:
            topic["category"] = "Uncategorized"

    return topics

def identify_topics(conversation_turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Runs the full topic analysis pipeline.
    """
    if not conversation_turns:
        return []

    # Use the 'processed_content' from the preprocessor
    processed_texts = [turn.get('processed_content', '') for turn in conversation_turns]
    # Keep a map to the original turn object
    text_to_turn_map = {proc: turn for proc, turn in zip(processed_texts, conversation_turns) if proc}

    if not text_to_turn_map:
        return []

    valid_processed_texts = list(text_to_turn_map.keys())
    embeddings = embedder_service.encode_cached(valid_processed_texts)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='cosine', cluster_selection_method='eom', allow_single_cluster=True)
    cluster_labels = clusterer.fit_predict(embeddings)

    # Group texts by cluster label
    clustered_texts = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        if label == -1: continue # Skip outliers
        clustered_texts[label].append(valid_processed_texts[i])

    # Extract keywords and form initial topics
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=5, features=None)
    initial_topics = []
    for label, texts in clustered_texts.items():
        if not texts: continue
        full_cluster_text = " ".join(texts)
        keywords = [kw for kw, score in kw_extractor.extract_keywords(full_cluster_text)]
        # Map back to original messages
        messages = [text_to_turn_map[text]['content'] for text in texts]
        initial_topics.append({
            "topic_id": int(label),
            "keywords": keywords,
            "messages": messages,
            "message_turns": [text_to_turn_map[text] for text in texts] # Keep original turn objects
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

        # Simple canonicalization: use the first keyword
        canonical_name = topic["keywords"][0]

        final_topics.append({
            "canonical_name": canonical_name,
            "keywords": topic["keywords"],
            "category": "Uncategorized", # will be filled by taxonomy mapping
            "message_count": len(topic["messages"]),
            "messages": topic["messages"],
            "message_turns": topic["message_turns"]
        })

    # Map to taxonomy
    categorized_topics = _map_topics_to_taxonomy(final_topics)

    # Sort by message count
    sorted_topics = sorted(categorized_topics, key=lambda x: x["message_count"], reverse=True)

    return sorted_topics
