# In topic_engine_v2.py
import numpy as np
import hdbscan
import yake
from typing import List, Dict, Any

from embedder import embedder_service
from preprocessor import preprocess_text

def run_topic_engine(conversation_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Runs the full topic analysis pipeline:
    1. Preprocesses text
    2. Creates sentence embeddings
    3. Clusters sentences with HDBSCAN
    4. Extracts keywords for each cluster with YAKE
    """

    if not conversation_turns:
        return {}

    # 1. Preprocess text
    # We're processing each turn's content.
    # We also keep track of original turns for context later.
    original_contents = [turn.get('content', '') for turn in conversation_turns]
    processed_texts = [preprocess_text(content) for content in original_contents]

    # Create a mapping from processed text back to original content, filtering out empty ones
    text_map = {
        proc: orig for proc, orig in zip(processed_texts, original_contents) if proc
    }

    if not text_map:
        return {"error": "No text left after preprocessing."}

    valid_processed_texts = list(text_map.keys())

    # 2. Create embeddings for the valid processed texts
    embeddings = embedder_service.encode_cached(valid_processed_texts)

    # 3. Cluster with HDBSCAN
    # Parameters for HDBSCAN can be tuned. These are reasonable defaults.
    # Using cosine metric as it's good for high-dimensional semantic vectors.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric='cosine',
        cluster_selection_method='eom'
    )
    cluster_labels = clusterer.fit_predict(embeddings)

    # 4. Group texts by cluster
    clustered_texts = {}
    # Note: We are clustering on processed texts, but we want to see the original content.
    for i, label in enumerate(cluster_labels):
        if label == -1:  # -1 is the noise cluster in HDBSCAN
            continue
        if label not in clustered_texts:
            clustered_texts[label] = []

        original_text = text_map[valid_processed_texts[i]]
        clustered_texts[label].append(original_text)

    # 5. Extract keywords using YAKE for each cluster
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=5, features=None)

    topics = []
    for label, texts in clustered_texts.items():
        if not texts:
            continue

        full_cluster_text = " ".join(texts)
        # YAKE works best on raw text, so we use the concatenated original texts
        keywords_with_scores = kw_extractor.extract_keywords(full_cluster_text)
        keywords = [kw for kw, score in keywords_with_scores]

        topics.append({
            "topic_id": int(label),
            "keywords": keywords,
            "messages": texts
        })

    # Sort topics by the number of messages in them, descending
    sorted_topics = sorted(topics, key=lambda x: len(x["messages"]), reverse=True)

    # 6. Map topics to a predefined taxonomy
    final_topics = _map_topics_to_taxonomy(sorted_topics)

    return {"identified_topics": final_topics}


DATING_TAXONOMY = {
    "Logistics": ["plan", "meet", "dinner", "coffee", "drinks", "when", "where", "time", "number", "schedule"],
    "Flirting": ["cute", "hot", "sexy", "beautiful", "gorgeous", "haha", "lol", "omg", "wow", "feel", "vibe", "kiss", "date"],
    "Hobbies & Interests": ["hobbies", "interests", "music", "movie", "book", "show", "art", "sport", "game", "hike", "outdoors", "travel", "food"],
    "Work & Ambition": ["work", "job", "career", "company", "office", "finance", "project", "ambition", "goal"],
    "Deeper Connection": ["family", "friends", "relationship", "values", "feelings", "life", "story", "dream", "connect"],
}

# Pre-compute embeddings for the taxonomy for efficiency
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
        # Create a single string from the topic's keywords to represent it
        keyword_str = " ".join(topic["keywords"])
        if not keyword_str.strip():
            topic["category"] = "Uncategorized"
            continue

        topic_embedding = embedder_service.encode_cached([keyword_str])[0]

        # Calculate similarity with each taxonomy category
        similarities = [
            cosine_similarity(topic_embedding.reshape(1, -1), tax_emb.reshape(1, -1))[0][0]
            for tax_emb in taxonomy_embeddings.values()
        ]

        # Find the best match
        if similarities:
            best_match_index = np.argmax(similarities)
            best_category = taxonomy_categories[best_match_index]
            topic["category"] = best_category
        else:
            topic["category"] = "Uncategorized"

    return topics
