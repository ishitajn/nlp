import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any, Tuple

def discover_and_label_topics(turns: List[Dict[str, Any]], vectors: np.ndarray, n_clusters: int = 4) -> Dict[str, Any]:
    """
    Discovers topics in the conversation by clustering embeddings and labels them
    using TF-IDF to find representative keywords.
    """
    if not turns or vectors.shape[0] < n_clusters:
        return {"topics": [], "turn_to_topic_map": {}}

    # 1. Cluster embeddings to discover topics
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(vectors)
    labels = kmeans.labels_

    # Map each turn to its topic cluster
    turn_to_topic_map = {}
    for i, turn in enumerate(turns):
        # Using turn content as a key; a unique ID would be better.
        turn_to_topic_map[turn['content']] = int(labels[i])

    # 2. Label each topic cluster
    labeled_topics = []
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue

        # Get all text for the current cluster
        cluster_texts = [turns[j]['content'] for j in cluster_indices]

        # Use TF-IDF to find top keywords for the topic label
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=3)
            vectorizer.fit(cluster_texts)
            keywords = vectorizer.get_feature_names_out()
            topic_label = ", ".join(keywords)
        except ValueError:
            # This can happen if the cluster has very few words (e.g., only "ok", "yes")
            topic_label = cluster_texts[0][:30] # Fallback to first few words


        # Find the most representative turn (closest to the centroid)
        centroid = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(vectors[cluster_indices] - centroid, axis=1)
        representative_idx = cluster_indices[np.argmin(distances)]
        representative_turn = turns[representative_idx]

        labeled_topics.append({
            "topic_id": int(i),
            "label": topic_label,
            "representative_turn": representative_turn,
            "turn_count": len(cluster_indices),
            # The prompt mentions dynamic labeling (positive, avoid, sensitive).
            # This would require another model (e.g., zero-shot classifier).
            "sentiment": "neutral" # Placeholder
        })

    return {
        "topics": labeled_topics,
        "turn_to_topic_map": turn_to_topic_map
    }
