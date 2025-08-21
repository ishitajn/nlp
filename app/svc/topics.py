import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any, Tuple, Set

# --- Topic Categorization Keywords ---
# These are simple keyword sets to categorize topics. A production system
# would benefit from a more sophisticated text classifier.
SEXUAL_KEYWORDS: Set[str] = {"sex", "sexy", "hot", "kink", "fetish", "fuck", "naked"}
AVOID_KEYWORDS: Set[str] = {"politics", "religion", "ex", "drama"}
# 'focus', 'sensitive', and 'fetish' categories are more complex and would require
# more advanced NLP models to be implemented accurately. We will focus on
# 'sexual', 'avoid', and 'neutral' for now.

def get_topic_category(keywords: List[str]) -> str:
    """Categorizes a topic based on its keywords."""
    keyword_set = set(keywords)
    if not keyword_set.isdisjoint(SEXUAL_KEYWORDS):
        return "sexual"
    if not keyword_set.isdisjoint(AVOID_KEYWORDS):
        return "avoid"
    return "neutral"

def discover_and_label_topics(turns: List[Dict[str, Any]], vectors: np.ndarray, n_clusters: int = 4) -> Dict[str, List[str]]:
    """
    Discovers topics, labels them with keywords, and categorizes them.
    The output is structured to fit the new `conversation_state` object.
    """
    if not turns or vectors.shape[0] < n_clusters:
        return {
            "focus": [], "avoid": [], "neutral": [], "sensitive": [],
            "fetish": [], "sexual": [], "recent_topics": []
        }

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(vectors)
    labels = kmeans.labels_

    # Prepare categorized topic lists
    categorized_topics: Dict[str, List[str]] = {
        "focus": [], "avoid": [], "neutral": [], "sensitive": [],
        "fetish": [], "sexual": []
    }

    recent_topic_labels = []

    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue

        cluster_texts = [turns[j]['content'] for j in cluster_indices]

        try:
            # Use TF-IDF to find top keywords for the topic label
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5)
            vectorizer.fit(cluster_texts)
            keywords = vectorizer.get_feature_names_out()
            topic_label = ", ".join(keywords)

            # Categorize the topic
            category = get_topic_category(keywords)
            categorized_topics[category].append(topic_label)
            recent_topic_labels.append(topic_label)

        except ValueError:
            # Fallback for very short texts
            topic_label = cluster_texts[0][:30]
            categorized_topics["neutral"].append(topic_label)
            recent_topic_labels.append(topic_label)

    # Add recent topics for context awareness
    output = categorized_topics
    output["recent_topics"] = recent_topic_labels

    return output
