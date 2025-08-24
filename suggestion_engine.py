# In suggestion_engine.py
import os
import json
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics.pairwise import cosine_similarity

from embedder import embedder_service

# --- Persistent Topic Metadata Handling (Optimized) ---
METADATA_PATH = "data/topic_metadata.json"
SUGGESTION_CATEGORIES = ["focus", "avoid", "neutral", "sensitive", "romantic", "fetish", "sexual"]

def _load_topic_metadata() -> Dict[str, str]:
    """Loads the topic metadata from a JSON file, once per application start."""
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return {}

def _save_topic_metadata(data: Dict[str, str]):
    """Saves the topic metadata to a JSON file."""
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, 'w') as f:
        json.dump(data, f, indent=4)

# Load metadata into a module-level cache on startup
TOPIC_METADATA = _load_topic_metadata()


class ContextualSuggestionEngine:
    """
    A predictive engine that generates novel, contextually-related topic suggestions
    based on topics already identified in the conversation.
    """
    def __init__(self, analysis_data: Dict[str, Any]):
        self.categorized_topics = analysis_data.get("categorized_topics", {})
        self.topic_map = analysis_data.get("topic_map", {})
        self.topic_metadata = TOPIC_METADATA  # Use the cached metadata
        self.discussed_topics = {
            topic.lower()
            for topic_list in self.categorized_topics.values()
            for topic in topic_list
        }

    def _update_and_save_persistent_data(self):
        """Updates the global topic metadata cache and saves it to disk if changed."""
        updated = False
        for category, topics in self.categorized_topics.items():
            for topic in topics:
                topic_lower = topic.lower()
                if topic_lower not in self.topic_metadata:
                    self.topic_metadata[topic_lower] = category
                    updated = True

        if updated:
            _save_topic_metadata(self.topic_metadata)

    def _find_semantically_similar_topics(self, seed_topics: List[str], category: str) -> List[str]:
        if not seed_topics: return []
        
        candidate_topics = [
            topic for topic, cat in self.topic_metadata.items()
            if cat == category and topic not in self.discussed_topics
        ]
        if not candidate_topics: return []

        # Get contextual embeddings for seed topics
        seed_embeddings = []
        for seed in seed_topics:
            source_contents = [turn['content'] for turn in self.topic_map.get(seed.lower(), [])]
            if source_contents:
                seed_embeddings.append(np.mean(embedder_service.encode_cached(source_contents), axis=0))
        
        if not seed_embeddings: return []

        candidate_embeddings = embedder_service.encode_cached(candidate_topics)
        if candidate_embeddings.size == 0: return []

        similarity_matrix = cosine_similarity(np.array(seed_embeddings), candidate_embeddings)
        best_scores = np.max(similarity_matrix, axis=0)
        ranked_indices = np.argsort(best_scores)[::-1]
        
        return [candidate_topics[i] for i in ranked_indices]

    def generate(self) -> Dict[str, List[str]]:
        suggestions = {cat: [] for cat in SUGGESTION_CATEGORIES}
        categories_to_predict = SUGGESTION_CATEGORIES

        for category in categories_to_predict:
            seed_topics = self.categorized_topics.get(category, [])
            if not seed_topics: continue
            
            predicted_topics = self._find_semantically_similar_topics(seed_topics, category)
            suggestions[category] = [topic.title() for topic in predicted_topics[:2]]

        self._update_and_save_persistent_data()
        return suggestions

def generate_suggestions(analysis_data: Dict[str, Any], **kwargs) -> Dict[str, List[str]]:
    engine = ContextualSuggestionEngine(analysis_data)
    return engine.generate()