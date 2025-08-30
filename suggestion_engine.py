"""
This engine is responsible for generating topic suggestions based on the
conversation analysis.
"""
import os
import json
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics.pairwise import cosine_similarity

from preprocessor import extract_canonical_phrases

# --- Constants ---
SUGGESTION_CATEGORIES = ["focus", "avoid", "neutral", "sensitive", "romantic", "fetish", "sexual"]

# --- Topic Metadata Handling ---
def _load_topic_metadata(path: str) -> Dict[str, str]:
    """Loads the topic metadata from a JSON file."""
    if os.path.exists(path):
        try:
            with open(path, 'r') as f: return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError): return {}
    return {}

def _save_topic_metadata(path: str, data: Dict[str, str]):
    """Saves the topic metadata to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f: json.dump(data, f, indent=4)

# --- Suggestion Engine Class ---
class ContextualSuggestionEngine:
    """
    A predictive engine that generates novel, contextually-related topic suggestions.
    """
    def __init__(
        self,
        services,
        categorized_topics: Dict[str, List[str]],
        topic_map: Dict[str, Any],
        my_profile: str = "",
        their_profile: str = "",
        use_enhanced_nlp: bool = False
    ):
        self.services = services
        self.my_profile = my_profile
        self.their_profile = their_profile
        self.use_enhanced_nlp = use_enhanced_nlp
        self.categorized_topics = categorized_topics
        self.topic_map = topic_map
        self.topic_metadata = _load_topic_metadata(services.config['paths']['topic_metadata'])
        self.discussed_topics = {topic.lower() for topics in self.categorized_topics.values() for topic in topics}

    def _update_and_save_persistent_data(self):
        """Updates the global topic metadata cache and saves it to disk if changed."""
        updated = False
        for category, topics in self.categorized_topics.items():
            for topic in topics:
                if (topic_lower := topic.lower()) not in self.topic_metadata:
                    self.topic_metadata[topic_lower] = category
                    updated = True
        if updated: _save_topic_metadata(self.services.config['paths']['topic_metadata'], self.topic_metadata)

    def _find_semantically_similar_topics(self, seed_topics: List[str], category: str) -> List[str]:
        """Finds new topics from the metadata that are similar to the seed topics."""
        if not seed_topics: return []

        candidate_topics = [
            topic for topic, cat in self.topic_metadata.items()
            if cat == category and topic not in self.discussed_topics
        ]
        if not candidate_topics: return []

        seed_embeddings = [
            np.mean(self.services.embedder.encode_cached([turn['content'] for turn in self.topic_map.get(seed.lower(), [])]), axis=0)
            for seed in seed_topics if self.topic_map.get(seed.lower())
        ]
        if not seed_embeddings: return []

        candidate_embeddings = self.services.embedder.encode_cached(candidate_topics)
        if candidate_embeddings.size == 0: return []

        similarity_matrix = cosine_similarity(np.array(seed_embeddings), candidate_embeddings)
        best_scores = np.max(similarity_matrix, axis=0)
        ranked_indices = np.argsort(best_scores)[::-1]

        return [candidate_topics[i] for i in ranked_indices]

    def generate(self, behavioral_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Main generation logic."""
        suggestions = self._generate_standard_suggestions()
        if self.use_enhanced_nlp:
            suggestions = self._generate_enhanced_suggestions(suggestions, behavioral_analysis)

        self._update_and_save_persistent_data()
        return suggestions

    def _generate_standard_suggestions(self) -> Dict[str, List[str]]:
        """Generates suggestions based on existing conversation topics."""
        suggestions = {cat: [] for cat in SUGGESTION_CATEGORIES}
        for category in SUGGESTION_CATEGORIES:
            if seed_topics := self.categorized_topics.get(category, []):
                predicted_topics = self._find_semantically_similar_topics(seed_topics, category)
                suggestions[category] = [topic.title() for topic in predicted_topics[:2]]
        return suggestions

    def _generate_enhanced_suggestions(self, suggestions: Dict, behavioral_analysis: Dict) -> Dict:
        """Adds richer suggestions using profile data and behavioral cues."""
        profile_text = f"{self.my_profile} {self.their_profile}"
        if profile_text.strip():
            profile_phrases = extract_canonical_phrases(profile_text, **self.services.get_preprocessor_args())
            new_profile_topics = [p for p in profile_phrases if p.lower() not in self.discussed_topics]

            suggestions["neutral"] = suggestions.get("neutral", []) + [t.title() for t in new_profile_topics[:3]]

            if behavioral_analysis.get("suggest_topic_shift") and new_profile_topics:
                suggestions["topic_shift_suggestion"] = [new_profile_topics[0].title()]
        return suggestions

# --- Public Interface ---
def generate_suggestions(
    services,
    categorized_topics: Dict[str, List[str]],
    topic_map: Dict[str, Any],
    behavioral_analysis: Dict[str, Any],
    my_profile: str = "",
    their_profile: str = "",
    use_enhanced_nlp: bool = False,
    **kwargs
) -> Dict[str, List[str]]:
    """
    Initializes and runs the ContextualSuggestionEngine.
    """
    engine = ContextualSuggestionEngine(
        services=services,
        categorized_topics=categorized_topics,
        topic_map=topic_map,
        my_profile=my_profile,
        their_profile=their_profile,
        use_enhanced_nlp=use_enhanced_nlp
    )
    return engine.generate(behavioral_analysis)