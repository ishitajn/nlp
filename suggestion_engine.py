# In suggestion_engine.py
import os
import json
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics.pairwise import cosine_similarity

from embedder import embedder_service
from preprocessor import extract_canonical_phrases

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
    def __init__(self, analysis_data: Dict[str, Any], my_profile: str = "", their_profile: str = "", use_enhanced_nlp: bool = False):
        self.analysis_data = analysis_data
        self.my_profile = my_profile
        self.their_profile = their_profile
        self.use_enhanced_nlp = use_enhanced_nlp
        self.categorized_topics = analysis_data.get("categorized_topics", {})
        self.topic_map = analysis_data.get("topic_map", {})
        self.topic_metadata = TOPIC_METADATA
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
        if self.use_enhanced_nlp:
            suggestions = self._generate_enhanced_suggestions()
        else:
            suggestions = self._generate_standard_suggestions()

        self._update_and_save_persistent_data()
        return suggestions

    def _generate_standard_suggestions(self) -> Dict[str, List[str]]:
        """Generates suggestions based on existing conversation topics."""
        suggestions = {cat: [] for cat in SUGGESTION_CATEGORIES}
        for category in SUGGESTION_CATEGORIES:
            seed_topics = self.categorized_topics.get(category, [])
            if not seed_topics: continue
            
            predicted_topics = self._find_semantically_similar_topics(seed_topics, category)
            suggestions[category] = [topic.title() for topic in predicted_topics[:2]]
        return suggestions

    def _generate_enhanced_suggestions(self) -> Dict[str, List[str]]:
        """Generates richer suggestions using profile data and behavioral cues."""
        suggestions = self._generate_standard_suggestions()
        new_profile_topics = [] # Initialize to prevent NameError

        # Add personalized suggestions from profiles
        profile_text = f"{self.my_profile} {self.their_profile}"
        if profile_text.strip():
            profile_phrases = extract_canonical_phrases(profile_text)
            new_profile_topics = [p for p in profile_phrases if p.lower() not in self.discussed_topics]

            if "neutral" not in suggestions:
                suggestions["neutral"] = []
            suggestions["neutral"].extend([t.title() for t in new_profile_topics[:3]])

        # If engagement is low, suggest a new topic from profiles
        behavioral_analysis = self.analysis_data.get("behavioral_analysis", {})
        if behavioral_analysis.get("suggest_topic_shift") and new_profile_topics:
            suggestions["topic_shift_suggestion"] = [new_profile_topics[0].title()]

        return suggestions

def generate_suggestions(analysis_data: Dict[str, Any], my_profile: str = "", their_profile: str = "", use_enhanced_nlp: bool = False, **kwargs) -> Dict[str, List[str]]:
    """
    Main function to generate conversational suggestions based on analysis data.

    This function initializes and runs the ContextualSuggestionEngine to generate
    topic suggestions. It can operate in a standard or enhanced mode.

    Args:
        analysis_data: The main analysis object from the analysis pipeline.
        my_profile: The user's profile text.
        their_profile: The match's profile text.
        use_enhanced_nlp: Flag to enable enhanced suggestion generation.

    Returns:
        A dictionary of categorized topic suggestions.
    """
    engine = ContextualSuggestionEngine(
        analysis_data,
        my_profile=my_profile,
        their_profile=their_profile,
        use_enhanced_nlp=use_enhanced_nlp
    )
    return engine.generate()