# In suggestion_engine.py
import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from embedder import embedder_service
from model import Feedback

# Constants
GRAPH_DATA_PATH = "data/transition_graph.json"
METADATA_PATH = "data/topic_metadata.json"

# A list of high-quality, generic topics to suggest when the conversation lacks direction.
EVERGREEN_TOPICS = [
    "Spontaneous adventures", "Hidden talents", "Favorite type of humor",
    "A passion project", "The perfect day", "Childhood dreams",
    "Learning a new skill", "Favorite travel stories", "A guilty pleasure",
    "What makes you feel alive"
]

class AdvancedSuggestionEngine:
    def __init__(self, analysis_data: Dict[str, Any], conversation_turns: List[Dict[str, Any]], identified_topics: List[Dict[str, Any]], feedback: Optional[List[Feedback]] = None):
        self.analysis = analysis_data
        self.turns = conversation_turns
        self.topics = identified_topics
        self.feedback = feedback or []
        self.graph_path = GRAPH_DATA_PATH
        self.metadata_path = METADATA_PATH
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Learning Pipeline
        self.transition_graph, self.topic_metadata = self._load_persistent_data()
        session_graph = self._build_transition_graph_from_session()
        self._merge_graphs(self.transition_graph, session_graph)
        self._apply_feedback()
        self._update_and_save_persistent_data()

    def _load_persistent_data(self) -> (Dict, Dict):
        graph = defaultdict(lambda: defaultdict(float))
        if os.path.exists(self.graph_path):
            with open(self.graph_path, 'r') as f:
                loaded_graph = json.load(f)
                for from_topic, to_topics in loaded_graph.items():
                    graph[from_topic] = defaultdict(float, to_topics)

        metadata = {}
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)

        return graph, metadata

    def _update_and_save_persistent_data(self):
        # Update metadata with topics from the current session
        for topic in self.topics:
            self.topic_metadata[topic["canonical_name"]] = topic["category"]

        # Save both files
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        with open(self.graph_path, 'w') as f:
            json.dump(self.transition_graph, f, indent=4)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.topic_metadata, f, indent=4)

    def _merge_graphs(self, main_graph: Dict, session_graph: Dict):
        for from_topic, to_topics in session_graph.items():
            for to_topic, score in to_topics.items():
                main_graph[from_topic][to_topic] += score

    def _apply_feedback(self):
        for fb in self.feedback:
            if fb.action == "chosen":
                from_topic = fb.current_topic
                to_topic = fb.chosen_suggestion
                self.transition_graph[from_topic][to_topic] *= 1.2
                self.transition_graph[from_topic][to_topic] += 0.5

    def _get_topic_sequence(self) -> List[Optional[str]]:
        """Reconstructs the topic sequence from the identified topics."""
        turn_to_topic_map = {}
        for topic in self.topics:
            for turn in topic["message_turns"]:
                # Using turn 'id' or a unique identifier if available, otherwise content
                turn_key = turn.get('id', turn['content'])
                turn_to_topic_map[turn_key] = topic["canonical_name"]

        sequence = []
        for turn in self.turns:
            turn_key = turn.get('id', turn['content'])
            sequence.append(turn_to_topic_map.get(turn_key))
        return sequence

    def _build_transition_graph_from_session(self) -> Dict[str, Dict[str, float]]:
        topic_sequence = self._get_topic_sequence()
        graph = defaultdict(lambda: defaultdict(float))
        num_turns = len(self.turns)
        for i in range(num_turns - 1):
            from_topic, to_topic = topic_sequence[i], topic_sequence[i+1]
            if not from_topic or not to_topic or from_topic == to_topic:
                continue
            
            from_turn = self.turns[i]
            to_turn = self.turns[i+1]

            recency_weight = 0.95 ** (num_turns - 1 - i)
            from_sentiment = self.sentiment_analyzer.polarity_scores(from_turn['content'])['compound']
            sentiment_weight = 1.0 + from_sentiment
            sentiment_alignment = 1.2 if (from_sentiment * self.sentiment_analyzer.polarity_scores(to_turn['content'])['compound']) > 0 else 0.8

            adjusted_score = 1.0 * recency_weight * sentiment_weight * sentiment_alignment
            graph[from_topic][to_topic] += adjusted_score

        return graph

    def get_suggestions(self) -> Dict[str, List[str]]:
        """
        Generates a list of new, relevant topics to suggest, avoiding repetition.
        """
        # 1. Get existing topic information
        existing_topic_names = {t["canonical_name"] for t in self.topics}
        existing_topic_embeddings = [t["centroid"] for t in self.topics if t.get("centroid") is not None]

        topic_sequence = self._get_topic_sequence()
        current_topic_name = next((t for t in reversed(topic_sequence) if t is not None), None)

        # 2. Generate a broad list of candidate topics
        candidate_topics = set()
        # Add candidates from the transition graph
        if current_topic_name and current_topic_name in self.transition_graph:
            for next_topic in self.transition_graph[current_topic_name]:
                candidate_topics.add(next_topic)
        # Add evergreen topics for variety
        for topic in EVERGREEN_TOPICS:
            candidate_topics.add(topic)

        # 3. Filter candidates
        # 3a. Remove topics already present in the conversation
        filtered_candidates = [t for t in candidate_topics if t not in existing_topic_names]

        # 3b. Remove topics that are too similar to any existing topic
        if existing_topic_embeddings and filtered_candidates:
            candidate_embeddings = embedder_service.encode_cached(filtered_candidates)
            # Create a single matrix of existing embeddings for efficient comparison
            existing_matrix = np.vstack(existing_topic_embeddings)

            # Calculate cosine similarity between all candidates and all existing topics
            similarity_matrix = cosine_similarity(candidate_embeddings, existing_matrix)

            # Find the max similarity for each candidate to any existing topic
            max_sim_per_candidate = similarity_matrix.max(axis=1)

            # Keep only candidates with a max similarity below a threshold
            SIMILARITY_THRESHOLD = 0.80
            final_candidates = [
                candidate for i, candidate in enumerate(filtered_candidates)
                if max_sim_per_candidate[i] < SIMILARITY_THRESHOLD
            ]
        else:
            final_candidates = filtered_candidates

        # 4. Score the remaining candidates (simple scoring for now, can be enhanced)
        # For now, we rely on the transition graph's implicit scores and the quality
        # of the evergreen list. A more advanced scoring could use context features.

        # 5. Return the top N suggestions
        # If the transition graph provided candidates, they are likely more relevant.
        # We can prioritize them, but for now, a simple slice is fine.
        return {"topics": final_candidates[:5]}

def generate_suggestions(
    analysis_data: Dict[str, Any],
    conversation_turns: List[Dict[str, Any]],
    identified_topics: List[Dict[str, Any]],
    feedback: Optional[List[Feedback]] = None
) -> Dict[str, List[str]]:
    """
    Generates 5 creative, context-aware suggestions for each category.
    """
    engine = AdvancedSuggestionEngine(analysis_data, conversation_turns, identified_topics, feedback)
    return engine.get_suggestions()
