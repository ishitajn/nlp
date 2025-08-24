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
        topic_sequence = self._get_topic_sequence()
        current_topic_name = next((t for t in reversed(topic_sequence) if t is not None), None)

        candidate_info = defaultdict(lambda: {'score': 0.0, 'reasons': set()})

        if current_topic_name and current_topic_name in self.transition_graph:
            for next_topic, score in self.transition_graph[current_topic_name].items():
                candidate_info[next_topic]['score'] += score
                candidate_info[next_topic]['reasons'].add("Natural transition")

        all_topic_names = {t["canonical_name"] for t in self.topics}
        for topic_name in all_topic_names:
            if topic_name not in candidate_info:
                candidate_info[topic_name]['score'] += 0.1
                candidate_info[topic_name]['reasons'].add("Mentioned in conversation")

        recency_map = self.analysis.get("contextual_features", {}).get("topic_recency", {})
        saliency_map = self.analysis.get("contextual_features", {}).get("topic_saliency", {})

        for topic_name, info in candidate_info.items():
            if topic_name in recency_map:
                info['score'] *= 0.5 * (1 / recency_map[topic_name])
            if topic_name in saliency_map:
                info['score'] += saliency_map[topic_name] * 0.2

        if current_topic_name:
            if current_topic_name in candidate_info:
                del candidate_info[current_topic_name]

            candidate_names = list(candidate_info.keys())
            if candidate_names:
                current_topic_embedding = embedder_service.encode_cached([current_topic_name])[0]
                candidate_embeddings = embedder_service.encode_cached(candidate_names)
                similarities = cosine_similarity(current_topic_embedding.reshape(1, -1), candidate_embeddings)[0]

                for i, topic_name in enumerate(candidate_names):
                    if similarities[i] > 0.85:
                        candidate_info[topic_name]['score'] *= 0.1
                        candidate_info[topic_name]['reasons'].add("Similar to current topic")

        sorted_candidates = sorted(candidate_info.items(), key=lambda item: item[1]['score'], reverse=True)
        top_topic_names = [t for t, i in sorted_candidates[:5]]

        return {"topics": top_topic_names}

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
