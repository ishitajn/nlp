# In app/svc/suggestion_engine.py

import os
import json
import random
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from embedder import embedder_service
from model import Feedback

# Constants
GRAPH_DATA_PATH = "app/data/transition_graph.json"
METADATA_PATH = "app/data/topic_metadata.json"

# ==============================================================================
# == THE DEFINITIVE, UNCENSORED & EXPANDED SUGGESTION BANK                    ==
# ==============================================================================
SUGGESTION_TEMPLATES = {
    "topics": {
        "contextual": ["Playful fantasy about {topic}", "Deeper dive into {topic}", "Adventurous take on {topic}", "A secret desire related to {topic}", "Connect {topic} to a personal story"],
        "escalation": ["Ideal late-night vibes", "Sharing a secret desire", "Discussing 'our' chemistry", "A fantasy scenario you've imagined", "What turns you on most"],
        "rapport": ["Favorite travel memories", "A skill they want to learn", "Their passion projects", "Childhood dreams or ambitions", "What makes them feel alive"],
        "fallback": ["Most adventurous thing they've done", "Their ideal perfect day", "A hidden talent they have", "Favorite type of humor", "A guilty pleasure"],
    },
    "questions": {
        "contextual": ["Favorite memory of {topic}?", "How does {topic} make you feel?", "A secret about your love for {topic}?", "Wildest experience with {topic}?", "Connect {topic} to a fantasy?"],
        "escalation": ["What makes you feel most desired?", "A fantasy you've never shared?", "What's our first adventure?", "Ideal cozy night together?", "Biggest turn-on?"],
        "rapport": ["What drives your passion for {topic}?", "A memory tied to {topic}?", "How did you get into {topic}?", "What does {topic} mean to you?", "Proudest moment related to {topic}?"],
        "fallback": ["Something you're curious about?", "What always makes you laugh?", "Biggest goal for this year?", "A story you love to tell?", "What's your love language?"],
    },
    "intimacy": {
        "contextual": ["Compliment their view on {topic}", "Relate {topic} to a personal feeling", "Appreciate their passion for {topic}", "Share a vulnerability about {topic}", "Connect {topic} to a shared future"],
        "validation": ["Acknowledge their irresistible energy", "Compliment their personality deeply", "Appreciate their openness and trust", "Validate their feelings or perspective", "Express how they make you feel"],
        "connection": ["Share a small, personal secret", "Mention a specific future experience", "Reference an inside joke we have", "Express genuine appreciation for them", "Ask about their emotional needs"],
        "fallback": ["Compliment their way of thinking", "Mention their great energy", "Acknowledge their unique humor", "Express enjoyment of the conversation", "Ask what makes them feel connected"],
    },
    "sexual": {
        "high_tension": ["Imagining feeling your body during {topic}", "Your energy is making me ache for you", "Describe a sensual act involving {topic}", "Can't stop thinking about you naked", "Tell me a dirty secret"],
        "medium_tension": ["Playfully tease about their charm", "Suggestive compliment on their vibe", "Hint at our 'dangerous' chemistry", "Curiosity about their morning energy", "What if we were together right now..."],
        "low_tension": ["A playful, flirty compliment", "Mentioning the obvious spark", "A cheeky 'what if' scenario", "A suggestive emoji 😉", "You have a captivating smile"],
        "fallback": ["Compliment their confidence", "Acknowledge their attractive energy", "A simple, playful wink emoji", "Mentioning a 'spark'", "You seem like fun"],
    }
}

# ==============================================================================
# == THE DEFINITIVE SUGGESTION STRATEGY MAP                                   ==
# ==============================================================================
# This map now uses the categories from topic_engine.py's DATING_TAXONOMY
SUGGESTION_STRATEGY_MAP = {
    "Work & Ambition": {"topics": ["rapport"], "questions": ["rapport"], "intimacy": ["validation"], "sexual": ["low_tension"]},
    "Deeper Connection": {"topics": ["rapport"], "questions": ["rapport"], "intimacy": ["connection"], "sexual": ["low_tension"]},
    "Hobbies & Interests": {"topics": ["rapport", "contextual"], "questions": ["contextual"], "intimacy": ["contextual"], "sexual": ["medium_tension"]},
    "Flirting": {"topics": ["escalation"], "questions": ["escalation"], "intimacy": ["validation", "contextual"], "sexual": ["medium_tension", "high_tension"]},
    "Logistics": {"topics": ["escalation"], "questions": ["escalation"], "intimacy": ["connection"], "sexual": ["high_tension"]},
    "Uncategorized": {"topics": ["fallback"], "questions": ["fallback"], "intimacy": ["fallback"], "sexual": ["fallback"]},
}

# Simplified phase model
CONVERSATION_PHASE_MAP = {
    "Icebreaker": {"next_themes": ["Hobbies & Interests", "Flirting"]},
    "Rapport Building": {"next_themes": ["Hobbies & Interests", "Flirting", "Deeper Connection"]},
    "Escalation": {"next_themes": ["Deeper Connection", "Logistics", "Flirting"]},
    "Explicit Banter": {"next_themes": ["Deeper Connection", "Flirting"]},
    "Logistics": {"next_themes": ["Flirting"]},
}


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
            
            # --- Enhanced Weighting ---
            from_turn = self.turns[i]
            to_turn = self.turns[i+1]

            # 1. Recency Weight
            recency_weight = 0.95 ** (num_turns - 1 - i)

            # 2. Sentiment Weight & Alignment
            from_sentiment = self.sentiment_analyzer.polarity_scores(from_turn['content'])['compound']
            to_sentiment = self.sentiment_analyzer.polarity_scores(to_turn['content'])['compound']

            sentiment_weight = 1.0 + from_sentiment

            # Reward transitions that maintain a consistent emotional tone
            sentiment_alignment = 1.2 if (from_sentiment > 0 and to_sentiment > 0) or \
                                        (from_sentiment < 0 and to_sentiment < 0) else 0.8

            # (Phase alignment would be added here if per-turn phase data was available)

            adjusted_score = 1.0 * recency_weight * sentiment_weight * sentiment_alignment
            graph[from_topic][to_topic] += adjusted_score

        return graph

    def get_suggestions(self) -> Dict[str, List[str]]:
        topic_sequence = self._get_topic_sequence()
        current_topic_name = next((t for t in reversed(topic_sequence) if t is not None), None)

        candidate_info = defaultdict(lambda: {'score': 0.0, 'reasons': set()})

        # 1. Add candidates from the transition graph
        if current_topic_name and current_topic_name in self.transition_graph:
            for next_topic, score in self.transition_graph[current_topic_name].items():
                candidate_info[next_topic]['score'] += score
                candidate_info[next_topic]['reasons'].add("Natural transition")

        # 2. Add other topics present in the conversation as candidates
        all_topic_names = {t["canonical_name"] for t in self.topics}
        for topic_name in all_topic_names:
            if topic_name not in candidate_info:
                candidate_info[topic_name]['score'] += 0.1 # Small boost for being mentioned
                candidate_info[topic_name]['reasons'].add("Mentioned in conversation")

        # 3. Apply scoring based on context
        recency_map = self.analysis.get("topic_recency", {})
        saliency_map = self.analysis.get("topic_saliency", {})

        for topic_name, info in candidate_info.items():
            # Penalize recent topics
            if topic_name in recency_map:
                penalty = 0.5 * (1 / recency_map[topic_name])
                info['score'] *= penalty
            # Boost salient topics
            if topic_name in saliency_map:
                info['score'] += saliency_map[topic_name] * 0.2

        # 4. Filter out the current topic and very similar topics
        if current_topic_name:
            if current_topic_name in candidate_info:
                del candidate_info[current_topic_name]

            # Semantic filtering
            candidate_names = list(candidate_info.keys())
            if candidate_names:
                current_topic_embedding = embedder_service.encode_cached([current_topic_name])
                candidate_embeddings = embedder_service.encode_cached(candidate_names)
                similarities = cosine_similarity(current_topic_embedding, candidate_embeddings)[0]

                for i, topic_name in enumerate(candidate_names):
                    if similarities[i] > 0.85:
                        candidate_info[topic_name]['score'] *= 0.1 # Penalize
                        candidate_info[topic_name]['reasons'].add("Similar to current topic")

        # 5. Sort and select top N
        sorted_candidates = sorted(candidate_info.items(), key=lambda item: item[1]['score'], reverse=True)
        top_5_suggestions = [{"topic": t, "score": i['score'], "reason": " ".join(list(i['reasons']))} for t, i in sorted_candidates[:5]]

        return self._format_suggestions_for_output(top_5_suggestions)

    def _format_suggestions_for_output(self, suggestions_with_reasons: List[Dict]) -> Dict[str, str]:
        suggested_topics = [s['topic'] for s in suggestions_with_reasons]
        suggestions = { "topics": [], "questions": [], "intimacy": [], "sexual": [] }

        # Create a lookup for topic details
        topic_details_map = {t["canonical_name"]: t for t in self.topics}

        for category in suggestions.keys():
            category_suggestions = set()
            for canonical_topic_name in suggested_topics:
                if len(category_suggestions) >= 5: break

                topic_details = topic_details_map.get(canonical_topic_name)

                # If topic is not from current convo, get its category from the metadata cache
                if not topic_details:
                    topic_category = self.topic_metadata.get(canonical_topic_name, "Uncategorized")
                    # We don't have keywords for these old topics, so we'll use the name itself
                    topic_details = {"category": topic_category, "keywords": [canonical_topic_name]}
                else:
                    topic_category = topic_details.get("category", "Uncategorized")
                strategies = SUGGESTION_STRATEGY_MAP.get(topic_category, {}).get(category, [])
                if not strategies: continue

                keyword_to_use = random.choice(topic_details.get("keywords", [canonical_topic_name]))
                strategy = random.choice(strategies)
                template = random.choice(SUGGESTION_TEMPLATES[category][strategy])
                suggestion = template.format(topic=keyword_to_use)

                if suggestion not in category_suggestions:
                    category_suggestions.add(suggestion)

            suggestions[category] = list(category_suggestions)

        # Fill with fallbacks if not enough suggestions were generated and format final output
        final_suggestions = {}
        for category, items in suggestions.items():
            if len(items) < 2: # Ensure at least 2 suggestions for the new format
                needed = 2 - len(items)
                fallback_templates = SUGGESTION_TEMPLATES[category]["fallback"]
                for _ in range(needed):
                    suggestion = random.choice(fallback_templates).format(topic="your vibe")
                    if suggestion not in items:
                        items.append(suggestion)

            # Truncate to 2, and format as a JSON string
            final_suggestions[category] = json.dumps(items[:2])

        return final_suggestions

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