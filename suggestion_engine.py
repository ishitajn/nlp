# In app/svc/suggestion_engine.py

import random
from typing import Dict, Any, List
from collections import defaultdict

# Import necessary services and libraries for advanced features
from embedder import embedder_service
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Import data from the analysis engine for topic mapping
from analysis_engine import nlp, canonical_topic_names, canonical_topic_vectors
from model import Feedback

# Constants
GRAPH_DATA_PATH = "app/data/transition_graph.json"

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
SUGGESTION_STRATEGY_MAP = {
    "Career & Ambition": {"topics": ["rapport"], "questions": ["rapport"], "intimacy": ["validation"], "sexual": ["low_tension"]},
    "Family & Background": {"topics": ["rapport"], "questions": ["rapport"], "intimacy": ["connection"], "sexual": ["low_tension"]},
    "Hobbies & Passions": {"topics": ["rapport", "contextual"], "questions": ["contextual"], "intimacy": ["contextual"], "sexual": ["medium_tension"]},
    "Travel & Adventure": {"topics": ["rapport", "contextual"], "questions": ["contextual"], "intimacy": ["connection"], "sexual": ["medium_tension"]},
    "Fitness & Health": {"topics": ["rapport"], "questions": ["rapport"], "intimacy": ["validation"], "sexual": ["low_tension"]},
    "Food & Drink": {"topics": ["rapport", "contextual"], "questions": ["contextual"], "intimacy": ["connection"], "sexual": ["medium_tension"]},
    "Flirting & Compliments": {"topics": ["escalation"], "questions": ["escalation"], "intimacy": ["validation", "contextual"], "sexual": ["medium_tension", "high_tension"]},
    "Deeper Connection": {"topics": ["rapport"], "questions": ["rapport"], "intimacy": ["validation", "connection"], "sexual": ["medium_tension"]},
    "Making Plans & Logistics": {"topics": ["escalation"], "questions": ["escalation"], "intimacy": ["connection"], "sexual": ["high_tension"]},
    "Sexual Escalation & Kinks": {"topics": ["escalation"], "questions": ["escalation"], "intimacy": ["validation"], "sexual": ["high_tension"]},
    "Pop Culture & Media": {"topics": ["rapport"], "questions": ["rapport"], "intimacy": ["connection"], "sexual": ["low_tension"]},
    "Inside Jokes & Nicknames": {"topics": ["contextual"], "questions": ["contextual"], "intimacy": ["connection"], "sexual": ["medium_tension"]},
}

# ==============================================================================
# == TOPIC HIERARCHY & CONVERSATION PHASE MODELS                              ==
# ==============================================================================
TOPIC_HIERARCHY = {
    "Personal Topics": {
        "sub_topics": ["Career & Ambition", "Family & Background", "Hobbies & Passions", "Fitness & Health", "Food & Drink", "Pop Culture & Media"],
    },
    "Interpersonal Topics": {
        "sub_topics": ["Flirting & Compliments", "Deeper Connection", "Making Plans & Logistics", "Inside Jokes & Nicknames"],
    },
    "Intimate Topics": {
        "sub_topics": ["Sexual Escalation & Kinks"],
    }
}

CONVERSATION_PHASE_MAP = {
    "Icebreaker": {"next": ["Personal Interests", "Flirting"], "themes": ["Personal Topics"]},
    "Personal Interests": {"next": ["Personal Interests", "Flirting", "Deeper Connection"], "themes": ["Personal Topics", "Interpersonal Topics"]},
    "Flirting": {"next": ["Personal Interests", "Deeper Connection", "Logistics / Planning", "Intimate Topics"], "themes": ["Interpersonal Topics"]},
    "Deeper Connection": {"next": ["Flirting", "Intimate Topics", "Logistics / Planning"], "themes": ["Interpersonal Topics", "Intimate Topics"]},
    "Logistics / Planning": {"next": ["Flirting"], "themes": ["Interpersonal Topics"]},
    "Intimate Topics": {"next": ["Deeper Connection", "Flirting"], "themes": ["Intimate Topics"]},
}


def _get_weighted_topics(occurrence: Dict[str, int], recency: Dict[str, int], count: int) -> List[str]:
    """Combines occurrence and recency for a human-like topic ranking."""
    scores = {}
    all_heatmap_topics = set(occurrence.keys()) | set(recency.keys())
    for topic in all_heatmap_topics:
        num = occurrence.get(topic, 0)
        recency_rank = recency.get(topic, 100)
        score = (1 / recency_rank) * 10 + num
        scores[topic] = score
    
    sorted_topics = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [topic for topic, score in sorted_topics[:count]]


class AdvancedSuggestionEngine:
    """
    A sophisticated suggestion engine that uses a weighted, persistent transition graph,
    semantic similarity, and conversation phase awareness to generate relevant suggestions.
    Includes a feedback loop to learn from user choices over time.
    """
    def __init__(self, analysis_data: Dict[str, Any], conversation_turns: List[Dict[str, Any]], feedback: Optional[List[Feedback]] = None):
        self.analysis = analysis_data
        self.turns = conversation_turns
        self.feedback = feedback or []
        self.graph_path = GRAPH_DATA_PATH

        # Initialize models and data structures
        self.topic_hierarchy = TOPIC_HIERARCHY
        self.phase_map = CONVERSATION_PHASE_MAP
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # --- Learning Pipeline ---
        # 1. Load the long-term knowledge graph
        self.transition_graph = self._load_persistent_graph()
        # 2. Build a graph from the current conversation
        session_graph = self._build_transition_graph_from_session()
        # 3. Merge session knowledge into the main graph
        self._merge_graphs(self.transition_graph, session_graph)
        # 4. Apply feedback from the user to refine the graph
        self._apply_feedback()
        # 5. Save the updated graph for future sessions
        self._save_persistent_graph()

    def _load_persistent_graph(self) -> Dict[str, Dict[str, float]]:
        """Loads the transition graph from a JSON file."""
        if os.path.exists(self.graph_path):
            with open(self.graph_path, 'r') as f:
                # The loaded JSON will have string keys, convert inner keys back to defaultdict
                loaded_graph = json.load(f)
                graph = defaultdict(lambda: defaultdict(float))
                for from_topic, to_topics in loaded_graph.items():
                    graph[from_topic] = defaultdict(float, to_topics)
                return graph
        return defaultdict(lambda: defaultdict(float))

    def _save_persistent_graph(self):
        """Saves the transition graph to a JSON file."""
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        with open(self.graph_path, 'w') as f:
            json.dump(self.transition_graph, f, indent=4)

    def _merge_graphs(self, main_graph: Dict, session_graph: Dict):
        """Merges the session graph into the main persistent graph."""
        for from_topic, to_topics in session_graph.items():
            for to_topic, score in to_topics.items():
                main_graph[from_topic][to_topic] += score

    def _apply_feedback(self):
        """
        Adjusts the transition graph weights based on user feedback.
        Chosen suggestions are reinforced.
        """
        for fb in self.feedback:
            if fb.action == "chosen":
                from_topic = fb.current_topic
                to_topic = fb.chosen_suggestion
                # Reinforce the chosen path with a significant boost
                self.transition_graph[from_topic][to_topic] *= 1.2
                self.transition_graph[from_topic][to_topic] += 0.5


    def _get_topic_sequence(self) -> List[str]:
        topic_sequence = []
        if not nlp: return []
        for turn in self.turns:
            content = turn.get('content', '')
            if not content.strip():
                topic_sequence.append(None)
                continue
            doc = nlp(content)
            sentences = [sent.text for sent in doc.sents if len(sent.text.split()) > 2]
            if not sentences:
                topic_sequence.append(None)
                continue
            sentence_vectors = embedder_service.encode_cached(sentences)
            similarity_matrix = cosine_similarity(sentence_vectors, canonical_topic_vectors)
            max_sim_score = np.max(similarity_matrix)
            if max_sim_score > 0.5:
                best_match_index = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)[1]
                topic_sequence.append(canonical_topic_names[best_match_index])
            else:
                topic_sequence.append(None)
        return topic_sequence

    def _build_transition_graph_from_session(self) -> Dict[str, Dict[str, float]]:
        topic_sequence = self._get_topic_sequence()
        graph = defaultdict(lambda: defaultdict(float))
        num_turns = len(self.turns)
        for i in range(num_turns - 1):
            from_topic, to_topic = topic_sequence[i], topic_sequence[i+1]
            if not from_topic or not to_topic: continue
            
            base_score = 1.0
            recency_weight = 0.95 ** (num_turns - 1 - i)
            sentiment_score = self.sentiment_analyzer.polarity_scores(self.turns[i]['content'])['compound']
            sentiment_weight = 1.0 + sentiment_score
            role_weight = 1.2 if self.turns[i].get('role') == 'user' else 1.0
            adjusted_score = base_score * recency_weight * sentiment_weight * role_weight
            graph[from_topic][to_topic] += adjusted_score
        return graph

    def _get_current_phase(self, priority: List[str]) -> str:
        detected_phases = self.analysis.get("detected_phases", ["Icebreaker"])
        for phase in priority:
            if phase in detected_phases: return phase
        return detected_phases[0] if detected_phases else "Icebreaker"

    def get_suggestions(self) -> Dict[str, List[str]]:
        topic_sequence = self._get_topic_sequence()
        current_topic = next((t for t in reversed(topic_sequence) if t is not None), None)
        phase_priority = ["Intimate Topics", "Logistics / Planning", "Deeper Connection", "Flirting", "Personal Interests", "Icebreaker"]
        current_phase = self._get_current_phase(phase_priority)
        sentiment = self.analysis.get("sentiment_analysis", {}).get("overall", "neutral")
        candidate_info = defaultdict(lambda: {'score': 0.0, 'reasons': set()})

        if current_topic and current_topic in self.transition_graph:
            for next_topic, score in self.transition_graph[current_topic].items():
                candidate_info[next_topic]['score'] += score
                candidate_info[next_topic]['reasons'].add(f"It's a natural transition from '{current_topic}'.")
                if sentiment in ["positive", "very positive"]:
                     candidate_info[next_topic]['reasons'].add("The conversation vibe is positive.")

        recency_map = self.analysis.get("conversation_state", {}).get("topic_recency_heatmap", {})
        occurrence_map = self.analysis.get("conversation_state", {}).get("topic_occurrence_heatmap", {})

        for topic in set(canonical_topic_names):
            if current_phase in self.phase_map:
                for theme in self.phase_map[current_phase]["themes"]:
                    if topic in self.topic_hierarchy.get(theme, {}).get("sub_topics", []):
                        candidate_info[topic]['score'] += 0.5
                        candidate_info[topic]['reasons'].add(f"It fits the '{current_phase}' phase.")
            if topic in occurrence_map:
                candidate_info[topic]['score'] += occurrence_map.get(topic, 0) * 0.1
                candidate_info[topic]['reasons'].add("It's a popular topic.")
            if topic in recency_map:
                penalty = (1 - (1 / (recency_map[topic] + 1)))
                candidate_info[topic]['score'] *= penalty
                if penalty < 0.7: candidate_info[topic]['reasons'].add("You just talked about this.")

        if current_topic:
            current_topic_index = canonical_topic_names.index(current_topic)
            current_topic_vector = canonical_topic_vectors[current_topic_index].reshape(1, -1)
            for topic in list(candidate_info.keys()):
                if topic == current_topic:
                    del candidate_info[topic]
                    continue
                topic_index = canonical_topic_names.index(topic)
                topic_vector = canonical_topic_vectors[topic_index].reshape(1, -1)
                similarity = cosine_similarity(current_topic_vector, topic_vector)[0][0]
                if similarity > 0.85:
                    candidate_info[topic]['score'] *= 0.1
                    candidate_info[topic]['reasons'].add("It's very similar to the current topic.")

        if not candidate_info:
            if current_phase in self.phase_map:
                for theme in self.phase_map[current_phase]["themes"]:
                    for topic in self.topic_hierarchy.get(theme, {}).get("sub_topics", []):
                        candidate_info[topic]['score'] += 1.0
                        candidate_info[topic]['reasons'].add(f"A good starting point for the '{current_phase}' phase.")

        sorted_candidates = sorted(candidate_info.items(), key=lambda item: item[1]['score'], reverse=True)
        top_5_suggestions = [{"topic": t, "score": i['score'], "reason": " ".join(list(i['reasons']))} for t, i in sorted_candidates[:5]]

        return self._format_suggestions_for_output(top_5_suggestions)

    def _format_suggestions_for_output(self, suggestions_with_reasons: List[Dict]) -> Dict[str, List[str]]:
        topics = [s['topic'] for s in suggestions_with_reasons]
        suggestions = { "topics": [], "questions": [], "intimacy": [], "sexual": [] }
        topic_mapping = self.analysis.get("conversation_state", {}).get("topic_mapping", {})
        for category in suggestions.keys():
            category_suggestions = set()
            for canonical_topic in topics:
                if len(category_suggestions) >= 5: break
                strategies = SUGGESTION_STRATEGY_MAP.get(canonical_topic, {}).get(category, [])
                if not strategies: continue
                raw_topics = topic_mapping.get(canonical_topic) or [canonical_topic.lower().replace(" & ", " ")]
                strategy = random.choice(strategies)
                template = random.choice(SUGGESTION_TEMPLATES[category][strategy])
                raw_topic_to_use = random.choice(raw_topics)
                suggestion = template.format(topic=raw_topic_to_use)
                if suggestion not in category_suggestions:
                    category_suggestions.add(suggestion)
            suggestions[category] = list(category_suggestions)
        for category, items in suggestions.items():
            if len(items) < 5:
                needed = 5 - len(items)
                fallback_templates = SUGGESTION_TEMPLATES[category]["fallback"]
                for _ in range(needed):
                    suggestion = random.choice(fallback_templates)
                    if "{topic}" in suggestion:
                        suggestion = suggestion.format(topic="your vibe")
                    if suggestion not in items:
                        items.append(suggestion)
        return suggestions

def generate_suggestions(analysis_data: Dict[str, Any], conversation_turns: List[Dict[str, Any]], feedback: Optional[List[Feedback]] = None) -> Dict[str, List[str]]:
    """
    Generates 5 creative, context-aware suggestions for each category.
    This function now delegates the work to the AdvancedSuggestionEngine.
    """
    engine = AdvancedSuggestionEngine(analysis_data, conversation_turns, feedback)
    return engine.get_suggestions()