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
    A sophisticated suggestion engine that uses a weighted transition graph,
    semantic similarity, and conversation phase awareness to generate relevant
    and engaging topic suggestions.
    """
    def __init__(self, analysis_data: Dict[str, Any], conversation_turns: List[Dict[str, Any]]):
        self.analysis = analysis_data
        self.turns = conversation_turns

        # Initialize models and data structures
        self.topic_hierarchy = TOPIC_HIERARCHY
        self.phase_map = CONVERSATION_PHASE_MAP
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Build the weighted topic transition graph
        self.transition_graph = self._build_transition_graph()

    def _get_topic_sequence(self) -> List[str]:
        """
        Processes the conversation turns to identify the dominant canonical topic for each turn.
        Returns a list of topic names in chronological order.
        """
        topic_sequence = []
        if not nlp:
            return []

        for turn in self.turns:
            content = turn.get('content', '')
            if not content.strip():
                topic_sequence.append(None) # No topic for empty turns
                continue

            doc = nlp(content)
            sentences = [sent.text for sent in doc.sents if len(sent.text.split()) > 2]
            if not sentences:
                topic_sequence.append(None)
                continue

            sentence_vectors = embedder_service.encode_cached(sentences)
            similarity_matrix = cosine_similarity(sentence_vectors, canonical_topic_vectors)

            # Find the topic with the maximum similarity score for this turn
            max_sim_score = np.max(similarity_matrix)
            if max_sim_score > 0.5: # Similarity threshold
                best_match_index = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)[1]
                dominant_topic = canonical_topic_names[best_match_index]
                topic_sequence.append(dominant_topic)
            else:
                topic_sequence.append(None) # No dominant topic found

        return topic_sequence

    def _build_transition_graph(self) -> Dict[str, Dict[str, float]]:
        """
        Builds a weighted directed graph of topic transitions from the conversation history.
        The weights are influenced by frequency, recency, and sentiment.
        """
        topic_sequence = self._get_topic_sequence()
        graph = defaultdict(lambda: defaultdict(float))
        num_turns = len(self.turns)

        for i in range(num_turns - 1):
            from_topic = topic_sequence[i]
            to_topic = topic_sequence[i+1]

            # Skip if either topic is None
            if not from_topic or not to_topic:
                continue
            
            # --- Weighting Logic ---
            # 1. Base score for frequency
            base_score = 1.0

            # 2. Recency weight (newer transitions are more important)
            recency_decay = 0.95
            recency_weight = recency_decay ** (num_turns - 1 - i)
            
            # 3. Sentiment weight
            turn_content = self.turns[i]['content']
            sentiment_score = self.sentiment_analyzer.polarity_scores(turn_content)['compound']
            sentiment_weight = 1.0 + sentiment_score # Ranges roughly from 0.0 to 2.0
            
            # 4. Role-based weight (e.g., give more weight to user-initiated topics)
            # Simple implementation: Boost if the user (not assistant) is speaking
            role_weight = 1.2 if self.turns[i].get('role') == 'user' else 1.0

            # Combine weights
            adjusted_score = base_score * recency_weight * sentiment_weight * role_weight
            
            graph[from_topic][to_topic] += adjusted_score

        return graph

    def _get_current_phase(self, priority: List[str]) -> str:
        """
        Determines the single most relevant current phase based on a priority list.
        This helps focus the suggestions on the most advanced stage of the conversation.
        """
        detected_phases = self.analysis.get("detected_phases", ["Icebreaker"])
        for phase in priority:
            if phase in detected_phases:
                return phase
        return detected_phases[0] if detected_phases else "Icebreaker"

    def get_suggestions(self) -> Dict[str, List[str]]:
        """
        Generates contextual topic suggestions based on a sophisticated, multi-factor scoring model.

        The process involves:
        1. Determining the current conversation state (topic, phase, sentiment).
        2. Using a pre-built topic transition graph to find likely next topics.
        3. Adjusting scores based on phase appropriateness, topic popularity, and recency.
        4. Penalizing suggestions that are too semantically similar to the current topic to ensure diversity.
        5. Using a fallback mechanism for conversations that are too short for graph-based analysis.
        6. Ranking the suggestions and formatting them with reasoning for the final output.

        This version now includes reasoning for each suggestion.
        """
        # --- 1. Determine Current State ---
        topic_sequence = self._get_topic_sequence()
        current_topic = next((topic for topic in reversed(topic_sequence) if topic is not None), None)

        phase_priority = ["Intimate Topics", "Logistics / Planning", "Deeper Connection", "Flirting", "Personal Interests", "Icebreaker"]
        current_phase = self._get_current_phase(phase_priority)
        sentiment = self.analysis.get("sentiment_analysis", {}).get("overall", "neutral")
        
        candidate_info = defaultdict(lambda: {'score': 0.0, 'reasons': set()})

        # --- 2. Get Candidates from Transition Graph ---
        if current_topic and current_topic in self.transition_graph:
            for next_topic, score in self.transition_graph[current_topic].items():
                candidate_info[next_topic]['score'] += score
                candidate_info[next_topic]['reasons'].add(f"It's a natural transition from '{current_topic}'.")
                if sentiment == "positive" or sentiment == "very positive":
                     candidate_info[next_topic]['reasons'].add("The conversation vibe is positive.")


        # --- 3. Score Adjustment based on other factors ---
        recency_map = self.analysis.get("conversation_state", {}).get("topic_recency_heatmap", {})
        occurrence_map = self.analysis.get("conversation_state", {}).get("topic_occurrence_heatmap", {})

        all_possible_topics = set(canonical_topic_names)
        for topic in all_possible_topics:
            # Boost topics that fit the current conversation phase
            if current_phase in self.phase_map:
                for theme in self.phase_map[current_phase]["themes"]:
                    if topic in self.topic_hierarchy.get(theme, {}).get("sub_topics", []):
                        candidate_info[topic]['score'] += 0.5
                        candidate_info[topic]['reasons'].add(f"It fits the '{current_phase}' phase.")

            # Boost based on historical popularity
            if topic in occurrence_map:
                candidate_info[topic]['score'] += occurrence_map.get(topic, 0) * 0.1
                candidate_info[topic]['reasons'].add("It's a popular topic.")

            # Penalize very recent topics to avoid repetition
            if topic in recency_map:
                penalty = (1 - (1 / (recency_map[topic] + 1)))
                candidate_info[topic]['score'] *= penalty
                if penalty < 0.7:
                    candidate_info[topic]['reasons'].add("You just talked about this.")


        # --- 4. Semantic Similarity Check for Diversity ---
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

                if similarity > 0.85: # High similarity threshold
                    candidate_info[topic]['score'] *= 0.1
                    candidate_info[topic]['reasons'].add("It's very similar to the current topic.")

        # --- 5. Cold Start / Fallback ---
        if not candidate_info:
            if current_phase in self.phase_map:
                for theme in self.phase_map[current_phase]["themes"]:
                    for topic in self.topic_hierarchy.get(theme, {}).get("sub_topics", []):
                        candidate_info[topic]['score'] += 1.0
                        candidate_info[topic]['reasons'].add(f"A good starting point for the '{current_phase}' phase.")

        # --- 6. Final Ranking & Formatting ---
        sorted_candidates = sorted(candidate_info.items(), key=lambda item: item[1]['score'], reverse=True)

        top_5_suggestions = []
        for topic, info in sorted_candidates[:5]:
            top_5_suggestions.append({
                "topic": topic,
                "score": info['score'],
                "reason": " ".join(list(info['reasons']))
            })

        return self._format_suggestions_for_output(top_5_suggestions)

    def _format_suggestions_for_output(self, suggestions_with_reasons: List[Dict]) -> Dict[str, List[str]]:
        """Formats the rich suggestion list into the legacy structure for the UI."""
        topics = [s['topic'] for s in suggestions_with_reasons]
        suggestions = { "topics": [], "questions": [], "intimacy": [], "sexual": [] }
        topic_mapping = self.analysis.get("conversation_state", {}).get("topic_mapping", {})

        for category in suggestions.keys():
            category_suggestions = set()
            for canonical_topic in topics:
                if len(category_suggestions) >= 5: break

                strategies = SUGGESTION_STRATEGY_MAP.get(canonical_topic, {}).get(category, [])
                if not strategies: continue

                raw_topics = topic_mapping.get(canonical_topic)
                if not raw_topics:
                    raw_topics = [canonical_topic.lower().replace(" & ", " ")]

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


def generate_suggestions(analysis_data: Dict[str, Any], conversation_turns: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Generates 5 creative, context-aware suggestions for each category.
    This function now delegates the work to the AdvancedSuggestionEngine.
    """
    engine = AdvancedSuggestionEngine(analysis_data, conversation_turns)
    return engine.get_suggestions()