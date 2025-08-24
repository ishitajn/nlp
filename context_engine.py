# In context_engine.py
import re
from typing import List, Dict, Any
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ANALYSIS_SCHEMA = {
    "phases": { "Icebreaker": [r'\b(h(i|ey|ello)|how are you|your profile|we matched)\b'], "Rapport Building": [r'\b(tell me more|what about you|hobbies|passions|family|career|work|job|hiking|trip|travel)\b'], "Escalation": [r'\b(tension|desire|imagining|in person|what if|chemistry)\b'], "Explicit Banter": [r'\b(fuck|sex|nude|kink|sexting|horny|aroused)\b'], "Logistics": [r'\b(when are you free|let\'s meet|what\'s your number|schedule|date)\b'], },
    "tones": { "Playful": [r'\b(haha|lol|lmao|kidding|teasing|banter|playful|cheeky)\b', r'[😉😜😏]'], "Serious": [r'\b(to be honest|actually|my values|looking for|seriously)\b'], "Romantic": [r'\b(connection|special|beautiful|chemistry|heart|adore|lovely)\b'], "Complimentary": [r'\b(great|amazing|impressive|gorgeous|handsome|hot|sexy|cute)\b'], "Vulnerable": [r'\b(my feelings|i feel|struggle|opening up is hard|i feel safe with you)\b'], },
    "intents": { "Gathering Information": [r'\?'], "Building Comfort": [r'\b(that makes sense|i understand|thank you for sharing)\b'], "Testing Boundaries": [r'\b(what are you into|how adventurous|are you open to)\b'], "Making Plans": [r'\b(we should|let\'s|are you free|wanna grab)\b'], "Expressing Desire": [r'\b(i want you|i need you|can\'t stop thinking about you|i desire you)\b'], }
}

sentiment_analyzer = SentimentIntensityAnalyzer()

def extract_contextual_features(
    conversation_turns: List[Dict[str, Any]],
    identified_topics: List[Dict[str, Any]],
    my_profile: str = "",
    their_profile: str = ""
) -> Dict[str, Any]:
    """
    Analyzes the conversation to extract contextual features like sentiment, phase, etc.
    """
    # --- 1. Prepare Full Text for Rule-Based Matching ---
    conversation_history_str = "\n".join([t.get('content', '') for t in conversation_turns])
    full_text_for_rules = f"{my_profile} {their_profile} {conversation_history_str}"
    full_text_lower = full_text_for_rules.lower()

    # --- 2. Schema-Based Tagging (Phases, Tones, Intents) ---
    detected_tags = { "detected_phases": set(), "detected_tones": set(), "detected_intents": set() }
    for category, rules in ANALYSIS_SCHEMA.items():
        for tag_name, patterns in rules.items():
            if any(re.search(pattern, full_text_lower) for pattern in patterns):
                detected_tags[f"detected_{category}"].add(tag_name)

    # Ensure at least one phase is detected
    if not detected_tags["detected_phases"]:
        detected_tags["detected_phases"].add("Rapport Building") # Default phase

    analysis = {k: list(v) for k, v in detected_tags.items()}

    # --- 3. Quantitative & Sentiment Analysis ---
    sentiment_scores = sentiment_analyzer.polarity_scores(conversation_history_str)
    compound_score = sentiment_scores['compound']
    sentiment = "neutral"
    if compound_score > 0.5: sentiment = "very positive"
    elif compound_score > 0.05: sentiment = "positive"
    elif compound_score < -0.5: sentiment = "very negative"
    elif compound_score < -0.05: sentiment = "negative"

    analysis["sentiment_analysis"] = { "overall": sentiment, "compound_score": compound_score }

    # --- 4. Topic-based Features (Recency and Salience) ---
    topic_saliency = Counter()
    for topic in identified_topics:
        topic_saliency[topic['canonical_name']] += len(topic.get("message_turns", []))

    topic_recency = {}
    turn_count = len(conversation_turns)
    for topic in identified_topics:
        last_turn_index = -1
        # Find the last turn index for the current topic
        for i, turn in enumerate(conversation_turns):
            if turn in topic["message_turns"]:
                last_turn_index = i

        if last_turn_index != -1:
            # Higher recency score for more recent topics
            topic_recency[topic['canonical_name']] = turn_count - last_turn_index

    # Sort by recency score (lower is more recent)
    sorted_recent_topics = sorted(topic_recency.items(), key=lambda item: item[1])

    analysis["topic_saliency"] = dict(topic_saliency.most_common(10))

    # Assign ranks, handling ties correctly (e.g., 1, 1, 3, 4)
    ranked_recency = {}
    last_score = -1
    last_rank = 0
    for i, (topic, score) in enumerate(sorted_recent_topics):
        if score > last_score:
            last_rank = i + 1
        ranked_recency[topic] = last_rank
        last_score = score
    analysis["topic_recency"] = ranked_recency


    # --- 5. Speaker Roles and Turn Position ---
    # This information is already in the `conversation_turns` objects and can be used directly
    # by the suggestion engine. We can add aggregated stats here if needed.
    user_turn_count = sum(1 for turn in conversation_turns if turn.get('role') == 'user')
    their_turn_count = sum(1 for turn in conversation_turns if turn.get('role') == 'assistant')

    analysis["speaker_metrics"] = {
        "user_turn_count": user_turn_count,
        "their_turn_count": their_turn_count,
        "turn_count": len(conversation_turns)
    }

    # --- 6. Engagement, Pace, and Flirtation Metrics ---
    num_turns = len(conversation_turns)
    question_count = conversation_history_str.count('?')
    avg_msg_len = len(conversation_history_str) / num_turns if num_turns > 0 else 0

    engagement = "low"
    if num_turns > 4 and question_count > 2: engagement = "medium"
    if num_turns > 8 and avg_msg_len > 50 and question_count > 4: engagement = "high"
    if num_turns > 12 and avg_msg_len > 70 and question_count > 6: engagement = "very high"

    pace = "steady"
    if avg_msg_len < 40 and num_turns > 10: pace = "fast"
    if avg_msg_len > 120: pace = "slow and thoughtful"
    if "Escalation" in analysis["detected_phases"]: pace = "steady with potential for escalation"

    flirt_keywords = ['flirt', 'teasing', 'sexy', 'hot', 'desire', 'tension', 'imagining', 'irresistible', '😉', '😏', 'cuddle', 'kiss']
    flirt_score = sum(full_text_lower.count(kw) for kw in flirt_keywords)
    flirtation_level = "low"
    if flirt_score > 5 or "Explicit Banter" in analysis.get("detected_phases", []): flirtation_level = "very high"
    elif flirt_score > 2 or "Escalation" in analysis.get("detected_phases", []): flirtation_level = "high"
    elif flirt_score > 0: flirtation_level = "medium"

    analysis["engagement_metrics"] = { "level": engagement, "pace": pace, "flirtation_level": flirtation_level }

    return analysis
