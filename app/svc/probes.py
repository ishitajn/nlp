import re
from typing import List, Dict, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Sentiment Analysis ---
sentiment_analyzer = SentimentIntensityAnalyzer()

def analyze_overall_sentiment(turns: List[Dict[str, Any]]) -> str:
    """
    Analyzes the overall sentiment of the conversation using VADER.
    """
    if not turns:
        return "neutral"

    full_conversation_text = " ".join([t.get("content", "") for t in turns])
    sentiment_scores = sentiment_analyzer.polarity_scores(full_conversation_text)

    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    else:
        return "neutral"

# --- Basic Probes (Rule-Based) from previous implementation ---

def is_question(text: str) -> bool:
    return text.strip().endswith('?')

def has_location_keywords(text: str) -> bool:
    return bool(re.search(r'\b(live|from|place|area|location|city)\b', text, re.IGNORECASE))

def is_greeting(text: str) -> bool:
    return bool(re.search(r'\b(hello|hi|hey|yo|sup)\b', text, re.IGNORECASE))


# --- Main Evaluation Engine (Refactored for new schema) ---

def evaluate_features(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates a set of features and probes to align with the new, detailed schema.
    Many of the more advanced metrics are returned as placeholders.
    """
    if not turns:
        return {}

    last_turn = turns[-1]
    last_text = last_turn.get("content", "")
    last_role = last_turn.get("role")

    # Determine who sent the last message for the 'analysis.last_response' field.
    # The payload uses 'user' and 'assistant'. The new schema uses 'user' and 'match'.
    # I will map 'assistant' to 'match'.
    last_responder = "user" if last_role == "user" else "match"

    # --- Constructing the 'analysis' object ---
    analysis_output = {
        # 'last_response' and 'last_match_response'
        "last_response": last_responder,
        "last_match_response": {
            "contains_question": is_question(last_text) if last_responder == 'match' else False,
            "related_to_location": has_location_keywords(last_text) if last_responder == 'match' else False,
        },

        # Engagement & Dynamics Keys (mostly placeholders)
        "match_engaged": "medium",  # Placeholder
        "comfort_level": "medium",  # Placeholder
        "escalation_readiness": "early",  # Placeholder
        "recent_greeting_used": is_greeting(turns[-1]['content']) or (len(turns) > 1 and is_greeting(turns[-2]['content'])),
        "conversation_pace": "balanced",  # Placeholder
        "reciprocity_balance": "balanced",  # Placeholder
        "flirtation_level": "low",  # Placeholder
        "sexual_response_allowed": False,  # Placeholder

        # Stylistic / Messaging Keys (all placeholders)
        "length": 80,
        "tone": 50,
        "linguistic_style": "casual",
        "emoji_strategy": "auto",
        "suggested_next_action": "MAINTAIN_ENGAGEMENT",
        "sexual_communication_style": "casual_and_flirty",
        "date_arc_phase": "rapport_building",
        "suggested_response_style": "thoughtful"
    }

    # --- Constructing the 'sentiment' object ---
    sentiment_output = {
        "overall": analyze_overall_sentiment(turns)
    }

    # The function will now return a dictionary with keys that match the assembler's expectation
    return {
        "analysis": analysis_output,
        "sentiment": sentiment_output
    }
