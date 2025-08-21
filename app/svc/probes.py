import re
from typing import List, Dict, Any

# --- Basic Probes (Rule-Based) ---

def is_greeting(text: str) -> bool:
    """Detects common greetings."""
    return bool(re.search(r'\b(hello|hi|hey|yo|sup)\b', text, re.IGNORECASE))

def is_question(text: str) -> bool:
    """Detects if a text is a question."""
    return text.strip().endswith('?')

def has_pet_keywords(text: str) -> bool:
    """Detects keywords related to pets."""
    return bool(re.search(r'\b(dog|cat|pet|puppy|kitten)\b', text, re.IGNORECASE))

def has_location_keywords(text: str) -> bool:
    """Detects keywords related to location."""
    return bool(re.search(r'\b(live|from|place|area|location|city)\b', text, re.IGNORECASE))

def has_disclosure(text: str) -> bool:
    """Detects 'I' statements, a simple proxy for self-disclosure."""
    return bool(re.search(r'\b(i am|i\'m|i like|i love|i enjoy|my favorite)\b', text, re.IGNORECASE))

def get_response_length_class(text: str) -> str:
    """Classifies response length."""
    word_count = len(text.split())
    if word_count < 5:
        return "short"
    elif word_count < 20:
        return "medium"
    else:
        return "long"

# --- Advanced Probes (Placeholders for ML Models) ---
# These would be replaced by trained sklearn LogisticRegression models or similar.

def detect_flirtation(text: str) -> bool:
    """Placeholder for flirtation detection model."""
    # A real implementation would use a trained classifier.
    # Simple keyword matching is often inaccurate for nuanced concepts.
    return bool(re.search(r'\b(wink|cute|hot|beautiful|sexy)\b', text, re.IGNORECASE))

def detect_sexual_tone(text: str) -> bool:
    """Placeholder for sexual tone detection model."""
    # This requires a carefully trained and calibrated model.
    return False

def detect_playful_energy(text: str) -> bool:
    """Placeholder for playful energy detection."""
    # Could look for banter, jokes, multiple exclamation marks, certain emojis, etc.
    return bool(re.search(r'(haha|lol|!{2,})', text))


# --- Main Evaluation Engine ---

def evaluate_features(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates a set of features and probes for the given conversation turns.
    This corresponds to the "Feature & Probe Engine".
    """
    if not turns:
        return {}

    last_turn = turns[-1]
    last_text = last_turn.get("content", "")

    # Engagement signals for the most recent turn
    engagement_signals = {
        "reciprocity": is_question(last_text), # Simplified: assumes asking a question back is reciprocity
        "disclosure": has_disclosure(last_text),
        "question_asking": is_question(last_text),
        "response_length_class": get_response_length_class(last_text)
    }

    # Sexual/Intimacy context flags
    sexual_intimacy_flags = {
        "flirtation_detected": detect_flirtation(last_text),
        "sexual_tone": detect_sexual_tone(last_text),
        "playful_energy": detect_playful_energy(last_text),
        "comfort_level": "medium",  # Placeholder - would need a more complex model
        "escalation_readiness": "early" # Placeholder
    }

    # General probes applied to the last message
    general_probes = {
        "is_greeting": is_greeting(last_text),
        "is_question": is_question(last_text),
        "is_pet_related": has_pet_keywords(last_text),
        "is_location_related": has_location_keywords(last_text),
    }

    # Aggregate features across the conversation
    # For example, count total questions from each participant
    user_questions = 0
    assistant_questions = 0
    for turn in turns:
        if is_question(turn.get("content", "")):
            if turn["role"] == "user":
                user_questions += 1
            else:
                assistant_questions += 1

    # Robust engagement level
    # This is a simplified logic based on available signals
    engagement_level = "low"
    if engagement_signals["disclosure"] and engagement_signals["response_length_class"] in ["medium", "long"]:
        engagement_level = "medium"
    if engagement_level == "medium" and user_questions > 0 and assistant_questions > 0:
        engagement_level = "high"

    match_engagement = {
        "level": engagement_level,
        "indicators": [k for k, v in engagement_signals.items() if v]
    }

    return {
        "engagement_signals": engagement_signals,
        "sexual_intimacy_flags": sexual_intimacy_flags,
        "general_probes": general_probes,
        "match_engagement_level": match_engagement,
        "conversation_stats": {
            "user_question_count": user_questions,
            "assistant_question_count": assistant_questions
        }
    }
