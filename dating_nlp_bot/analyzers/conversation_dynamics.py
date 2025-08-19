from datetime import datetime, timedelta
from . import adult_content

GREETING_KEYWORDS = ["hello", "hi", "hey", "yo", "sup"]

def _get_common_dynamics(conversation_history: list[dict]) -> dict:
    """
    Calculates the common dynamics for both fast and enhanced modes.
    """
    if not conversation_history:
        return {
            "question_detected": False, "recent_greeting": False, "pace": "slow",
            "stage": "starting", "is_engaged": False, "reciprocity_balance": "balanced"
        }

    # Reciprocity
    user_messages = sum(1 for msg in conversation_history if msg['role'] == 'user')
    match_messages = sum(1 for msg in conversation_history if msg['role'] == 'assistant')
    if user_messages > match_messages * 1.5:
        reciprocity_balance = "user_dominant"
    elif match_messages > user_messages * 1.5:
        reciprocity_balance = "match_dominant"
    else:
        reciprocity_balance = "balanced"

    # Last message details
    last_message = conversation_history[-1]
    last_message_text = last_message.get("content", "").lower()
    last_message_date = datetime.fromisoformat(last_message.get("date"))

    # Question detection (in last match message)
    last_match_message = next((msg for msg in reversed(conversation_history) if msg['role'] == 'assistant'), None)
    question_detected = '?' in last_match_message['content'] if last_match_message else False

    # Recent greeting
    recent_greeting = any(greet in last_message_text for greet in GREETING_KEYWORDS) and (datetime.now() - last_message_date < timedelta(days=1))

    # Pace
    if len(conversation_history) > 1:
        time_diffs = [(datetime.fromisoformat(conversation_history[i]['date']) - datetime.fromisoformat(conversation_history[i-1]['date'])).total_seconds() for i in range(1, len(conversation_history))]
        avg_diff = sum(time_diffs) / len(time_diffs)
        pace = "fast" if avg_diff < 3600 else "slow" if avg_diff > 86400 else "balanced"
    else:
        pace = "slow"

    # Stage
    time_since_last_message = datetime.now() - last_message_date
    if len(conversation_history) < 3:
        stage = "starting"
    elif time_since_last_message < timedelta(days=2):
        stage = "active"
    elif time_since_last_message < timedelta(days=7):
        stage = "break_2_days"
    else:
        stage = "break_over_month"

    # Engagement
    is_engaged = pace == "fast" and reciprocity_balance == "balanced" and stage == "active"

    return {
        "question_detected": question_detected,
        "recent_greeting": recent_greeting,
        "pace": pace,
        "stage": stage,
        "is_engaged": is_engaged,
        "reciprocity_balance": reciprocity_balance,
    }

def analyze_dynamics_fast(conversation_history: list[dict]) -> dict:
    """
    Analyzes conversation dynamics using rule-based methods (fast mode).
    """
    dynamics = _get_common_dynamics(conversation_history)
    adult_results = adult_content.analyze_adult_content_fast(conversation_history)
    dynamics.update(adult_results)
    return dynamics

def analyze_dynamics_enhanced(conversation_history: list[dict]) -> dict:
    """
    Analyzes conversation dynamics using rule-based methods and enhanced adult content analysis.
    """
    dynamics = _get_common_dynamics(conversation_history)
    adult_results = adult_content.analyze_adult_content_enhanced(conversation_history)
    dynamics.update(adult_results)
    return dynamics
