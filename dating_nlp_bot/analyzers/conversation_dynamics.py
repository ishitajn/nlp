from datetime import datetime, timedelta
from .adult_content import analyze_adult_content_fast, analyze_adult_content_enhanced

GREETING_KEYWORDS = ["hello", "hi", "hey", "yo", "sup"]

def analyze_dynamics_enhanced(conversation_history: list[dict]) -> dict:
    """
    Analyzes conversation dynamics using rule-based methods and enhanced adult content analysis.
    """
    # Most of the logic is the same as the fast version,
    # but it calls the enhanced adult content analyzer.
    fast_results = analyze_dynamics_fast(conversation_history)
    adult_content_results = analyze_adult_content_enhanced(conversation_history)
    fast_results.update(adult_content_results)
    return fast_results

def analyze_dynamics_fast(conversation_history: list[dict]) -> dict:
    """
    Analyzes conversation dynamics using rule-based methods (fast mode).
    """
    if not conversation_history:
        return {
            "question_detected": False, "recent_greeting": False, "pace": "slow",
            "stage": "starting", "is_engaged": False, "reciprocity_balance": "balanced",
            "flirtation_level": "low", "sexualResponseSuggestion": False
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
    question_detected = '?' in last_message_text if last_message['role'] == 'assistant' else False

    # Recent greeting
    recent_greeting = any(greet in last_message_text for greet in GREETING_KEYWORDS) and (datetime.now() - last_message_date < timedelta(days=1))

    # Pace
    if len(conversation_history) > 1:
        time_diffs = []
        for i in range(1, len(conversation_history)):
            date1 = datetime.fromisoformat(conversation_history[i-1]['date'])
            date2 = datetime.fromisoformat(conversation_history[i]['date'])
            time_diffs.append((date2 - date1).total_seconds())

        avg_diff = sum(time_diffs) / len(time_diffs)
        if avg_diff < 3600:  # < 1 hour
            pace = "fast"
        elif avg_diff > 86400:  # > 1 day
            pace = "slow"
        else:
            pace = "balanced"
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
    elif time_since_last_message < timedelta(days=30):
        stage = "break_1_week"
    else:
        stage = "break_over_month"

    # Engagement
    is_engaged = pace == "fast" and reciprocity_balance == "balanced" and stage == "active"

    # Adult content analysis
    adult_content_results = analyze_adult_content_fast(conversation_history)

    return {
        "question_detected": question_detected,
        "recent_greeting": recent_greeting,
        "pace": pace,
        "stage": stage,
        "is_engaged": is_engaged,
        "reciprocity_balance": reciprocity_balance,
        "flirtation_level": adult_content_results["flirtation_level"],
        "sexualResponseSuggestion": adult_content_results["sexualResponseSuggestion"],
    }
