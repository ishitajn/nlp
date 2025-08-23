import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Define keyword lists for various checks
GREETING_KEYWORDS = [r'\b(h(i|ey|ello)|yo|sup|wassup)\b']
FLIRT_KEYWORDS = [r'\b(cute|hot|sexy|beautiful|gorgeous|kiss|cuddle|desire|irresistible|captivating|wink|😉|😏)\b']
TIME_KEYWORDS = [r'\b(today|tonight|tomorrow|weekend|week|day|date|morning|afternoon|evening|night|when)\b']
LOCATION_KEYWORDS = [r'\b(place|area|neighborhood|city|country|location|distance|close|far|meet|here|there)\b']
FALLBACK_KEYWORDS = {
    "idk": r'\b(idk|i don\'?t know)\b',
    "lol": r'\b(lol|lmao|haha|hehe)\b',
    "maybe": r'\b(maybe|perhaps|possibly|we\'?ll see)\b'
}

def _parse_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """Safely parse an ISO 8601 timestamp string."""
    if not ts_str:
        return None
    try:
        # Handle 'Z' suffix for UTC
        if ts_str.endswith('Z'):
            ts_str = ts_str[:-1] + '+00:00'
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None

def analyze_conversation_behavior(
    conversation_turns: List[Dict[str, Any]],
    identified_topics: List[Dict[str, Any]]
) -> Dict[str, Any]:

    if not conversation_turns:
        return {}

    # --- Initialize variables ---
    last_user_turn: Optional[Dict] = None
    last_match_turn: Optional[Dict] = None
    last_turn: Optional[Dict] = conversation_turns[-1]

    for turn in reversed(conversation_turns):
        sender = turn.get('sender', 'unknown').lower()
        if sender == 'user' and not last_user_turn:
            last_user_turn = turn
        if sender != 'user' and not last_match_turn:
            last_match_turn = turn
        if last_user_turn and last_match_turn:
            break

    # --- Basic Last Message Info ---
    analysis = {
        "last_message_from_user": last_user_turn.get('content') if last_user_turn else None,
        "last_message_from_match": last_match_turn.get('content') if last_match_turn else None,
        "Last_message_from": last_turn.get('sender') if last_turn else None,
    }

    # --- Last Match Message Analysis ---
    match_last_message_content = (analysis['last_message_from_match'] or "").lower()
    analysis['match_last_message_has_question'] = '?' in match_last_message_content
    analysis['Match_last_message_geo_context'] = any(re.search(p, match_last_message_content) for p in TIME_KEYWORDS + LOCATION_KEYWORDS)

    # --- Time-based Analysis ---
    now = datetime.utcnow()
    two_days_ago = now - timedelta(days=2)
    one_day_ago = now - timedelta(days=1)
    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)

    analysis['last_user_greeted'] = False
    user_active_recently = False
    match_active_recently = False

    for turn in conversation_turns:
        turn_time = _parse_timestamp(turn.get('timestamp'))
        if not turn_time: continue

        sender = turn.get('sender', 'unknown').lower()
        content = turn.get('content', '').lower()

        if sender == 'user':
            if turn_time > one_day_ago:
                user_active_recently = True
            if turn_time > two_days_ago and any(re.search(p, content) for p in GREETING_KEYWORDS):
                analysis['last_user_greeted'] = True
        else: # Match
            if turn_time > one_day_ago:
                match_active_recently = True

    # --- Conversation Flags ---
    flags = {}
    flags['user_active_recently'] = user_active_recently
    flags['match_active_recently'] = match_active_recently

    # Last message day classification
    last_turn_time = _parse_timestamp(last_turn.get('timestamp')) if last_turn else None
    if not last_turn_time:
        flags['Last_message_day'] = "Unknown"
    elif len(conversation_turns) < 5:
        flags['Last_message_day'] = "New Conversation"
    elif last_turn_time > two_days_ago:
        flags['Last_message_day'] = "Active Conversation"
    elif last_turn_time > seven_days_ago:
        flags['Last_message_day'] = "Recent Conversation"
    elif last_turn_time > thirty_days_ago:
        flags['Last_message_day'] = "Stale Conversation"
    else:
        flags['Last_message_day'] = "Old Conversation"

    # Last topic category
    last_topic = identified_topics[0] if identified_topics else {}
    flags['last_topic_category'] = last_topic.get('category', 'N/A')

    # Last message content analysis
    last_message_content = last_turn.get('content', '').lower() if last_turn else ""
    flags['contains_fallback_keywords'] = [name for name, pattern in FALLBACK_KEYWORDS.items() if re.search(pattern, last_message_content)]
    flags['greeting_detected'] = any(re.search(p, last_message_content) for p in GREETING_KEYWORDS)
    flags['flirtation_indicator'] = any(re.search(p, last_message_content) for p in FLIRT_KEYWORDS)
    flags['time_reference_detected'] = any(re.search(p, last_message_content) for p in TIME_KEYWORDS)
    flags['location_reference_detected'] = any(re.search(p, last_message_content) for p in LOCATION_KEYWORDS)

    # Recent engagement score
    recent_turns = conversation_turns[-5:]
    question_count = sum(1 for t in recent_turns if '?' in t.get('content', ''))
    avg_len = sum(len(t.get('content', '')) for t in recent_turns) / len(recent_turns) if recent_turns else 0
    if question_count > 1 or avg_len > 80:
        flags['recent_engagement_score'] = "high"
    elif question_count > 0 or avg_len > 40:
        flags['recent_engagement_score'] = "medium"
    else:
        flags['recent_engagement_score'] = "low"

    analysis['conversation_flags'] = flags

    # --- Suggestion Flags ---
    suggestion_flags = {}
    suggestion_flags['suggest_follow_up_question'] = not analysis['match_last_message_has_question']
    suggestion_flags['suggest_flirtation'] = not flags['flirtation_indicator'] and flags['last_topic_category'] in ['Flirting', 'Deeper Connection', 'Hobbies & Interests']
    suggestion_flags['suggest_topic_shift'] = flags['recent_engagement_score'] == 'low' or bool(flags['contains_fallback_keywords'])
    suggestion_flags['suggest_greeting'] = not analysis['last_user_greeted'] and not user_active_recently

    analysis['conversation_suggestion_flags'] = suggestion_flags

    return analysis
