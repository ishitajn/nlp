# In behavioral_engine.py
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

from embedder import embedder_service
from semantic_concepts import CONCEPT_EMBEDDINGS

# --- Constants ---
# Thresholds for deciding if a message matches a semantic concept.
SIMILARITY_THRESHOLDS = {
    "GREETING": 0.70,
    "ASKING_A_QUESTION": 0.65,
    "FLIRTATION": 0.60,
    "TIME_REFERENCE": 0.60,
    "LOCATION_REFERENCE": 0.60,
    "DISENGAGEMENT": 0.55,
    "PLANNING_LOGISTICS": 0.65
}

def _parse_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """Safely parse a timestamp string from a few common formats."""
    if not ts_str or not isinstance(ts_str, str):
        return None
    if ts_str.endswith('Z'):
        ts_str = ts_str[:-1] + '+00:00'
    formats_to_try = ["%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"]
    for fmt in formats_to_try:
        try:
            return datetime.strptime(ts_str, fmt)
        except (ValueError, TypeError):
            continue
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        pass
    return None

def _check_semantic_similarity(text_embedding: np.ndarray, concept_name: str) -> bool:
    """Checks if a text embedding is similar to a pre-computed concept embedding."""
    concept_embedding = CONCEPT_EMBEDDINGS.get(concept_name)
    if concept_embedding is None or text_embedding is None or not hasattr(text_embedding, 'reshape'):
        return False

    similarity = cosine_similarity(text_embedding.reshape(1, -1), concept_embedding.reshape(1, -1))[0][0]
    return similarity > SIMILARITY_THRESHOLDS.get(concept_name, 0.6)

def analyze_conversation_behavior(
    conversation_turns: List[Dict[str, Any]],
    identified_topics: List[Dict[str, Any]]
) -> Dict[str, Any]:

    if not conversation_turns:
        return {}

    # --- Pre-process turns for robust sender and embedding info ---
    has_sender_key = any('sender' in turn for turn in conversation_turns)
    if not has_sender_key:
        for i, turn in enumerate(conversation_turns):
            turn['sender'] = 'user' if (i % 2) != 0 else 'match'

    all_contents = [turn.get('content', '') for turn in conversation_turns]
    all_embeddings = embedder_service.encode_cached(all_contents)
    for i, turn in enumerate(conversation_turns):
        turn['embedding'] = all_embeddings[i] if i < len(all_embeddings) else None

    # --- Initialize variables ---
    last_user_turn: Optional[Dict] = None
    last_match_turn: Optional[Dict] = None
    last_turn: Optional[Dict] = conversation_turns[-1]

    for turn in reversed(conversation_turns):
        sender = turn.get('sender', 'match').lower()
        if sender == 'user' and not last_user_turn: last_user_turn = turn
        if sender != 'user' and not last_match_turn: last_match_turn = turn
        if last_user_turn and last_match_turn: break

    # --- Basic Last Message Info ---
    analysis = {
        "last_message_from_user": last_user_turn.get('content') if last_user_turn else None,
        "last_message_from_match": last_match_turn.get('content') if last_match_turn else None,
        "Last_message_from": last_turn.get('sender') if last_turn else None,
    }

    # --- Semantic Analysis of Last Match Message ---
    last_match_embedding = last_match_turn.get('embedding') if last_match_turn else None
    analysis['match_last_message_has_question'] = _check_semantic_similarity(last_match_embedding, "ASKING_A_QUESTION")
    analysis['Match_last_message_geo_context'] = _check_semantic_similarity(last_match_embedding, "LOCATION_REFERENCE") or \
                                                 _check_semantic_similarity(last_match_embedding, "TIME_REFERENCE")

    # --- Time-based Analysis (Remains Rule-Based as per assumption) ---
    now = datetime.utcnow()
    one_day_ago = now - timedelta(days=1)
    two_days_ago = now - timedelta(days=2)
    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)

    analysis['last_user_greeted'] = False
    user_active_recently = False
    match_active_recently = False

    for turn in conversation_turns:
        turn_time = _parse_timestamp(turn.get('timestamp'))
        if not turn_time: continue

        sender = turn.get('sender', 'unknown').lower()
        if sender == 'user':
            if turn_time > one_day_ago: user_active_recently = True
            if turn_time > two_days_ago and _check_semantic_similarity(turn.get('embedding'), "GREETING"):
                analysis['last_user_greeted'] = True
        else:
            if turn_time > one_day_ago: match_active_recently = True

    # --- Conversation Flags ---
    flags = {}
    flags['user_active_recently'] = user_active_recently
    flags['match_active_recently'] = match_active_recently

    last_turn_time = _parse_timestamp(last_turn.get('timestamp')) if last_turn else None
    if not last_turn_time: flags['Last_message_day'] = "Unknown"
    elif len(conversation_turns) < 5: flags['Last_message_day'] = "New Conversation"
    elif last_turn_time > two_days_ago: flags['Last_message_day'] = "Active Conversation"
    elif last_turn_time > seven_days_ago: flags['Last_message_day'] = "Recent Conversation"
    elif last_turn_time > thirty_days_ago: flags['Last_message_day'] = "Stale Conversation"
    else: flags['Last_message_day'] = "Old Conversation"

    last_topic = identified_topics[0] if identified_topics else {}
    flags['last_topic_category'] = last_topic.get('category', 'N/A')

    last_turn_embedding = last_turn.get('embedding') if last_turn else None
    flags['contains_fallback_keywords'] = ["disengagement"] if _check_semantic_similarity(last_turn_embedding, "DISENGAGEMENT") else []
    flags['greeting_detected'] = _check_semantic_similarity(last_turn_embedding, "GREETING")
    flags['flirtation_indicator'] = _check_semantic_similarity(last_turn_embedding, "FLIRTATION")
    flags['time_reference_detected'] = _check_semantic_similarity(last_turn_embedding, "TIME_REFERENCE")
    flags['location_reference_detected'] = _check_semantic_similarity(last_turn_embedding, "LOCATION_REFERENCE")

    recent_turns = conversation_turns[-5:]
    question_count = sum(1 for t in recent_turns if _check_semantic_similarity(t.get('embedding'), "ASKING_A_QUESTION"))
    if question_count > 1: flags['recent_engagement_score'] = "high"
    elif question_count > 0: flags['recent_engagement_score'] = "medium"
    else: flags['recent_engagement_score'] = "low"

    analysis['conversation_flags'] = flags

    # --- Suggestion Flags ---
    suggestion_flags = {}
    suggestion_flags['suggest_follow_up_question'] = not analysis['match_last_message_has_question']
    suggestion_flags['suggest_flirtation'] = not flags['flirtation_indicator'] and flags['last_topic_category'] in ['Flirting', 'Deeper Connection', 'Hobbies & Interests']
    suggestion_flags['suggest_topic_shift'] = flags['recent_engagement_score'] == 'low' or bool(flags['contains_fallback_keywords'])
    suggestion_flags['suggest_greeting'] = not analysis['last_user_greeted'] and not user_active_recently

    analysis['conversation_suggestion_flags'] = suggestion_flags

    return analysis
