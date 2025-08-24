# In behavioral_engine.py
import re
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

from embedder import embedder_service
from semantic_concepts import CONCEPT_EMBEDDINGS

# --- Constants ---
SIMILARITY_THRESHOLDS = {
    "GREETING": 0.70, "ASKING_A_QUESTION": 0.60, "FLIRTATION": 0.60,
    "TIME_REFERENCE": 0.60, "LOCATION_REFERENCE": 0.60, "DISENGAGEMENT": 0.55,
    "PLANNING_LOGISTICS": 0.65
}

def _parse_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    if not ts_str or not isinstance(ts_str, str): return None
    if ts_str.endswith('Z'): ts_str = ts_str[:-1] + '+00:00'
    formats_to_try = ["%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
    for fmt in formats_to_try:
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except (ValueError, TypeError): continue
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except (ValueError, TypeError): pass
    return None

def _check_semantic_similarity(text: str, text_embedding: np.ndarray, concept_name: str) -> bool:
    if concept_name == "ASKING_A_QUESTION":
        if '?' in text.strip(): return True
        question_starters = r'^(who|what|where|when|why|how|is|are|do|does|did|will|can|could|should|would|have|has|had|am|was|were|don\'t|isn\'t|aren\'t)\b'
        if re.match(question_starters, text.strip(), re.IGNORECASE): return True

    concept_embedding = CONCEPT_EMBEDDINGS.get(concept_name)
    if concept_embedding is None or text_embedding is None or not hasattr(text_embedding, 'reshape'): return False

    similarity = cosine_similarity(text_embedding.reshape(1, -1), concept_embedding.reshape(1, -1))[0][0]
    threshold = SIMILARITY_THRESHOLDS.get(concept_name, 0.55 if concept_name == "ASKING_A_QUESTION" else 0.6)
    return bool(similarity > threshold)

def analyze_conversation_behavior(
    conversation_turns: List[Dict[str, Any]]
) -> Dict[str, Any]:
    if not conversation_turns: return {}

    all_contents = [turn.get('content', '') for turn in conversation_turns]
    all_embeddings = embedder_service.encode_cached(all_contents)
    for i, turn in enumerate(conversation_turns):
        turn['embedding'] = all_embeddings[i] if i < len(all_embeddings) else None

    last_user_turn: Optional[Dict] = None
    last_match_turn: Optional[Dict] = None
    for turn in reversed(conversation_turns):
        role = turn.get('role', 'assistant').lower()
        if role == 'user' and not last_user_turn: last_user_turn = turn
        elif role == 'assistant' and not last_match_turn: last_match_turn = turn
        if last_user_turn and last_match_turn: break

    last_turn = conversation_turns[-1]
    last_message_from = 'match' if last_turn.get('role') == 'assistant' else 'user'

    analysis = {
        "last_message_from_user": last_user_turn.get('content') if last_user_turn else None,
        "last_message_from_match": last_match_turn.get('content') if last_match_turn else None,
        "Last_message_from": last_message_from,
    }

    last_match_embedding = last_match_turn.get('embedding') if last_match_turn else None
    last_match_content = last_match_turn.get('content', '') if last_match_turn else ''
    analysis['match_last_message_has_question'] = _check_semantic_similarity(last_match_content, last_match_embedding, "ASKING_A_QUESTION")
    analysis['Match_last_message_geo_context'] = _check_semantic_similarity(last_match_content, last_match_embedding, "LOCATION_REFERENCE") or \
                                                 _check_semantic_similarity(last_match_content, last_match_embedding, "TIME_REFERENCE")

    now = datetime.now(timezone.utc)
    analysis['last_user_greeted'] = False
    user_active_recently = False
    for turn in conversation_turns:
        turn_time = _parse_timestamp(turn.get('date') or turn.get('timestamp'))
        if not turn_time: continue
        if turn.get('role', 'user') == 'user':
            if turn_time > (now - timedelta(days=1)): user_active_recently = True
            if turn_time > (now - timedelta(days=2)) and _check_semantic_similarity(turn.get('content', ''), turn.get('embedding'), "GREETING"):
                analysis['last_user_greeted'] = True

    # --- Flattened Flags ---
    last_turn_time = _parse_timestamp(last_turn.get('date') or last_turn.get('timestamp'))
    if not last_turn_time: analysis['Last_message_day'] = "Unknown"
    elif len(conversation_turns) < 5: analysis['Last_message_day'] = "New Conversation"
    elif last_turn_time > (now - timedelta(days=2)): analysis['Last_message_day'] = "Active Conversation"
    elif last_turn_time > (now - timedelta(days=7)): analysis['Last_message_day'] = "Recent Conversation"
    else: analysis['Last_message_day'] = "Old Conversation"

    last_turn_embedding = last_turn.get('embedding')
    last_turn_content = last_turn.get('content', '')
    analysis['greeting_detected'] = _check_semantic_similarity(last_turn_content, last_turn_embedding, "GREETING")
    analysis['flirtation_indicator'] = _check_semantic_similarity(last_turn_content, last_turn_embedding, "FLIRTATION")
    analysis['time_reference_detected'] = _check_semantic_similarity(last_turn_content, last_turn_embedding, "TIME_REFERENCE")
    analysis['location_reference_detected'] = _check_semantic_similarity(last_turn_content, last_turn_embedding, "LOCATION_REFERENCE")

    recent_turns = conversation_turns[-5:]
    question_count = sum(1 for t in recent_turns if _check_semantic_similarity(t.get('content', ''), t.get('embedding'), "ASKING_A_QUESTION"))
    if question_count > 1: analysis['recent_engagement_score'] = "high"
    elif question_count > 0: analysis['recent_engagement_score'] = "medium"
    else: analysis['recent_engagement_score'] = "low"

    analysis['suggest_follow_up_question'] = not analysis['match_last_message_has_question']
    analysis['suggest_flirtation'] = not analysis['flirtation_indicator']
    analysis['suggest_topic_shift'] = analysis['recent_engagement_score'] == 'low' or _check_semantic_similarity(last_turn_content, last_turn_embedding, "DISENGAGEMENT")
    analysis['suggest_greeting'] = not analysis['last_user_greeted'] and not user_active_recently

    return analysis