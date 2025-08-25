# In behavioral_engine.py
import re
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

from embedder import embedder_service
from semantic_concepts import CONCEPT_EMBEDDINGS

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
SIMILARITY_THRESHOLDS = {
    "GREETING": 0.70, "ASKING_A_QUESTION": 0.60, "FLIRTATION": 0.60,
    "TIME_REFERENCE": 0.60, "LOCATION_REFERENCE": 0.60, "DISENGAGEMENT": 0.55,
    "PLANNING_LOGISTICS": 0.65
}
QUESTION_STARTERS_REGEX = re.compile(
    r'^(who|what|where|when|why|how|is|are|do|does|did|will|can|could|should|would|have|has|had|am|was|were|don\'t|isn\'t|aren\'t)\b',
    re.IGNORECASE
)

def _parse_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """
    Robustly parses a timestamp string into a timezone-aware datetime object.

    Handles multiple common formats, including ISO 8601 with and without 'Z'.
    """
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
    except (ValueError, TypeError):
        logging.warning(f"Could not parse timestamp: {ts_str}")
    return None

def _check_semantic_similarity(text: str, text_embedding: np.ndarray, concept_name: str) -> bool:
    """
    Checks if a text's embedding is semantically similar to a pre-defined concept.
    Includes special handling for questions to improve accuracy.
    """
    if concept_name == "ASKING_A_QUESTION":
        text_stripped = text.strip()
        if '?' in text_stripped: return True
        if QUESTION_STARTERS_REGEX.match(text_stripped): return True

    concept_embedding = CONCEPT_EMBEDDINGS.get(concept_name)
    if concept_embedding is None or text_embedding is None or not hasattr(text_embedding, 'reshape'): return False

    similarity = cosine_similarity(text_embedding.reshape(1, -1), concept_embedding.reshape(1, -1))[0][0]
    threshold = SIMILARITY_THRESHOLDS.get(concept_name, 0.55 if concept_name == "ASKING_A_QUESTION" else 0.6)
    return bool(similarity > threshold)

def analyze_conversation_behavior(
    conversation_turns: List[Dict[str, Any]],
    use_enhanced_nlp: bool = False
) -> Dict[str, Any]:
    """
    Analyzes the behavioral aspects of a conversation.

    This includes:
    - Identifying the last message from each participant.
    - Detecting questions, geo/time context, greetings, and flirtation.
    - Calculating engagement score and conversation pace.
    - Generating flags for downstream suggestion logic.
    """
    if not conversation_turns: return {}

    all_contents = [turn.get('content', '') for turn in conversation_turns]
    all_embeddings = embedder_service.encode_cached(all_contents)
    for i, turn in enumerate(conversation_turns):
        turn['embedding'] = all_embeddings[i] if i < len(all_embeddings) else None

    # --- Local cache for semantic checks to avoid re-computation ---
    semantic_cache = {}
    def check_semantic_similarity_cached(text: str, embedding: np.ndarray, concept: str) -> bool:
        cache_key = (text, concept)
        if cache_key in semantic_cache:
            return semantic_cache[cache_key]
        result = _check_semantic_similarity(text, embedding, concept)
        semantic_cache[cache_key] = result
        return result

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
    analysis['match_last_message_has_question'] = check_semantic_similarity_cached(last_match_content, last_match_embedding, "ASKING_A_QUESTION")
    analysis['Match_last_message_geo_context'] = check_semantic_similarity_cached(last_match_content, last_match_embedding, "LOCATION_REFERENCE") or \
                                                 check_semantic_similarity_cached(last_match_content, last_match_embedding, "TIME_REFERENCE")

    now = datetime.now(timezone.utc)
    analysis['last_user_greeted'] = False
    user_active_recently = False
    for turn in conversation_turns:
        turn_time = _parse_timestamp(turn.get('date') or turn.get('timestamp'))
        if not turn_time: continue
        if turn.get('role', 'user') == 'user':
            if turn_time > (now - timedelta(days=1)): user_active_recently = True
            if turn_time > (now - timedelta(days=2)) and check_semantic_similarity_cached(turn.get('content', ''), turn.get('embedding'), "GREETING"):
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
    analysis['greeting_detected'] = check_semantic_similarity_cached(last_turn_content, last_turn_embedding, "GREETING")
    analysis['flirtation_indicator'] = check_semantic_similarity_cached(last_turn_content, last_turn_embedding, "FLIRTATION")
    analysis['time_reference_detected'] = check_semantic_similarity_cached(last_turn_content, last_turn_embedding, "TIME_REFERENCE")
    analysis['location_reference_detected'] = check_semantic_similarity_cached(last_turn_content, last_turn_embedding, "LOCATION_REFERENCE")

    # --- Engagement Score ---
    recent_turns = conversation_turns[-5:]
    if use_enhanced_nlp:
        # Enhanced mode uses a more nuanced engagement score
        user_questions = sum(1 for t in recent_turns if t.get('role', 'user') == 'user' and check_semantic_similarity_cached(t.get('content', ''), t.get('embedding'), "ASKING_A_QUESTION"))
        match_questions = sum(1 for t in recent_turns if t.get('role', 'user') != 'user' and check_semantic_similarity_cached(t.get('content', ''), t.get('embedding'), "ASKING_A_QUESTION"))
        user_word_count = sum(len(t.get('content', '').split()) for t in recent_turns if t.get('role', 'user') == 'user')

        if user_questions > 0 and match_questions > 0:
            analysis['recent_engagement_score'] = "high"
        elif user_questions > 0 or match_questions > 0:
            analysis['recent_engagement_score'] = "medium"
        elif user_word_count > 15:
            analysis['recent_engagement_score'] = "medium"
        else:
            analysis['recent_engagement_score'] = "low"
    else:
        # Standard mode uses a simple question count
        question_count = sum(1 for t in recent_turns if check_semantic_similarity_cached(t.get('content', ''), t.get('embedding'), "ASKING_A_QUESTION"))
        if question_count > 1:
            analysis['recent_engagement_score'] = "high"
        elif question_count > 0:
            analysis['recent_engagement_score'] = "medium"
        else:
            analysis['recent_engagement_score'] = "low"

    analysis['suggest_follow_up_question'] = not analysis['match_last_message_has_question']
    analysis['suggest_flirtation'] = not analysis['flirtation_indicator']
    analysis['suggest_topic_shift'] = analysis['recent_engagement_score'] == 'low' or check_semantic_similarity_cached(last_turn_content, last_turn_embedding, "DISENGAGEMENT")
    analysis['suggest_greeting'] = not analysis['last_user_greeted'] and not user_active_recently

    # --- Pace Calculation ---
    time_deltas = []
    if len(conversation_turns) > 1:
        for i in range(1, len(conversation_turns)):
            prev_turn_time = _parse_timestamp(conversation_turns[i-1].get('date'))
            curr_turn_time = _parse_timestamp(conversation_turns[i].get('date'))
            if prev_turn_time and curr_turn_time:
                delta = (curr_turn_time - prev_turn_time).total_seconds()
                if delta > 0:
                    time_deltas.append(delta)

    if not time_deltas:
        analysis['pace'] = "steady"
    else:
        avg_delta_minutes = (sum(time_deltas) / len(time_deltas)) / 60
        if avg_delta_minutes < 5:
            analysis['pace'] = "fast"
        elif avg_delta_minutes < 60:
            analysis['pace'] = "steady"
        else:
            analysis['pace'] = "slow"

    return analysis