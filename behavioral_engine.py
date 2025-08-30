"""
This module analyzes the behavioral aspects of a conversation, such as pace,
engagement, and the subtext of the last message.
"""
import re
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

from utils import parse_timestamp

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
SIMILARITY_THRESHOLDS = {
    "GREETING": 0.70, "ASKING_A_QUESTION": 0.60, "FLIRTATION": 0.60,
    "TIME_REFERENCE": 0.60, "LOCATION_REFERENCE": 0.60, "DISENGAGEMENT": 0.55,
    "PLANNING_LOGISTICS": 0.65
}
LOW_EFFORT_PHRASES = {"ok", "lol", "haha", "k", "cool", "nice", "sure", "yep", "yeah", "yup"}
HIGH_AROUSAL_EMOTIONS = {"joy": 0.8, "anger": 0.9, "surprise": 0.7}
LOW_AROUSAL_EMOTIONS = {"sadness": -0.5, "fear": -0.6, "disgust": -0.4}

def _check_semantic_similarity(text: str, text_embedding: np.ndarray, concept_name: str, services) -> bool:
    """Checks if a text's embedding is semantically similar to a pre-defined concept."""
    if concept_name == "ASKING_A_QUESTION" and ('?' in text.strip() or services.question_starters_regex.match(text.strip())):
        return True

    concept_embedding = services.concept_embeddings.get(concept_name)
    if concept_embedding is None or text_embedding is None or not hasattr(text_embedding, 'reshape'):
        return False

    similarity = cosine_similarity(text_embedding.reshape(1, -1), concept_embedding.reshape(1, -1))[0][0]
    threshold = SIMILARITY_THRESHOLDS.get(concept_name, 0.6)
    return bool(similarity > threshold)

def analyze_conversation_behavior(services, conversation_turns: List[Dict[str, Any]], use_enhanced_nlp: bool = False) -> Dict[str, Any]:
    """Analyzes overall conversation behavior like pace and engagement."""
    if not conversation_turns: return {}

    all_contents = [turn.get('content', '') for turn in conversation_turns]
    all_embeddings = services.embedder.encode_cached(all_contents)
    for i, turn in enumerate(conversation_turns):
        turn['embedding'] = all_embeddings[i] if i < len(all_embeddings) else None

    semantic_cache = {}
    def check_semantic_similarity_cached(text: str, embedding: np.ndarray, concept: str) -> bool:
        cache_key = (text, concept)
        if cache_key not in semantic_cache:
            semantic_cache[cache_key] = _check_semantic_similarity(text, embedding, concept, services)
        return semantic_cache[cache_key]

    last_user_turn = next((turn for turn in reversed(conversation_turns) if turn.get('role') == 'user'), None)
    last_match_turn = next((turn for turn in reversed(conversation_turns) if turn.get('role') == 'assistant'), None)
    last_turn = conversation_turns[-1]

    analysis = {
        "last_message_from_user": last_user_turn.get('content') if last_user_turn else None,
        "last_message_from_match": last_match_turn.get('content') if last_match_turn else None,
        "Last_message_from": 'match' if last_turn.get('role') == 'assistant' else 'user',
    }

    if last_match_turn:
        analysis['match_last_message_has_question'] = check_semantic_similarity_cached(last_match_turn.get('content', ''), last_match_turn.get('embedding'), "ASKING_A_QUESTION")
    else:
        analysis['match_last_message_has_question'] = False

    now = datetime.now(timezone.utc)
    user_turns = [t for t in conversation_turns if t.get('role') == 'user']
    user_active_recently = any(parse_timestamp(t.get('date')) > (now - timedelta(days=1)) for t in user_turns if t.get('date'))
    analysis['last_user_greeted'] = any(
        parse_timestamp(t.get('date')) > (now - timedelta(days=2)) and check_semantic_similarity_cached(t.get('content', ''), t.get('embedding'), "GREETING")
        for t in user_turns if t.get('date')
    )

    last_turn_time = parse_timestamp(last_turn.get('date'))
    if not last_turn_time: state = "Unknown"
    elif len(conversation_turns) <= 5: state = "EARLY_CONVO"
    else:
        days_since_last_message = (now - last_turn_time).days
        if days_since_last_message < 2: state = "ACTIVE_CONVO"
        elif days_since_last_message < 7: state = "REENGAGING_DAY"
        elif days_since_last_message < 30: state = "REENGAGING_WEEK"
        else: state = "REENGAGING_MONTH"
    analysis['conversation_state'] = state

    analysis['flirtation_indicator'] = check_semantic_similarity_cached(last_turn.get('content', ''), last_turn.get('embedding'), "FLIRTATION")

    recent_turns = conversation_turns[-5:]
    question_count = sum(1 for t in recent_turns if check_semantic_similarity_cached(t.get('content',''), t.get('embedding'), "ASKING_A_QUESTION"))
    if question_count > 1: analysis['recent_engagement_score'] = "high"
    elif question_count > 0: analysis['recent_engagement_score'] = "medium"
    else: analysis['recent_engagement_score'] = "low"

    analysis['suggest_topic_shift'] = analysis['recent_engagement_score'] == 'low' or check_semantic_similarity_cached(last_turn.get('content', ''), last_turn.get('embedding'), "DISENGAGEMENT")
    analysis['suggest_greeting'] = not user_active_recently and not analysis['last_user_greeted']

    time_deltas = [
        (parse_timestamp(curr.get('date')) - parse_timestamp(prev.get('date'))).total_seconds()
        for prev, curr in zip(conversation_turns, conversation_turns[1:])
        if prev.get('date') and curr.get('date') and parse_timestamp(curr.get('date')) > parse_timestamp(prev.get('date'))
    ]
    if not time_deltas: analysis['pace'] = "steady"
    else:
        avg_delta_minutes = (sum(time_deltas) / len(time_deltas)) / 60
        if avg_delta_minutes < 5: analysis['pace'] = "fast"
        elif avg_delta_minutes < 60: analysis['pace'] = "steady"
        else: analysis['pace'] = "slow"

    return analysis

def analyze_last_message_details(services, last_turn: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Performs a detailed analysis of the last message in the conversation."""
    default_response = {
        "isDirectQuestion": False, "isLowEffort": True, "isSarcastic": False,
        "isAmbiguous": False, "isVulnerable": False, "valence": 0.0, "arousal": 0.0, "intents": []
    }
    if not last_turn or not last_turn.get("content"): return default_response

    content = last_turn.get("content", "")
    content_lower = content.lower()
    word_count = len(content.split())

    sentiment_result = services.sentiment_analyzer.predict(content)
    emotion_result = services.emotion_analyzer.predict(content)

    probas = sentiment_result.probas
    valence = probas.get('POS', 0.0) - probas.get('NEG', 0.0)
    detected_emotion = emotion_result.output
    arousal = HIGH_AROUSAL_EMOTIONS.get(detected_emotion, 0.0) or LOW_AROUSAL_EMOTIONS.get(detected_emotion, 0.0)

    is_vulnerable = any(re.search(p, content_lower) for p in services.analysis_schema['tones']['Vulnerable'])
    detected_intents = [
        name for name, patterns in services.analysis_schema['intents'].items()
        if any(re.search(p, content_lower) for p in patterns)
    ]
    is_direct_question = "Gathering Information" in detected_intents

    is_low_effort = (word_count <= 3 and content_lower in LOW_EFFORT_PHRASES) or word_count <= 2
    positive_words = ["love", "great", "amazing", "so fun", "fantastic"]
    is_sarcastic = any(word in content_lower for word in positive_words) and sentiment_result.output == 'NEG'
    ambiguous_phrases = ["i guess", "maybe", "i don't know", "perhaps"]
    is_ambiguous = any(phrase in content_lower for phrase in ambiguous_phrases)

    return {
        "isDirectQuestion": is_direct_question, "isLowEffort": is_low_effort,
        "isSarcastic": is_sarcastic, "isAmbiguous": is_ambiguous,
        "isVulnerable": is_vulnerable, "valence": round(valence, 2),
        "arousal": round(arousal, 2), "intents": detected_intents
    }