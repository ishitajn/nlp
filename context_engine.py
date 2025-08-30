"""
This module is responsible for extracting high-level contextual and memory-based
features from the conversation.
"""
import re
from typing import List, Dict, Any
from collections import Counter

from utils import parse_timestamp
from preprocessor import extract_canonical_phrases

def analyze_power_dynamics(conversation_turns: List[Dict[str, Any]], services) -> Dict[str, Any]:
    """Analyzes power dynamics based on word/question counts and response times."""
    if len(conversation_turns) < 2:
        return {"summary": "Not enough data", "user_is_leading": False, "power_score": 0}

    user_word_count, match_word_count, user_questions, match_questions = 0, 0, 0, 0
    user_response_times, match_response_times = [], []
    last_turn_time, last_turn_role = None, None

    for turn in conversation_turns:
        role = turn.get('role', 'assistant').lower()
        content = turn.get('content', '')
        timestamp = parse_timestamp(turn.get('date'))

        user_word_count += len(content.split()) if role == 'user' else 0
        match_word_count += len(content.split()) if role != 'user' else 0

        if '?' in content or services.question_starters_regex.match(content):
            if role == 'user': user_questions += 1
            else: match_questions += 1

        if timestamp and last_turn_time and role != last_turn_role:
            delta_seconds = (timestamp - last_turn_time).total_seconds()
            if delta_seconds > 0:
                if role == 'user': user_response_times.append(delta_seconds)
                else: match_response_times.append(delta_seconds)

        if timestamp: last_turn_time, last_turn_role = timestamp, role

    total_words = user_word_count + match_word_count
    word_score = (user_word_count - match_word_count) / total_words if total_words > 0 else 0
    total_questions = user_questions + match_questions
    question_score = (user_questions - match_questions) / total_questions if total_questions > 0 else 0
    avg_user_response = sum(user_response_times) / len(user_response_times) if user_response_times else 0
    avg_match_response = sum(match_response_times) / len(match_response_times) if match_response_times else 0
    total_response_time = avg_user_response + avg_match_response
    response_score = (avg_match_response - avg_user_response) / total_response_time if total_response_time > 0 else 0
    initiator_role = conversation_turns[0].get('role', 'assistant').lower()
    initiation_score = 0.1 if initiator_role == 'user' else -0.1
    power_score = round(max(-1.0, min(1.0, (word_score * 0.2) + (question_score * 0.5) + (response_score * 0.3) + initiation_score)), 2)
    summary = "User is leading" if power_score > 0.25 else "Match is leading" if power_score < -0.25 else "Balanced"

    return {
        "summary": summary, "user_is_leading": power_score > 0, "power_score": power_score,
        "details": { "user_word_count": user_word_count, "match_word_count": match_word_count, "user_question_count": user_questions, "match_question_count": match_questions, "user_avg_response_s": round(avg_user_response) if avg_user_response else None, "match_avg_response_s": round(avg_match_response) if avg_match_response else None, }
    }

def _extract_memory_features(services, conversation_turns: List[Dict[str, Any]], identified_topics_map: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts memory-related features like question history and inside jokes."""
    question_history = [
        {"role": turn.get("role"), "question": content}
        for turn in conversation_turns
        if (content := turn.get('content', '')) and ('?' in content or services.question_starters_regex.match(content))
    ]

    preprocessor_args = services.get_preprocessor_args()
    user_phrases = Counter(p for t in conversation_turns if t.get("role") == "user" for p in extract_canonical_phrases(t.get("content", ""), **preprocessor_args))
    match_phrases = Counter(p for t in conversation_turns if t.get("role") != "user" for p in extract_canonical_phrases(t.get("content", ""), **preprocessor_args))

    potential_jokes = [phrase for phrase, count in user_phrases.items() if count > 1 and match_phrases[phrase] > 0]

    inside_jokes = [
        joke.title() for joke in potential_jokes
        if (joke_turns := [t.get('content') for t in conversation_turns if joke in t.get('content', '')])
        and services.sentiment_analyzer.predict(" ".join(joke_turns)).output == 'POS'
    ]

    return {
        "question_history": question_history[-10:],
        "inside_jokes": inside_jokes,
        "avoided_topics": identified_topics_map.get("avoid", [])
    }

def extract_contextual_features(
    services, conversation_turns: List[Dict[str, Any]], identified_topics_map: Dict[str, List[str]],
    my_profile: str = "", their_profile: str = "", use_enhanced_nlp: bool = False
) -> Dict[str, Any]:
    """Extracts high-level contextual features from the conversation."""
    conversation_history_str = "\n".join([t.get('content', '') for t in conversation_turns])
    
    turns_for_sentiment = conversation_turns[-4:] if use_enhanced_nlp else conversation_turns[-10:]
    text_for_sentiment = "\n".join([t.get('content', '') for t in turns_for_sentiment])
    if not text_for_sentiment.strip():
        sentiment_analysis = {"overall": "neutral", "probas": {}}
    else:
        result = services.sentiment_analyzer.predict(text_for_sentiment)
        probas = result.probas
        sentiment = "neutral"
        if result.output == 'POS': sentiment = "very positive" if probas['POS'] > 0.8 else "positive"
        elif result.output == 'NEG': sentiment = "very negative" if probas['NEG'] > 0.8 else "negative"
        sentiment_analysis = { "overall": sentiment, "probas": probas }

    full_text_lower = f"{my_profile} {their_profile} {conversation_history_str}".lower()
    detected_tags = { "detected_phases": set(), "detected_tones": set(), "detected_intents": set() }
    for category, rules in services.analysis_schema.items():
        for tag_name, patterns in rules.items():
            if any(re.search(pattern, full_text_lower) for pattern in patterns):
                detected_tags[f"detected_{category}"].add(tag_name)

    detected_phases = list(detected_tags["detected_phases"]) or ["Rapport Building"]

    memory_features = _extract_memory_features(services, conversation_turns, identified_topics_map)
    memory_features["date_arc_phase"] = detected_phases[0]

    return {
        "sentiment_analysis": sentiment_analysis,
        "detected_phases": detected_phases,
        "detected_tones": list(detected_tags["detected_tones"]),
        "detected_intents": list(detected_tags["detected_intents"]),
        "power_dynamics": analyze_power_dynamics(conversation_turns, services),
        "memory_features": memory_features
    }