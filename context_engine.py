# In context_engine.py
import re
from typing import List, Dict, Any
from collections import Counter
from pysentimiento import create_analyzer

from utils import parse_timestamp
from preprocessor import extract_canonical_phrases

# --- Service Initialization ---
sentiment_analyzer = create_analyzer(task="sentiment", lang="en")
emotion_analyzer = create_analyzer(task="emotion", lang="en")

# --- Constants ---
QUESTION_STARTERS_REGEX = re.compile(
    r'^(who|what|where|when|why|how|is|are|do|does|did|will|can|could|should|would|have|has|had|am|was|were|don\'t|isn\'t|aren\'t)\b',
    re.IGNORECASE
)
ANALYSIS_SCHEMA_STRINGS = {
    "phases": { "Icebreaker": [r'\b(h(i|e+y+|ello)|how (are |u )?(you|u)( doin)?|your profile|we matched)\b'], "Rapport Building": [r'\b(tell me more|what about you|hobbies|passions|family|career|work|job|hiking|trip|travel)\b'], "Escalation": [r'\b(tension|desire|imagining|in person|what if|chemistry)\b'], "Explicit Banter": [r'\b(fuck|sex|nude|kink|sexting|horny|aroused)\b'], "Logistics": [r'\b(when are you free|let\'s meet|what\'s your number|schedule|date)\b'], },
    "tones": { "Playful": [r'\b(haha|lol|lmao|kidding|teasing|banter|playful|cheeky)\b', r'[😉😜😏]'], "Serious": [r'\b(to be honest|actually|my values|looking for|seriously)\b'], "Romantic": [r'\b(connection|special|beautiful|chemistry|heart|adore|lovely)\b'], "Complimentary": [r'\b(great|amazing|impressive|gorgeous|handsome|hot|sexy|cute)\b'], "Vulnerable": [r'\b(my feelings|i feel|struggle|opening up is hard|i feel safe with you)\b'], },
    "intents": { "Gathering Information": [r'\?'], "Building Comfort": [r'\b(that makes sense|i understand|thank you for sharing)\b'], "Testing Boundaries": [r'\b(what are you into|how adventurous|are you open to)\b'], "Making Plans": [r'\b(we should|let\'s|are you free|wanna grab)\b'], "Expressing Desire": [r'\b(i want you|i need you|can\'t stop thinking about you|i desire you)\b'], }
}
ANALYSIS_SCHEMA = {
    category: { tag_name: [re.compile(p) for p in patterns] for tag_name, patterns in rules.items() }
    for category, rules in ANALYSIS_SCHEMA_STRINGS.items()
}

def analyze_power_dynamics(conversation_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(conversation_turns) < 2:
        return {"summary": "Not enough data", "user_is_leading": False, "power_score": 0}
    # ... (rest of the function is unchanged)
    user_word_count, match_word_count, user_questions, match_questions = 0, 0, 0, 0
    user_response_times, match_response_times = [], []
    last_turn_time, last_turn_role = None, None
    for turn in conversation_turns:
        role, content, timestamp = turn.get('role', 'assistant').lower(), turn.get('content', ''), parse_timestamp(turn.get('date'))
        word_count = len(content.split())
        if role == 'user': user_word_count += word_count
        else: match_word_count += word_count
        if '?' in content or QUESTION_STARTERS_REGEX.match(content):
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

def extract_memory_features(conversation_turns: List[Dict[str, Any]], identified_topics_map: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts memory-related features like question history and inside jokes."""
    # 1. Extract Question History
    question_history = []
    for turn in conversation_turns:
        content = turn.get('content', '')
        if '?' in content or QUESTION_STARTERS_REGEX.match(content):
            question_history.append({"role": turn.get("role"), "question": content})

    # 2. Extract Inside Jokes (Heuristic)
    user_phrases = Counter()
    match_phrases = Counter()
    for turn in conversation_turns:
        role = turn.get("role")
        phrases = extract_canonical_phrases(turn.get("content", ""))
        if role == "user":
            user_phrases.update(phrases)
        else:
            match_phrases.update(phrases)

    potential_jokes = [phrase for phrase, count in user_phrases.items() if count > 1 and match_phrases[phrase] > 0]

    # Simple validation: check if sentiment around joke is positive
    inside_jokes = []
    for joke in potential_jokes:
        joke_turns = [t.get('content') for t in conversation_turns if joke in t.get('content', '')]
        if joke_turns:
            sentiment = sentiment_analyzer.predict(" ".join(joke_turns)).output
            if sentiment == 'POS':
                inside_jokes.append(joke.title())

    # 3. Get Avoided Topics from topic map
    avoided_topics = identified_topics_map.get("avoid", [])

    return {
        "question_history": question_history[-10:], # Limit to last 10
        "inside_jokes": inside_jokes,
        "avoided_topics": avoided_topics
    }

def extract_contextual_features(
    conversation_turns: List[Dict[str, Any]],
    identified_topics_map: Dict[str, List[str]],
    my_profile: str = "",
    their_profile: str = "",
    use_enhanced_nlp: bool = False
) -> Dict[str, Any]:
    """Extracts high-level contextual features from the conversation."""
    conversation_history_str = "\n".join([t.get('content', '') for t in conversation_turns])
    
    # --- Sentiment Analysis ---
    turns_for_sentiment = conversation_turns[-4:] if use_enhanced_nlp else conversation_turns[-10:]
    text_for_sentiment = "\n".join([t.get('content', '') for t in turns_for_sentiment])
    if not text_for_sentiment.strip():
        sentiment_analysis = {"overall": "neutral", "probas": {}}
    else:
        result = sentiment_analyzer.predict(text_for_sentiment)
        probas = result.probas
        sentiment = "neutral"
        if result.output == 'POS': sentiment = "very positive" if probas['POS'] > 0.8 else "positive"
        elif result.output == 'NEG': sentiment = "very negative" if probas['NEG'] > 0.8 else "negative"
        sentiment_analysis = { "overall": sentiment, "probas": probas }

    # --- Phase, Tone, Intent Detection ---
    full_text_lower = f"{my_profile} {their_profile} {conversation_history_str}".lower()
    detected_tags = { "detected_phases": set(), "detected_tones": set(), "detected_intents": set() }
    for category, rules in ANALYSIS_SCHEMA.items():
        for tag_name, patterns in rules.items():
            if any(re.search(pattern, full_text_lower) for pattern in patterns):
                detected_tags[f"detected_{category}"].add(tag_name)

    detected_phases = list(detected_tags["detected_phases"])
    if not detected_phases: detected_phases.append("Rapport Building")

    # --- Memory Features ---
    memory_features = extract_memory_features(conversation_turns, identified_topics_map)
    memory_features["date_arc_phase"] = detected_phases[0] # Simplistic mapping for now

    analysis_output = {
        "sentiment_analysis": sentiment_analysis,
        "detected_phases": detected_phases,
        "detected_tones": list(detected_tags["detected_tones"]),
        "detected_intents": list(detected_tags["detected_intents"]),
        "power_dynamics": analyze_power_dynamics(conversation_turns),
        "memory_features": memory_features
    }
    
    return analysis_output