# In context_engine.py
import re
from typing import List, Dict, Any
from pysentimiento import create_analyzer

# --- Service Initialization ---
# Initialize the sentiment analyzer once at the module level for efficiency
# This model is fine-tuned for social media and understands emotion, hate speech, etc.
sentiment_analyzer = create_analyzer(task="sentiment", lang="en")

# ... (analyze_power_dynamics function remains the same) ...
ANALYSIS_SCHEMA_STRINGS = {
    "phases": { "Icebreaker": [r'\b(h(i|e+y+|ello)|how (are |u )?(you|u)( doin)?|your profile|we matched)\b'], "Rapport Building": [r'\b(tell me more|what about you|hobbies|passions|family|career|work|job|hiking|trip|travel)\b'], "Escalation": [r'\b(tension|desire|imagining|in person|what if|chemistry)\b'], "Explicit Banter": [r'\b(fuck|sex|nude|kink|sexting|horny|aroused)\b'], "Logistics": [r'\b(when are you free|let\'s meet|what\'s your number|schedule|date)\b'], },
    "tones": { "Playful": [r'\b(haha|lol|lmao|kidding|teasing|banter|playful|cheeky)\b', r'[😉😜😏]'], "Serious": [r'\b(to be honest|actually|my values|looking for|seriously)\b'], "Romantic": [r'\b(connection|special|beautiful|chemistry|heart|adore|lovely)\b'], "Complimentary": [r'\b(great|amazing|impressive|gorgeous|handsome|hot|sexy|cute)\b'], "Vulnerable": [r'\b(my feelings|i feel|struggle|opening up is hard|i feel safe with you)\b'], },
    "intents": { "Gathering Information": [r'\?'], "Building Comfort": [r'\b(that makes sense|i understand|thank you for sharing)\b'], "Testing Boundaries": [r'\b(what are you into|how adventurous|are you open to)\b'], "Making Plans": [r'\b(we should|let\'s|are you free|wanna grab)\b'], "Expressing Desire": [r'\b(i want you|i need you|can\'t stop thinking about you|i desire you)\b'], }
}

ANALYSIS_SCHEMA = {
    category: {
        tag_name: [re.compile(p) for p in patterns]
        for tag_name, patterns in rules.items()
    }
    for category, rules in ANALYSIS_SCHEMA_STRINGS.items()
}

from behavioral_engine import _parse_timestamp

def analyze_power_dynamics(conversation_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyzes the power dynamics of a conversation based on several metrics.

    Metrics calculated:
    - Word count ratio (user vs. match)
    - Question count ratio (user vs. match)
    - Average response time (user vs. match)
    - Initiation (who sent the first message)

    Returns:
        A dictionary containing the analysis, including a final power score
        where > 0 means the user is leading, < 0 means the match is leading.
    """
    if len(conversation_turns) < 2:
        return {"summary": "Not enough data", "user_is_leading": False, "power_score": 0}

    user_word_count = 0
    match_word_count = 0
    user_questions = 0
    match_questions = 0
    user_response_times = []
    match_response_times = []

    last_turn_time = None
    last_turn_role = None

    for turn in conversation_turns:
        role = turn.get('role', 'assistant').lower()
        content = turn.get('content', '')
        timestamp = _parse_timestamp(turn.get('date'))

        # Word count
        word_count = len(content.split())
        if role == 'user':
            user_word_count += word_count
        else:
            match_word_count += word_count

        # Question count
        if '?' in content:
            if role == 'user':
                user_questions += 1
            else:
                match_questions += 1

        # Response time
        if timestamp and last_turn_time and role != last_turn_role:
            delta_seconds = (timestamp - last_turn_time).total_seconds()
            if delta_seconds > 0: # Avoid negative times from out-of-order messages
                if role == 'user':
                    user_response_times.append(delta_seconds)
                else:
                    match_response_times.append(delta_seconds)

        if timestamp:
            last_turn_time = timestamp
            last_turn_role = role

    # --- Calculate Ratios and Scores ---
    total_words = user_word_count + match_word_count
    word_score = (user_word_count - match_word_count) / total_words if total_words > 0 else 0

    total_questions = user_questions + match_questions
    question_score = (user_questions - match_questions) / total_questions if total_questions > 0 else 0

    avg_user_response = sum(user_response_times) / len(user_response_times) if user_response_times else 0
    avg_match_response = sum(match_response_times) / len(match_response_times) if match_response_times else 0

    # Lower response time is better (more engaged), so we subtract it.
    # We normalize by the sum to keep the score bounded.
    total_response_time = avg_user_response + avg_match_response
    response_score = (avg_match_response - avg_user_response) / total_response_time if total_response_time > 0 else 0

    # Initiation score
    initiator_role = conversation_turns[0].get('role', 'assistant').lower()
    initiation_score = 0.1 if initiator_role == 'user' else -0.1

    # --- Final Power Score ---
    # Weighted average of the individual scores. Question asking is weighted highest.
    power_score = (word_score * 0.2) + (question_score * 0.5) + (response_score * 0.3) + initiation_score
    power_score = round(max(-1.0, min(1.0, power_score)), 2) # Clamp between -1 and 1

    summary = "Balanced"
    if power_score > 0.25: summary = "User is leading"
    elif power_score < -0.25: summary = "Match is leading"

    return {
        "summary": summary,
        "user_is_leading": power_score > 0,
        "power_score": power_score,
        "details": {
            "user_word_count": user_word_count,
            "match_word_count": match_word_count,
            "user_question_count": user_questions,
            "match_question_count": match_questions,
            "user_avg_response_s": round(avg_user_response) if avg_user_response else None,
            "match_avg_response_s": round(avg_match_response) if avg_match_response else None,
        }
    }

def extract_contextual_features(
    conversation_turns: List[Dict[str, Any]],
    identified_topics_map: Dict[str, List[str]],
    my_profile: str = "",
    their_profile: str = "",
    use_enhanced_nlp: bool = False
) -> Dict[str, Any]:
    """
    Extracts high-level contextual features from the conversation.

    This includes:
    - Sentiment analysis of recent turns.
    - Detection of conversational phases, tones, and intents using regex.
    - Analysis of power dynamics.

    Args:
        conversation_turns: The list of conversation turns.
        identified_topics_map: A map of identified topics.
        my_profile: The user's profile text.
        their_profile: The match's profile text.
        use_enhanced_nlp: Flag to determine which logic to use.

    Returns:
        A dictionary containing the extracted contextual features.
    """
    conversation_history_str = "\n".join([t.get('content', '') for t in conversation_turns])
    
    # --- Sentiment Analysis ---
    # Enhanced mode uses a smaller window for more immediate sentiment.
    # Standard mode uses a slightly larger window.
    if use_enhanced_nlp:
        turns_for_sentiment = conversation_turns[-4:]
    else:
        turns_for_sentiment = conversation_turns[-10:]

    text_for_sentiment = "\n".join([t.get('content', '') for t in turns_for_sentiment])

    if not text_for_sentiment.strip():
        analysis = {"sentiment_analysis": {"overall": "neutral", "probas": {}}}
    else:
        result = sentiment_analyzer.predict(text_for_sentiment)
        probas = result.probas
        sentiment = "neutral"
        if result.output == 'POS':
            sentiment = "very positive" if probas['POS'] > 0.8 else "positive"
        elif result.output == 'NEG':
            sentiment = "very negative" if probas['NEG'] > 0.8 else "negative"
        analysis = {"sentiment_analysis": { "overall": sentiment, "probas": probas }}

    # --- Other contextual features (remain the same) ---
    full_text_lower = f"{my_profile} {their_profile} {conversation_history_str}".lower()
    detected_tags = { "detected_phases": set(), "detected_tones": set(), "detected_intents": set() }
    for category, rules in ANALYSIS_SCHEMA.items():
        for tag_name, patterns in rules.items():
            if any(re.search(pattern, full_text_lower) for pattern in patterns):
                detected_tags[f"detected_{category}"].add(tag_name)
    if not detected_tags["detected_phases"]: detected_tags["detected_phases"].add("Rapport Building")
    analysis.update({k: list(v) for k, v in detected_tags.items()})

    analysis["power_dynamics"] = analyze_power_dynamics(conversation_turns)
    
    return analysis