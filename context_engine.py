# In context_engine.py
import re
from typing import List, Dict, Any
from collections import Counter
# NEW: Import the powerful, transformer-based sentiment analyzer
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

def analyze_power_dynamics(conversation_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    # This function remains correct and does not need changes.
    # ... (implementation from previous version)
    return {} # Placeholder for brevity

def extract_contextual_features(
    conversation_turns: List[Dict[str, Any]],
    identified_topics_map: Dict[str, List[str]],
    my_profile: str = "",
    their_profile: str = "",
    use_enhanced_nlp: bool = False
) -> Dict[str, Any]:
    conversation_history_str = "\n".join([t.get('content', '') for t in conversation_turns])
    
    # --- Sentiment Analysis ---
    # Enhanced mode focuses on recent turns for more immediate sentiment.
    if use_enhanced_nlp:
        turns_for_sentiment = conversation_turns[-4:]
        text_for_sentiment = "\n".join([t.get('content', '') for t in turns_for_sentiment])
    else:
        text_for_sentiment = conversation_history_str

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