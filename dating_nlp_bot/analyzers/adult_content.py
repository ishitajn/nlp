import re
from dating_nlp_bot.models.adult_model import EnhancedAdultModel
from dating_nlp_bot.utils.keywords import TOPIC_KEYWORDS

FLIRTATION_KEYWORDS = TOPIC_KEYWORDS['flirt'] + TOPIC_KEYWORDS['sexual']

def analyze_adult_content_enhanced(conversation_history: list[dict]) -> dict:
    """
    Analyzes adult content in a conversation history using the enhanced model.
    """
    model = EnhancedAdultModel()
    full_text = " ".join([message.get("content", "").lower() for message in conversation_history])

    scores = model.predict(full_text)

    obscene_score = scores.get('obscene', 0.0)

    if obscene_score > 0.8:
        flirtation_level = "explicit"
    elif obscene_score > 0.5:
        flirtation_level = "high"
    elif obscene_score > 0.2:
        flirtation_level = "medium"
    else:
        flirtation_level = "low"

    sexual_response_suggestion = flirtation_level in ["high", "explicit"]

    return {
        "flirtation_level": flirtation_level,
        "sexualResponseSuggestion": sexual_response_suggestion,
    }

def analyze_adult_content_fast(conversation_history: list[dict]) -> dict:
    """
    Analyzes adult content in a conversation history using keyword matching (fast mode).
    """
    full_text = " ".join([message.get("content", "").lower() for message in conversation_history])

    # Reusing the logic from the old implementation, but simplified
    level = "low"
    if any(re.search(r'\b' + kw + r'\b', full_text) for kw in TOPIC_KEYWORDS['sexual']):
        level = "high" # Simplified mapping for fast mode
    elif any(re.search(r'\b' + kw + r'\b', full_text) for kw in TOPIC_KEYWORDS['flirt']):
        level = "medium"

    sexual_response_suggestion = level in ["high", "explicit"]

    return {
        "flirtation_level": level,
        "sexualResponseSuggestion": sexual_response_suggestion,
    }
