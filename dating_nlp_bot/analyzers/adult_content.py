import re
from ..models.adult_model import EnhancedAdultModel

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

FLIRTATION_KEYWORDS = {
    "low": ["cute", "sweet", "nice", "kind"],
    "medium": ["hot", "sexy", "beautiful", "gorgeous", "attractive"],
    "high": ["desire", "passion", "intimate", "erotic", "seductive"],
    "explicit": ["sex", "fuck", "dick", "pussy", "cum", "oral", "anal", "kinky"],
}

def analyze_adult_content_fast(conversation_history: list[dict]) -> dict:
    """
    Analyzes adult content in a conversation history using keyword matching (fast mode).
    """
    full_text = " ".join([message.get("content", "").lower() for message in conversation_history])

    level_counts = {level: 0 for level in FLIRTATION_KEYWORDS.keys()}

    for level, keywords in FLIRTATION_KEYWORDS.items():
        for keyword in keywords:
            if re.search(r'\b' + keyword + r'\b', full_text):
                level_counts[level] += 1

    flirtation_level = "low"
    if level_counts["explicit"] > 0:
        flirtation_level = "explicit"
    elif level_counts["high"] > 0:
        flirtation_level = "high"
    elif level_counts["medium"] > 0:
        flirtation_level = "medium"

    sexual_response_suggestion = flirtation_level in ["high", "explicit"]

    return {
        "flirtation_level": flirtation_level,
        "sexualResponseSuggestion": sexual_response_suggestion,
    }
