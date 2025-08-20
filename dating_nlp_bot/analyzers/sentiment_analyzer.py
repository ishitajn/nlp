from dating_nlp_bot.model_loader import get_models
from dating_nlp_bot.config import (
    SENTIMENT_THRESHOLD_FAST_POSITIVE,
    SENTIMENT_THRESHOLD_FAST_NEGATIVE,
    SENTIMENT_THRESHOLD_ENHANCED_POSITIVE,
    SENTIMENT_THRESHOLD_ENHANCED_NEGATIVE,
)

models = get_models()

def analyze_sentiment_enhanced(conversation_history: list[dict]) -> dict:
    """
    Analyzes sentiment for a conversation history using VADER (enhanced mode).
    This is now aligned with the fast analysis but can use different thresholds.
    """
    analyzer = models.sentiment_analyzer_fast
    if not analyzer:
        # Fallback or error handling
        return {"overall": "neutral", "error": "VADER sentiment analyzer not available"}

    overall_scores = []

    for message in conversation_history:
        text = message.get("content", "")
        vs = analyzer.polarity_scores(text)
        overall_scores.append(vs['compound'])

    overall_sentiment_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0

    if overall_sentiment_score >= SENTIMENT_THRESHOLD_ENHANCED_POSITIVE:
        overall_sentiment = "positive"
    elif overall_sentiment_score <= SENTIMENT_THRESHOLD_ENHANCED_NEGATIVE:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"

    return {"overall": overall_sentiment}

def analyze_sentiment_fast(conversation_history: list[dict]) -> dict:
    """
    Analyzes sentiment for a conversation history using VADER (fast mode).
    """
    analyzer = models.sentiment_analyzer_fast
    if not analyzer:
        # Fallback or error handling
        return {"overall": "neutral", "error": "VADER sentiment analyzer not available"}

    overall_scores = []

    for message in conversation_history:
        text = message.get("content", "")
        vs = analyzer.polarity_scores(text)
        overall_scores.append(vs['compound'])

    overall_sentiment_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0

    if overall_sentiment_score >= SENTIMENT_THRESHOLD_FAST_POSITIVE:
        overall_sentiment = "positive"
    elif overall_sentiment_score <= SENTIMENT_THRESHOLD_FAST_NEGATIVE:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"

    return {"overall": overall_sentiment}
