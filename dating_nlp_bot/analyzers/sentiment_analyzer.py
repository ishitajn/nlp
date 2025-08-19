from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dating_nlp_bot.models.sentiment_model import EnhancedSentimentModel

def analyze_sentiment_enhanced(conversation_history: list[dict]) -> dict:
    """
    Analyzes sentiment for a conversation history using the enhanced model.
    """
    model = EnhancedSentimentModel()
    overall_scores = []

    for message in conversation_history:
        text = message.get("content", "")
        label, _ = model.predict(text)
        if label == "positive":
            overall_scores.append(1)
        elif label == "negative":
            overall_scores.append(-1)
        else:
            overall_scores.append(0)

    overall_sentiment_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0

    if overall_sentiment_score > 0.1:
        overall_sentiment = "positive"
    elif overall_sentiment_score < -0.1:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"

    return {"overall": overall_sentiment}

def analyze_sentiment_fast(conversation_history: list[dict]) -> dict:
    """
    Analyzes sentiment for a conversation history using VADER (fast mode).
    """
    analyzer = SentimentIntensityAnalyzer()
    overall_scores = []

    for message in conversation_history:
        text = message.get("content", "")
        vs = analyzer.polarity_scores(text)
        overall_scores.append(vs['compound'])

    overall_sentiment_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0

    if overall_sentiment_score >= 0.05:
        overall_sentiment = "positive"
    elif overall_sentiment_score <= -0.05:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"

    return {"overall": overall_sentiment}
