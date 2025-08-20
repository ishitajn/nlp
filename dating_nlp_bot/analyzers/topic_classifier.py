import re
from collections import defaultdict
from dating_nlp_bot.model_loader import get_models
from dating_nlp_bot.utils.keywords import TOPIC_KEYWORDS
from dating_nlp_bot.config import (
    TOPIC_SENTIMENT_THRESHOLD_POSITIVE,
    TOPIC_SENTIMENT_THRESHOLD_NEGATIVE,
    GENERAL_TOPICS,
    FEMALE_CENTRIC_TOPICS,
)

models = get_models()

def classify_topics_fast(conversation_history: list[dict]) -> dict:
    """
    Classifies topics in a conversation history using keyword matching (fast mode).
    """
    topic_map = defaultdict(list)
    sensitive = []
    kinks_and_fetishes = []
    porn_references = []

    topic_map['female_centric'] = defaultdict(list)

    full_text = " ".join([message.get("content", "").lower() for message in conversation_history])

    for topic, keywords in TOPIC_KEYWORDS.items():
        matched_keywords = [kw for kw in keywords if re.search(r'\b' + kw + r'\b', full_text)]
        if not matched_keywords:
            continue

        if topic in GENERAL_TOPICS:
            topic_map[topic].extend(matched_keywords)
        elif topic in FEMALE_CENTRIC_TOPICS:
            topic_map['female_centric'][topic].extend(matched_keywords)

        if topic in ["flirt", "sexual"]:
            if topic not in sensitive:
                sensitive.append(topic)
        if topic == "kinksAndFetishes":
            kinks_and_fetishes.extend(matched_keywords)
        if topic == "pornReferences":
            porn_references.extend(matched_keywords)

    for key, value in topic_map.items():
        if isinstance(value, list):
            topic_map[key] = list(set(value))
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                topic_map[key][sub_key] = list(set(sub_value))

    return {
        "liked": [], "disliked": [], "neutral": [],
        "sensitive": list(set(sensitive)),
        "kinksAndFetishes": list(set(kinks_and_fetishes)),
        "pornReferences": list(set(porn_references)),
        "map": dict(topic_map),
    }

def classify_topics_enhanced(conversation_history: list[dict]) -> dict:
    """
    Classifies topics using a zero-shot classification model and VADER sentiment analysis.
    This approach is more accurate and doesn't rely on hardcoded keywords.
    The user's request mentioned sentence-transformers, and this zero-shot model is a powerful
    application of transformer technology for classification without needing a pre-trained model
    on specific topics.
    """
    classifier = models.topic_classifier_enhanced
    sentiment_analyzer = models.sentiment_analyzer_fast
    if not classifier or not sentiment_analyzer:
        return classify_topics_fast(conversation_history)

    full_text = " ".join([message.get("content", "") for message in conversation_history])
    if not full_text:
        return {"liked": [], "disliked": [], "neutral": [], "sensitive": [], "map": {}}

    candidate_labels = [
        "travel", "food", "sports", "career", "fashion", "wellness",
        "hobbies", "social", "relationships", "emotions", "flirting",
        "sexual topics", "kinks and fetishes", "pornography"
    ]

    results = classifier(full_text, candidate_labels, multi_label=True)

    topic_map = defaultdict(list)
    sensitive_topics = ["flirting", "sexual topics", "kinks and fetishes", "pornography"]
    sensitive = []
    topic_sentiments = defaultdict(list)

    identified_topics = []
    if results:
        for topic, score in zip(results['labels'], results['scores']):
            if score > 0.5:  # Confidence threshold
                identified_topics.append(topic)
                topic_map[topic].append(f"score: {score:.2f}")
                if topic in sensitive_topics:
                    sensitive.append(topic)

    # Correlate topics with sentiment from messages
    for topic in identified_topics:
        for message in conversation_history:
            text = message.get("content", "")
            if re.search(r'\b' + re.escape(topic.split(" ")[0]) + r'\b', text, re.IGNORECASE):
                vs = sentiment_analyzer.polarity_scores(text)
                sentiment_score = vs['compound']
                topic_sentiments[topic].append(sentiment_score)

    liked, disliked, neutral = [], [], []
    for topic in identified_topics:
        if topic in topic_sentiments and topic_sentiments[topic]:
            avg_sentiment = sum(topic_sentiments[topic]) / len(topic_sentiments[topic])
            if avg_sentiment > TOPIC_SENTIMENT_THRESHOLD_POSITIVE:
                liked.append(topic)
            elif avg_sentiment < TOPIC_SENTIMENT_THRESHOLD_NEGATIVE:
                disliked.append(topic)
            else:
                neutral.append(topic)
        else:
            neutral.append(topic)

    return {
        "liked": list(set(liked)),
        "disliked": list(set(disliked)),
        "neutral": list(set(neutral)),
        "sensitive": list(set(sensitive)),
        "kinksAndFetishes": topic_map.get("kinks and fetishes", []),
        "pornReferences": topic_map.get("pornography", []),
        "map": dict(topic_map),
    }
