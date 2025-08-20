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
    Classifies topics using keyword matching and VADER sentiment analysis (enhanced mode).
    """
    fast_results = classify_topics_fast(conversation_history)
    topic_map = fast_results["map"]
    analyzer = models.sentiment_analyzer_fast
    if not analyzer:
        return fast_results

    topic_sentiments = defaultdict(list)
    for message in conversation_history:
        text = message.get("content", "")
        vs = analyzer.polarity_scores(text)
        sentiment_score = vs['compound']

        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(re.search(r'\b' + kw + r'\b', text.lower()) for kw in keywords):
                topic_sentiments[topic].append(sentiment_score)

    liked, disliked, neutral = [], [], []
    all_topics = list(topic_map.keys())
    if 'female_centric' in topic_map and isinstance(topic_map['female_centric'], dict):
        all_topics.extend(topic_map['female_centric'].keys())

    for topic in all_topics:
        if topic in topic_sentiments:
            avg_sentiment = sum(topic_sentiments[topic]) / len(topic_sentiments[topic])
            if avg_sentiment > TOPIC_SENTIMENT_THRESHOLD_POSITIVE:
                liked.append(topic)
            elif avg_sentiment < TOPIC_SENTIMENT_THRESHOLD_NEGATIVE:
                disliked.append(topic)
            else:
                neutral.append(topic)
        else:
            neutral.append(topic)

    fast_results["liked"] = list(set(liked))
    fast_results["disliked"] = list(set(disliked))
    fast_results["neutral"] = list(set(neutral))

    return fast_results
