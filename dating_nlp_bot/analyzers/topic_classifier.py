import re
from collections import defaultdict
from ..models.topic_model import EnhancedTopicModel
from ..models.sentiment_model import EnhancedSentimentModel
from ..utils.keywords import TOPIC_KEYWORDS

def classify_topics_enhanced(conversation_history: list[dict]) -> dict:
    """
    Classifies topics in a conversation history using the enhanced model.
    """
    messages = [msg.get("content", "") for msg in conversation_history]

    # Get topics and message cluster labels
    topic_model = EnhancedTopicModel(num_clusters=min(5, len(messages)))
    topic_map, labels = topic_model.get_topics(messages)

    # Get sentiment for each message
    sentiment_model = EnhancedSentimentModel()
    sentiments = [sentiment_model.predict(msg) for msg in messages]

    # Group sentiments by cluster
    cluster_sentiments = defaultdict(list)
    for i, label in enumerate(labels):
        sentiment_score = 1 if sentiments[i][0] == 'positive' else -1 if sentiments[i][0] == 'negative' else 0
        cluster_sentiments[label].append(sentiment_score)

    # Determine liked/disliked topics
    liked, disliked, neutral = [], [], []

    # This is a simplification. A better approach would be to map messages to topics.
    # For now, we assume all messages contribute to all identified topics.
    avg_sentiment = sum(score for scores in cluster_sentiments.values() for score in scores) / len(messages) if messages else 0

    all_topics = list(topic_map.keys())
    if 'female_centric' in topic_map:
        all_topics.extend(topic_map['female_centric'].keys())

    for topic in all_topics:
        # A more advanced implementation would calculate sentiment per topic.
        # For now, we use the overall sentiment to classify all topics.
        if avg_sentiment > 0.2:
            liked.append(topic)
        elif avg_sentiment < -0.2:
            disliked.append(topic)
        else:
            neutral.append(topic)

    # For other categories, we can still use keyword matching for simplicity
    fast_results = classify_topics_fast(conversation_history)

    return {
        "liked": list(set(liked)),
        "disliked": list(set(disliked)),
        "neutral": list(set(neutral)),
        "sensitive": fast_results["sensitive"],
        "kinksAndFetishes": fast_results["kinksAndFetishes"],
        "pornReferences": fast_results["pornReferences"],
        "map": topic_map,
    }


def classify_topics_fast(conversation_history: list[dict]) -> dict:
    """
    Classifies topics in a conversation history using keyword matching (fast mode).
    """
    topic_map = defaultdict(list)
    sensitive = []
    kinks_and_fetishes = []
    porn_references = []

    # Initialize female_centric map
    topic_map['female_centric'] = {
        "fashion": [], "wellness": [], "hobbies": [], "social": [], "relationships": []
    }

    full_text = " ".join([message.get("content", "").lower() for message in conversation_history])

    # A bit of refactoring to handle nested female_centric topics
    general_topics = ["travel", "food", "sports", "career", "flirt", "sexual", "emotions"]
    female_centric_topics = ["fashion", "wellness", "hobbies", "social", "relationships"]

    for topic, keywords in TOPIC_KEYWORDS.items():
        matched_keywords = [kw for kw in keywords if re.search(r'\b' + kw + r'\b', full_text)]
        if not matched_keywords:
            continue

        if topic in general_topics:
            topic_map[topic].extend(matched_keywords)
        elif topic in female_centric_topics:
            topic_map['female_centric'][topic].extend(matched_keywords)

        if topic in ["flirt", "sexual"]:
            if topic not in sensitive:
                sensitive.append(topic)
        if topic == "kinksAndFetishes":
            kinks_and_fetishes.extend(matched_keywords)
        if topic == "pornReferences":
            porn_references.extend(matched_keywords)

    # Deduplicate keywords
    for key, value in topic_map.items():
        if isinstance(value, list):
            topic_map[key] = list(set(value))
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                topic_map[key][sub_key] = list(set(sub_value))


    # Liked/disliked/neutral are not determined in fast mode.
    return {
        "liked": [],
        "disliked": [],
        "neutral": [],
        "sensitive": list(set(sensitive)),
        "kinksAndFetishes": list(set(kinks_and_fetishes)),
        "pornReferences": list(set(porn_references)),
        "map": dict(topic_map),
    }
