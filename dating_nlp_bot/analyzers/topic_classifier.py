import re
import torch
from collections import defaultdict
from sentence_transformers import util
from dating_nlp_bot.model_loader import get_models
from dating_nlp_bot.utils.keywords import TOPIC_KEYWORDS
from dating_nlp_bot.config import (
    TOPIC_SENTIMENT_THRESHOLD_POSITIVE,
    TOPIC_SENTIMENT_THRESHOLD_NEGATIVE,
    GENERAL_TOPICS,
    FEMALE_CENTRIC_TOPICS,
    ENHANCED_TOPIC_CANDIDATE_LABELS,
    TOPIC_CONFIDENCE_THRESHOLD,
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

    full_text = " ".join([message.get("content", "").lower() for message in conversation_history])

    for topic, keywords in TOPIC_KEYWORDS.items():
        matched_keywords = [kw for kw in keywords if re.search(r'\b' + kw + r'\b', full_text)]
        if not matched_keywords:
            continue

        if topic in GENERAL_TOPICS or topic in FEMALE_CENTRIC_TOPICS:
            topic_map[topic].extend(matched_keywords)

        if topic in ["flirt", "sexual"]:
            if topic not in sensitive:
                sensitive.append(topic)
        if topic == "kinksAndFetishes":
            kinks_and_fetishes.extend(matched_keywords)
        if topic == "pornReferences":
            porn_references.extend(matched_keywords)

    # Convert lists to sets to remove duplicates, then back to lists
    for key, value in topic_map.items():
        topic_map[key] = list(set(value))

    return {
        "liked": [], "disliked": [], "neutral": [],
        "sensitive": list(set(sensitive)),
        "kinksAndFetishes": list(set(kinks_and_fetishes)),
        "pornReferences": list(set(porn_references)),
        "map": dict(topic_map),
    }

def classify_topics_enhanced(conversation_history: list[dict]) -> dict:
    """
    Classifies topics using sentence embeddings and cosine similarity.
    This approach provides more accurate semantic topic matching.
    """
    embedding_model = models.embedding_model
    sentiment_analyzer = models.sentiment_analyzer_fast
    if not embedding_model or not sentiment_analyzer:
        return classify_topics_fast(conversation_history)

    full_text = " ".join([message.get("content", "") for message in conversation_history])
    if not full_text:
        return {"liked": [], "disliked": [], "neutral": [], "sensitive": [], "map": {}}

    # Generate embeddings for the conversation and candidate topics
    conversation_embedding_list = embedding_model.get_embeddings([full_text])
    topic_embeddings_list = embedding_model.get_embeddings(ENHANCED_TOPIC_CANDIDATE_LABELS)

    # Convert lists to tensors for cosine similarity calculation
    conversation_embedding = torch.tensor(conversation_embedding_list)
    topic_embeddings = torch.tensor(topic_embeddings_list)

    # Compute cosine similarities
    cosine_scores = util.cos_sim(conversation_embedding, topic_embeddings)

    topic_map = defaultdict(list)
    sensitive_topics = ["flirting", "sexual topics", "kinks and fetishes", "pornography"]
    sensitive = []
    kinks_and_fetishes = []
    porn_references = []
    topic_sentiments = defaultdict(list)
    identified_topics = []

    for i, topic in enumerate(ENHANCED_TOPIC_CANDIDATE_LABELS):
        if cosine_scores[0][i] > TOPIC_CONFIDENCE_THRESHOLD:
            identified_topics.append(topic)
            topic_map[topic] = []  # No keywords to add for now
            if topic in sensitive_topics:
                sensitive.append(topic)
            if topic == "kinks and fetishes":
                kinks_and_fetishes.append(topic)
            if topic == "pornography":
                porn_references.append(topic)

    # Correlate topics with sentiment from messages
    for message in conversation_history:
        text = message.get("content", "").lower()
        vs = sentiment_analyzer.polarity_scores(text)
        sentiment_score = vs['compound']
        for topic in identified_topics:
            if re.search(r'\b' + re.escape(topic) + r'\b', text, re.IGNORECASE):
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
        "kinksAndFetishes": list(set(kinks_and_fetishes)),
        "pornReferences": list(set(porn_references)),
        "map": dict(topic_map),
    }
