import random
from dating_nlp_bot.config import SUGGESTIBLE_TOPICS

def suggest_topics_fast(topics: dict, dynamics: dict) -> dict:
    """
    Suggests next topics, topics to avoid, and topics to escalate based on context.
    """
    discussed_topics = set(topics.get("map", {}).keys())
    liked_topics = set(topics.get("liked", []))
    disliked_topics = set(topics.get("disliked", []))

    # Suggest next topic
    undiscovered_topics = [t for t in SUGGESTIBLE_TOPICS if t not in discussed_topics]
    next_topic = random.choice(undiscovered_topics) if undiscovered_topics else "common interests"

    # Suggest topic to avoid
    avoid_topic = disliked_topics.pop() if disliked_topics else None

    # Suggest topic to escalate
    escalate_topic = None
    flirt_level = dynamics.get("flirtation_level")
    if flirt_level == "medium" or "flirt" in liked_topics or "flirting" in liked_topics:
        escalate_topic = "flirtation"
    elif flirt_level in ["high", "explicit"] or "sexual" in liked_topics or "sexual topics" in liked_topics:
        escalate_topic = "sexual chemistry"

    return {
        "next_topic": next_topic,
        "avoid_topic": avoid_topic,
        "escalate_topic": escalate_topic,
    }
