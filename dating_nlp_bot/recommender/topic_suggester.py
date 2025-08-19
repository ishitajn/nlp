GENERAL_TOPICS = [
    "travel", "food", "career", "hobbies", "movies", "music",
    "books", "pets", "fitness", "family", "goals"
]

def suggest_topics_fast(topics: dict, dynamics: dict) -> dict:
    """
    Suggests next topics, topics to avoid, and topics to escalate (fast mode).
    """
    discussed_topics = topics.get("map", {}).keys()

    # Suggest next topic
    next_topic = "hobbies" # default
    for topic in GENERAL_TOPICS:
        if topic not in discussed_topics:
            next_topic = topic
            break

    # Suggest topic to avoid
    avoid_topic = None
    sensitive_topics = topics.get("sensitive", [])
    if sensitive_topics:
        avoid_topic = sensitive_topics[0]

    # Suggest topic to escalate
    escalate_topic = None
    flirt_level = dynamics.get("flirtation_level")
    if flirt_level == "medium":
        escalate_topic = "flirtation"
    elif flirt_level in ["high", "explicit"]:
        escalate_topic = "sexual chemistry"

    return {
        "next_topic": next_topic,
        "avoid_topic": avoid_topic,
        "escalate_topic": escalate_topic,
    }
