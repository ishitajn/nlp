import re
TOPIC_MAP = {
    "morning": r"\b(morning|breakfast|coffee)\b",
    "fitness": r"\b(workout|gym|run|jog|exercise)\b",
    "career": r"\b(job|career|work|company|degree|management)\b",
    "romance": r"\b(kiss|romance|date|chemistry|attraction)\b",
    "sexual": r"\b(erotic|sex|turn on|fantasy|foreplay)\b",
    "family": r"\b(niece|family|parents|kids|sibling)\b"
}

def tag_topics(t):
    tags = []
    for k, pat in TOPIC_MAP.items():
        if re.search(pat, t, re.I):
            tags.append(k)
    return tags
