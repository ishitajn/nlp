TRANSITIONS = {
    "morning": ["coffee", "fitness", "romance"],
    "career": ["ambitions", "lifestyle", "travel", "romance"],
    "family": ["values", "future plans", "intimacy"],
    "romance": ["sexual", "intimacy"],
    "sexual": ["intimacy", "aftercare"]
}

def transition_score(recent_tags, candidate_tags):
    # +0.1 if any candidate is in a next-step from any recent tag
    score = 0.0
    for rt in recent_tags:
        for ct in candidate_tags:
            if ct in TRANSITIONS.get(rt, []):
                score += 0.1
    return min(score, 0.2)
