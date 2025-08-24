# In suggestion_engine.py
import random
from typing import Dict, Any, List, Optional, Set

from model import Feedback

# ==============================================================================
# == REFINED & CONTEXT-AWARE SUGGESTION TEMPLATES                             ==
# ==============================================================================
# These templates are designed to be more open-ended and adaptable.
# They are now structured by the INTENT of the suggestion.
SUGGESTION_TEMPLATES = {
    # Suggestions for new topics to introduce
    "topics": {
        "curiosity": [
            "I'm curious, what's your take on {topic}?",
            "Changing gears a bit, but I've been thinking about {topic} lately.",
            "Tell me about your experience with {topic}.",
        ],
        "playful": [
            "Random thought: are you a fan of {topic}?",
            "Let's talk about something fun, like {topic}.",
            "I have a feeling you might have some interesting stories about {topic}.",
        ],
        "deep": [
            "I'd love to hear your thoughts on {topic}.",
            "On a deeper note, how do you feel about {topic}?",
            "I feel like we could have a good conversation about {topic}.",
        ],
    },
    # Questions to ask about an existing topic
    "questions": {
        "open_ended": [
            "What's your favorite memory related to {topic}?",
            "How does {topic} make you feel?",
            "What's something about {topic} that most people don't know?",
        ],
        "personal": [
            "What's your personal connection to {topic}?",
            "Has {topic} played a big role in your life?",
            "Tell me a secret about your love for {topic}.",
        ],
        "flirty": [
            "What's the most adventurous thing you've done involving {topic}?",
            "If we were to explore {topic} together, what would we do?",
            "Does {topic} ever get you in a flirty mood?",
        ],
    },
    # Ways to build intimacy and connection
    "intimacy": {
        "validation": [
            "I really like how you think about {topic}.",
            "It's cool that you're so passionate about {topic}.",
            "Thanks for sharing that with me, I feel like I understand you better now.",
        ],
        "shared_experience": [
            "I feel the exact same way about {topic}!",
            "That's so funny, I had a similar experience with {topic}.",
            "We should totally do {topic} together sometime.",
        ],
        "vulnerability": [
            "To be honest, I've always been a little nervous about {topic}.",
            "I'm opening up here, but I've always wanted to try {topic}.",
            "You make me feel comfortable talking about things like {topic}.",
        ],
    },
    # Suggestions with a sexual or romantic undertone
    "sexual": {
        "playful_tease": [
            "I have a feeling that talking about {topic} with you could be dangerous... in a good way.",
            "Is it just me, or is there some tension in this conversation about {topic}?",
            "You have a really captivating energy when you talk about {topic}.",
        ],
        "direct_desire": [
            "Talking about {topic} is making me think about you in a different way.",
            "I can't help but imagine what it would be like to explore {topic} with you.",
            "Let's just say my imagination is running wild thinking about you and {topic}.",
        ],
        "romantic_connection": [
            "I feel a real spark talking with you about {topic}.",
            "Our chemistry is undeniable, especially when we discuss things like {topic}.",
            "I feel like we could have a really special connection, and it starts with conversations like this.",
        ],
    },
}

# General fallback suggestions for when context is weak
FALLBACK_SUGGESTIONS = {
    "topics": [
        "What's the most spontaneous thing you've ever done?",
        "What's a skill you'd love to learn?",
        "Tell me about a travel destination that's on your bucket list.",
    ],
    "questions": [
        "What's something that always makes you smile?",
        "What's your go-to way to de-stress after a long week?",
        "What's a movie you can watch over and over again?",
    ],
    "intimacy": [
        "I'm really enjoying this conversation.",
        "You have a great sense of humor.",
        "I feel like I could talk to you for hours.",
    ],
    "sexual": [
        "You have a really attractive energy.",
        "I'm definitely intrigued by you.",
        "There's a definite spark between us.",
    ],
}


def _get_recent_and_salient_topics(
    identified_topics: List[Dict[str, Any]],
    analysis_data: Dict[str, Any],
    count: int = 3
) -> List[Dict[str, Any]]:
    """Gets the most recent and talked-about topics."""
    topic_map = {t["canonical_name"]: t for t in identified_topics}

    recency = analysis_data.get("contextual_features", {}).get("topic_recency", {})
    saliency = analysis_data.get("contextual_features", {}).get("topic_saliency", {})

    # Combine scores, giving more weight to recency
    combined_scores = {
        name: (1 / recency.get(name, 10)) + (saliency.get(name, 0) * 0.1)
        for name in topic_map.keys()
    }

    sorted_topics = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    top_topic_names = [name for name, score in sorted_topics[:count]]

    return [topic_map[name] for name in top_topic_names if name in topic_map]


def _generate_suggestions_for_category(
    category: str,
    topics: List[Dict[str, Any]],
    used_suggestions: Set[str],
    max_suggestions: int = 5
) -> List[str]:
    """Generates a list of suggestions for a specific category (topics, questions, etc.)."""
    suggestions = []

    for topic in topics:
        if len(suggestions) >= max_suggestions:
            break

        topic_name = topic.get("canonical_name", "this")
        topic_category = topic.get("category", "neutral")

        # Determine the intent/strategy based on the topic's category
        strategy = "playful" # default
        if category == "questions":
            strategy = "open_ended"
            if topic_category == "sexual":
                strategy = "flirty"
            elif topic_category in ["sensitive", "focus"]:
                strategy = "personal"
        elif category == "intimacy":
            strategy = "validation"
            if topic_category == "sexual":
                strategy = "shared_experience"
        elif category == "sexual":
            strategy = "playful_tease"
            if topic_category == "sexual":
                strategy = "direct_desire"

        # Get a template and format it
        template = random.choice(SUGGESTION_TEMPLATES[category].get(strategy, []))
        if template:
            suggestion = template.format(topic=topic_name)
            if suggestion not in used_suggestions:
                suggestions.append(suggestion)
                used_suggestions.add(suggestion)

    # Fill with fallbacks if needed
    while len(suggestions) < max_suggestions:
        fallback = random.choice(FALLBACK_SUGGESTIONS[category])
        if fallback not in used_suggestions:
            suggestions.append(fallback)
            used_suggestions.add(fallback)
            
    return suggestions


def generate_suggestions(
    analysis_data: Dict[str, Any],
    conversation_turns: List[Dict[str, Any]],
    identified_topics: List[Dict[str, Any]],
    feedback: Optional[List[Feedback]] = None # Feedback is kept for future model-based versions
) -> Dict[str, List[str]]:
    """
    Generates creative, context-aware suggestions based on a stateless, rule-based engine.
    """
    # Get the top topics to focus on for suggestions
    contextual_topics = _get_recent_and_salient_topics(identified_topics, analysis_data)

    # Get all topic names for generating new topic ideas
    all_topic_names = [t["canonical_name"] for t in identified_topics]

    # If no topics were identified, use generic placeholders
    if not contextual_topics:
        contextual_topics = [{"canonical_name": "this conversation", "category": "neutral"}]
    if not all_topic_names:
        all_topic_names = ["travel", "movies", "passions", "dreams", "adventures"]

    # Generate suggestions for each category
    used_suggestions: Set[str] = set()
    final_suggestions = {}

    # For 'topics', suggest topics that are NOT the current ones
    other_topics = [
        {"canonical_name": name, "category": "neutral"}
        for name in all_topic_names
        if name not in {t["canonical_name"] for t in contextual_topics}
    ]
    if not other_topics: # If all topics are contextual, use fallbacks
        other_topics = [{"canonical_name": name, "category": "neutral"} for name in ["a new adventure", "a shared secret"]]

    final_suggestions["topics"] = _generate_suggestions_for_category("topics", other_topics, used_suggestions)
    final_suggestions["questions"] = _generate_suggestions_for_category("questions", contextual_topics, used_suggestions)
    final_suggestions["intimacy"] = _generate_suggestions_for_category("intimacy", contextual_topics, used_suggestions)
    final_suggestions["sexual"] = _generate_suggestions_for_category("sexual", contextual_topics, used_suggestions)

    return final_suggestions
