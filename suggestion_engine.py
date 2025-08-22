# In app/svc/suggestion_engine.py

import random
from typing import Dict, Any, List

# ==============================================================================
# == THE DEFINITIVE, UNCENSORED & EXPANDED SUGGESTION BANK                    ==
# ==============================================================================
SUGGESTION_TEMPLATES = {
    "topics": {
        "contextual": ["Playful fantasy about {topic}", "Deeper dive into {topic}", "Adventurous take on {topic}", "A secret desire related to {topic}", "Connect {topic} to a personal story"],
        "escalation": ["Ideal late-night vibes", "Sharing a secret desire", "Discussing 'our' chemistry", "A fantasy scenario you've imagined", "What turns you on most"],
        "rapport": ["Favorite travel memories", "A skill they want to learn", "Their passion projects", "Childhood dreams or ambitions", "What makes them feel alive"],
        "fallback": ["Most adventurous thing they've done", "Their ideal perfect day", "A hidden talent they have", "Favorite type of humor", "A guilty pleasure"],
    },
    "questions": {
        "contextual": ["Favorite memory of {topic}?", "How does {topic} make you feel?", "A secret about your love for {topic}?", "Wildest experience with {topic}?", "Connect {topic} to a fantasy?"],
        "escalation": ["What makes you feel most desired?", "A fantasy you've never shared?", "What's our first adventure?", "Ideal cozy night together?", "Biggest turn-on?"],
        "rapport": ["What drives your passion for {topic}?", "A memory tied to {topic}?", "How did you get into {topic}?", "What does {topic} mean to you?", "Proudest moment related to {topic}?"],
        "fallback": ["Something you're curious about?", "What always makes you laugh?", "Biggest goal for this year?", "A story you love to tell?", "What's your love language?"],
    },
    "intimacy": {
        "contextual": ["Compliment their view on {topic}", "Relate {topic} to a personal feeling", "Appreciate their passion for {topic}", "Share a vulnerability about {topic}", "Connect {topic} to a shared future"],
        "validation": ["Acknowledge their irresistible energy", "Compliment their personality deeply", "Appreciate their openness and trust", "Validate their feelings or perspective", "Express how they make you feel"],
        "connection": ["Share a small, personal secret", "Mention a specific future experience", "Reference an inside joke we have", "Express genuine appreciation for them", "Ask about their emotional needs"],
        "fallback": ["Compliment their way of thinking", "Mention their great energy", "Acknowledge their unique humor", "Express enjoyment of the conversation", "Ask what makes them feel connected"],
    },
    "sexual": {
        "high_tension": ["Imagining feeling your body during {topic}", "Your energy is making me ache for you", "Describe a sensual act involving {topic}", "Can't stop thinking about you naked", "Tell me a dirty secret"],
        "medium_tension": ["Playfully tease about their charm", "Suggestive compliment on their vibe", "Hint at our 'dangerous' chemistry", "Curiosity about their morning energy", "What if we were together right now..."],
        "low_tension": ["A playful, flirty compliment", "Mentioning the obvious spark", "A cheeky 'what if' scenario", "A suggestive emoji 😉", "You have a captivating smile"],
        "fallback": ["Compliment their confidence", "Acknowledge their attractive energy", "A simple, playful wink emoji", "Mentioning a 'spark'", "You seem like fun"],
    }
}

# ==============================================================================
# == THE DEFINITIVE SUGGESTION STRATEGY MAP                                   ==
# ==============================================================================
# This is the brain. It links canonical topics to appropriate suggestion strategies.
SUGGESTION_STRATEGY_MAP = {
    "Career & Ambition": {"topics": ["rapport"], "questions": ["rapport"], "intimacy": ["validation"], "sexual": ["low_tension"]},
    "Family & Background": {"topics": ["rapport"], "questions": ["rapport"], "intimacy": ["connection"], "sexual": ["low_tension"]},
    "Hobbies & Passions": {"topics": ["rapport", "contextual"], "questions": ["contextual"], "intimacy": ["contextual"], "sexual": ["medium_tension"]},
    "Travel & Adventure": {"topics": ["rapport", "contextual"], "questions": ["contextual"], "intimacy": ["connection"], "sexual": ["medium_tension"]},
    "Fitness & Health": {"topics": ["rapport"], "questions": ["rapport"], "intimacy": ["validation"], "sexual": ["low_tension"]},
    "Food & Drink": {"topics": ["rapport", "contextual"], "questions": ["contextual"], "intimacy": ["connection"], "sexual": ["medium_tension"]},
    "Flirting & Compliments": {"topics": ["escalation"], "questions": ["escalation"], "intimacy": ["validation", "contextual"], "sexual": ["medium_tension", "high_tension"]},
    "Deeper Connection": {"topics": ["rapport"], "questions": ["rapport"], "intimacy": ["validation", "connection"], "sexual": ["medium_tension"]},
    "Making Plans & Logistics": {"topics": ["escalation"], "questions": ["escalation"], "intimacy": ["connection"], "sexual": ["high_tension"]},
    "Sexual Escalation & Kinks": {"topics": ["escalation"], "questions": ["escalation"], "intimacy": ["validation"], "sexual": ["high_tension"]},
    "Pop Culture & Media": {"topics": ["rapport"], "questions": ["rapport"], "intimacy": ["connection"], "sexual": ["low_tension"]},
    "Inside Jokes & Nicknames": {"topics": ["contextual"], "questions": ["contextual"], "intimacy": ["connection"], "sexual": ["medium_tension"]},
}

def _get_weighted_topics(occurrence: Dict[str, int], recency: Dict[str, int], count: int) -> List[str]:
    """Combines occurrence and recency for a human-like topic ranking."""
    scores = {}
    all_heatmap_topics = set(occurrence.keys()) | set(recency.keys())
    for topic in all_heatmap_topics:
        num = occurrence.get(topic, 0)
        recency_rank = recency.get(topic, 100)
        score = (1 / recency_rank) * 10 + num
        scores[topic] = score
    
    sorted_topics = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [topic for topic, score in sorted_topics[:count]]

def generate_suggestions(analysis_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Generates 5 creative, context-aware suggestions for each category using a rule-based engine.
    """
    suggestions = { "topics": [], "questions": [], "intimacy": [], "sexual": [] }
    
    # --- 1. Get Rich Context from Analysis ---
    phases = set(analysis_data.get("detected_phases", []))
    occurrence = analysis_data.get("conversation_state", {}).get("topic_occurrence_heatmap", {})
    recency = analysis_data.get("conversation_state", {}).get("topic_recency_heatmap", {})
    topic_mapping = analysis_data.get("conversation_state", {}).get("topic_mapping", {})
    
    # --- 2. Determine Top 5 Most Relevant Canonical Topics ---
    top_canonical_topics = _get_weighted_topics(occurrence, recency, 5)

    # --- 3. Generate Suggestions Based on Top Topics ---
    for category in suggestions.keys():
        generated_count = 0
        # Use a set to avoid duplicate suggestions
        category_suggestions = set()
        
        # Iterate through the most relevant topics first
        for canonical_topic in top_canonical_topics:
            if generated_count >= 5: break
            
            # Get the strategy for this topic and category
            strategies = SUGGESTION_STRATEGY_MAP.get(canonical_topic, {}).get(category, [])
            if not strategies: continue

            # Get the actual raw topics that were mapped to this canonical topic
            raw_topics = topic_mapping.get(canonical_topic)
            if not raw_topics: continue
            
            # Pick a strategy and a template
            strategy = random.choice(strategies)
            template = random.choice(SUGGESTION_TEMPLATES[category][strategy])
            
            # Format it with a real, raw topic
            raw_topic_to_use = random.choice(raw_topics)
            suggestion = template.format(topic=raw_topic_to_use)
            
            if suggestion not in category_suggestions:
                category_suggestions.add(suggestion)
                generated_count += 1
        
        suggestions[category] = list(category_suggestions)

    # --- 4. Fallback Pass: Ensure every category has exactly 5 suggestions ---
    for category, items in suggestions.items():
        if len(items) < 5:
            needed = 5 - len(items)
            # Use a broader set of fallbacks based on conversation phase
            phase_strategy_map = {
                "Icebreaker": "rapport",
                "Rapport Building": "rapport",
                "Escalation": "escalation",
                "Explicit Banter": "high_tension" if category == "sexual" else "escalation",
                "Logistics": "escalation"
            }
            
            fallback_strategy = "fallback"
            for phase, strategy in phase_strategy_map.items():
                if phase in phases:
                    # Ensure the strategy exists for the category
                    if strategy in SUGGESTION_TEMPLATES[category]:
                        fallback_strategy = strategy
                        break
            
            fallback_templates = SUGGESTION_TEMPLATES[category][fallback_strategy]
            # Add unique fallbacks until we have 5
            for _ in range(needed):
                suggestion = random.choice(fallback_templates)
                if "{topic}" in suggestion: # Handle fallback templates that might have a topic
                    suggestion = suggestion.format(topic="your personality")
                if suggestion not in suggestions[category]:
                    suggestions[category].append(suggestion)
                if len(suggestions[category]) >= 5: break

    return suggestions