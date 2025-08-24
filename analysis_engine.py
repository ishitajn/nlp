# In analysis_engine.py
from typing import List, Dict, Any
from collections import Counter

from topic_engine import identify_and_canonicalize_topics, score_and_categorize_topics
from context_engine import extract_contextual_features
from behavioral_engine import analyze_conversation_behavior

def run_full_analysis(
    my_profile: str,
    their_profile: str,
    processed_turns: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Orchestrates the full analysis pipeline with the final, robust topic engine.
    """
    
    # 1. Extract and canonicalize topics
    topic_map, profile_topics = identify_and_canonicalize_topics(processed_turns, their_profile)
    canonical_topics = list(topic_map.keys())

    # 2. Calculate Recency, Frequency, and Salience for all topics
    topic_last_seen = {}
    for i, turn in enumerate(processed_turns):
        content = turn.get("content", "").lower()
        for topic in canonical_topics:
            if topic in content:
                topic_last_seen[topic] = i
    
    recent_topics_sorted = sorted(topic_last_seen.items(), key=lambda item: item[1], reverse=True)
    recent_topics = [topic.title() for topic, index in recent_topics_sorted[:10]]
    
    # The single "focus" topic is the most recent one.
    focus_topic = recent_topics_sorted[0][0] if recent_topics_sorted else ""

    topic_frequency = Counter({topic: len(turns) for topic, turns in topic_map.items()})
    topic_salience = {
        topic: topic_frequency.get(topic, 0) * (1 + (topic_last_seen.get(topic, 0) / len(processed_turns)))
        for topic in canonical_topics
    }

    # 3. Classify topics using the calculated focus and salience scores
    categorized_topics = score_and_categorize_topics(
        topic_map=topic_map,
        profile_topics=profile_topics,
        focus_topic=focus_topic,
        topic_salience=topic_salience
    )

    # 4. Run other analysis modules
    behavioral_analysis = analyze_conversation_behavior(processed_turns)
    contextual_features = extract_contextual_features(
        conversation_turns=processed_turns,
        identified_topics_map=categorized_topics,
        my_profile=my_profile,
        their_profile=their_profile
    )

    # 5. Assemble the final analysis object
    final_analysis = {
        "categorized_topics": categorized_topics,
        "recent_topics": recent_topics,
        "contextual_features": contextual_features,
        "behavioral_analysis": behavioral_analysis,
        "topic_map": topic_map 
    }
    
    return final_analysis