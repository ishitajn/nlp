# In analysis_engine.py
from typing import List, Dict, Any
from collections import Counter

from topic_engine import identify_and_canonicalize_topics, score_and_categorize_topics
from context_engine import extract_contextual_features
from behavioral_engine import analyze_conversation_behavior, analyze_last_message_details

def run_full_analysis(
    my_profile: str,
    their_profile: str,
    processed_turns: List[Dict[str, Any]],
    use_enhanced_nlp: bool = False
) -> Dict[str, Any]:
    """
    Orchestrates the full analysis pipeline, including new memory and last-message features.
    """
    # 1. Topic Analysis (unchanged)
    topic_map, profile_topics = identify_and_canonicalize_topics(
        processed_turns, their_profile, use_enhanced_nlp=use_enhanced_nlp
    )
    canonical_topics = list(topic_map.keys())
    turn_to_index = {frozenset(turn.items()): i for i, turn in enumerate(processed_turns)}
    topic_last_seen = {
        topic: max(valid_indices) for topic, turns in topic_map.items()
        if turns and (valid_indices := [i for i in [turn_to_index.get(frozenset(t.items())) for t in turns] if i is not None])
    }
    recent_topics_sorted = sorted(topic_last_seen.items(), key=lambda item: item[1], reverse=True)
    recent_topics = [topic.title() for topic, index in recent_topics_sorted[:10]]
    focus_topic = recent_topics_sorted[0][0] if recent_topics_sorted else ""
    topic_frequency = Counter({topic: len(turns) for topic, turns in topic_map.items()})
    topic_salience = {
        topic: topic_frequency.get(topic, 0) * (1 + (topic_last_seen.get(topic, 0) / len(processed_turns)))
        for topic in canonical_topics
    }
    categorized_topics = score_and_categorize_topics(
        topic_map=topic_map, profile_topics=profile_topics, focus_topic=focus_topic,
        topic_salience=topic_salience, use_enhanced_nlp=use_enhanced_nlp
    )

    # 2. Run other analysis modules
    behavioral_analysis = analyze_conversation_behavior(processed_turns, use_enhanced_nlp=use_enhanced_nlp)
    contextual_and_memory_features = extract_contextual_features(
        conversation_turns=processed_turns,
        identified_topics_map=categorized_topics,
        my_profile=my_profile,
        their_profile=their_profile,
        use_enhanced_nlp=use_enhanced_nlp
    )

    # 3. Run New, Detailed Last Message Analysis
    last_turn = processed_turns[-1] if processed_turns else None
    last_message_analysis = analyze_last_message_details(last_turn)

    # 4. Assemble the final analysis object
    final_analysis = {
        "categorized_topics": categorized_topics,
        "recent_topics": recent_topics,
        "contextual_features": contextual_and_memory_features, # This now contains memory_features
        "behavioral_analysis": behavioral_analysis,
        "last_message_analysis": last_message_analysis,
        "topic_map": topic_map 
    }
    
    return final_analysis