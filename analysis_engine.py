# In analysis_engine.py
from typing import List, Dict, Any

# Import the new, modular engines
from topic_engine import identify_topics
from context_engine import extract_contextual_features
from behavioral_engine import analyze_conversation_behavior
from topic_categorizer import score_and_categorize_topics

def run_full_analysis(
    my_profile: str,
    their_profile: str,
    processed_turns: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Orchestrates the full conversation analysis pipeline.
    """
    
    # 1. Identify raw topic clusters
    topic_clusters = identify_topics(processed_turns, their_profile)
    
    # 2. Extract contextual features (sentiment, recency, etc.)
    contextual_features = extract_contextual_features(
        conversation_turns=processed_turns,
        identified_topics=topic_clusters, # Use raw clusters here
        my_profile=my_profile,
        their_profile=their_profile
    )
    
    # 3. Analyze behavioral patterns
    behavioral_analysis = analyze_conversation_behavior(
        conversation_turns=processed_turns,
        identified_topics=topic_clusters
    )

    # 4. Score and categorize topics using the new engine
    categorized_topics = score_and_categorize_topics(
        topic_clusters=topic_clusters,
        # conversation_turns=processed_turns,
        context=contextual_features
    )

    # 5. Integrate categorization back into topic clusters
    # Create a reverse map from topic name to its category
    topic_to_category_map = {
        topic_name: category
        for category, topic_list in categorized_topics.items()
        for topic_name in topic_list
    }

    # Update the category for each topic in the original list
    for topic in topic_clusters:
        topic_name = topic.get("canonical_name")
        if topic_name in topic_to_category_map:
            topic["category"] = topic_to_category_map[topic_name]

    # 6. Assemble the final analysis object
    final_analysis = {
        "topic_clusters": topic_clusters, # Now with categories
        "categorized_topics": categorized_topics,
        "contextual_features": contextual_features,
        "behavioral_analysis": behavioral_analysis
    }
    
    return final_analysis