# In analysis_engine.py
from typing import List, Dict, Any

# Import the new, modular engines
from topic_engine import identify_topics
from context_engine import extract_contextual_features
from behavioral_engine import analyze_conversation_behavior
from scoring_engine import score_and_categorize_topics

def run_full_analysis(
    my_profile: str,
    their_profile: str,
    processed_turns: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Orchestrates the full conversation analysis pipeline.
    """
    
    # 1. Identify raw topic clusters
    topic_clusters = identify_topics(processed_turns)
    
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

    # 5. Assemble the final analysis object
    final_analysis = {
        "topic_clusters": topic_clusters, # Raw topics
        "categorized_topics": categorized_topics, # New categorized topics
        "contextual_features": contextual_features,
        "behavioral_analysis": behavioral_analysis
    }
    
    return final_analysis