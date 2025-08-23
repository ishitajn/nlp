# In analysis_engine.py
from typing import List, Dict, Any

# Import the new, modular engines
from topic_engine import identify_topics
from context_engine import extract_contextual_features

def run_full_analysis(
    my_profile: str,
    their_profile: str,
    processed_turns: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Orchestrates the full conversation analysis pipeline.
    
    1. Identifies topics from the conversation.
    2. Extracts contextual features based on the conversation and topics.
    3. Assembles the final analysis dictionary.
    """
    
    # 1. Identify topics using the new topic engine
    # The conversation turns are expected to be pre-processed at this point
    identified_topics = identify_topics(processed_turns)
    
    # 2. Extract contextual features using the new context engine
    contextual_features = extract_contextual_features(
        conversation_turns=processed_turns,
        identified_topics=identified_topics,
        my_profile=my_profile,
        their_profile=their_profile
    )
    
    # 3. Assemble the final analysis object
    # The format should be clean and directly usable by the suggestion engine
    final_analysis = {
        "identified_topics": identified_topics,
        "contextual_features": contextual_features
    }
    
    return final_analysis