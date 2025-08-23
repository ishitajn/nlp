# In app/svc/assembler.py

from typing import Dict, Any, List

def build_final_json(
    payload: Dict[str, Any],
    analysis_data: Dict[str, Any],
    suggestions: Dict[str, List[str]],
    geo: Dict[str, Any]
) -> Dict[str, Any]:
    
    context = analysis_data.get("contextual_features", {})
    topics = analysis_data.get("identified_topics", [])

    # Create a new, clean conversation state from the new data structures
    conversation_state = {
        "identified_topics": [
            {
                "topic": t.get("canonical_name"),
                "category": t.get("category"),
                "keywords": t.get("keywords"),
                "message_count": t.get("message_count")
            } for t in topics
        ],
        "topic_saliency": context.get("topic_saliency", {}),
        "topic_recency": context.get("topic_recency", {}),
        "detected_phases": context.get("detected_phases", []),
        "detected_tones": context.get("detected_tones", []),
        "detected_intents": context.get("detected_intents", []),
    }

    geo_output = {
        "userLocation": {
            "City": geo.get("my_location", {}).get("city_state", "N/A"), "Current Time": geo.get("my_location", {}).get("current_time", "N/A"),
            "Time of Day": geo.get("my_location", {}).get("time_of_day", "N/A"), "Time Zone": geo.get("my_location", {}).get("timezone", "N/A"),
            "Country": geo.get("my_location", {}).get("country", "N/A"),
        },
        "matchLocation": {
            "City": geo.get("their_location", {}).get("city_state", "N/A"), "Current Time": geo.get("their_location", {}).get("current_time", "N/A"),
            "Time of Day": geo.get("their_location", {}).get("time_of_day", "N/A"), "Time Zone": geo.get("their_location", {}).get("timezone", "N/A"),
            "Country": geo.get("their_location", {}).get("country", "N/A"),
        },
        "Time Difference": f"{geo.get('time_difference_hours', 'N/A')} hours",
        "Distance": f"{geo.get('distance_km', 'N/A')} km"
    }

    # Simplified analysis object
    final_analysis_object = {
        "sentiment": context.get("sentiment_analysis", {}).get("overall", "neutral"),
        "sentiment_score": context.get("sentiment_analysis", {}).get("compound_score", 0.0),
        "user_turn_count": context.get("speaker_metrics", {}).get("user_turn_count", 0),
        "their_turn_count": context.get("speaker_metrics", {}).get("their_turn_count", 0),
    }

    final_output = {
        "matchId": payload.get("matchId"),
        "conversation_state": conversation_state,
        "geo": geo_output,
        "suggestions": suggestions,
        "analysis": final_analysis_object,
        "pipeline": "modular_semantic_v1.5" # Updated pipeline name
    }

    return final_output