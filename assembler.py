# In assembler.py
from typing import Dict, Any, List
from collections import defaultdict

def build_final_json(
    payload: Dict[str, Any],
    analysis_data: Dict[str, Any],
    suggestions: Dict[str, str], # Now receives stringified JSON
    geo: Dict[str, Any]
) -> Dict[str, Any]:
    
    # Extract data from the different engines
    context = analysis_data.get("contextual_features", {})
    # This is the output of the scoring engine
    categorized_topics = analysis_data.get("categorized_topics", {})
    behavior = analysis_data.get("behavioral_analysis", {})

    # --- Build conversation_state to match new spec ---
    topic_recency_heatmap = context.get("topic_recency", {})
    recent_topics = sorted(topic_recency_heatmap.keys(), key=lambda k: topic_recency_heatmap[k])

    # Ensure all keys are present in the topics object, even if empty
    final_topics_object = {
        "focus": categorized_topics.get("focus", []),
        "avoid": categorized_topics.get("avoid", []),
        "neutral": categorized_topics.get("neutral", []),
        "sensitive": categorized_topics.get("sensitive", []),
        "fetish": categorized_topics.get("fetish", []),
        "sexual": categorized_topics.get("sexual", [])
    }

    conversation_state = {
        "topics": final_topics_object,
        "recent_topics": recent_topics
    }

    # --- Build geo object ---
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
    } if geo else None

    # --- Build analysis object ---
    engagement_metrics = context.get("engagement_metrics", {})
    final_analysis_object = {
        "sentiment": context.get("sentiment_analysis", {}).get("overall", "neutral"),
        "flirtation_level": engagement_metrics.get("flirtation_level", "low"),
        "engagement": engagement_metrics.get("level", "low"),
        "pace": engagement_metrics.get("pace", "steady"),
    }

    # --- Build final output ---
    final_output = {
        "matchId": payload.get("matchId"),
        "conversation_state": conversation_state,
        "geo": geo_output,
        "suggestions": suggestions,
        "analysis": final_analysis_object,
        "sentiment": { "overall": final_analysis_object["sentiment"] },
        "conversation_analysis": behavior, # Add the new section
        "pipeline": "modular_semantic_v2.0" # Update pipeline version
    }

    return final_output