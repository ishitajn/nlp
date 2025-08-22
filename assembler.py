# In app/svc/assembler.py

from typing import Dict, Any, List

def build_final_json(
    payload: Dict[str, Any],
    analysis_data: Dict[str, Any],
    suggestions: Dict[str, List[str]],
    geo: Dict[str, Any]
) -> Dict[str, Any]:
    
    # Extract the full conversation_state, which now includes the mapping
    conversation_state = analysis_data.get("conversation_state", {
        "topics": {"focus": [], "avoid": [], "neutral": [], "sensitive": [], "fetish": [], "sexual": [], "inside_jokes": []},
        "recent_topics": [], "topic_occurrence_heatmap": {}, "topic_recency_heatmap": {}, "topic_mapping": {}
    })

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

    final_analysis_object = {
        "sentiment": analysis_data.get("sentiment_analysis", {}).get("overall", "neutral"),
        "flirtation_level": analysis_data.get("engagement_metrics", {}).get("flirtation_level", "low"),
        "engagement": analysis_data.get("engagement_metrics", {}).get("level", "low"),
        "pace": analysis_data.get("engagement_metrics", {}).get("pace", "steady"),
    }

    final_output = {
        "matchId": payload.get("matchId"),
        "conversation_state": conversation_state,
        "geo": geo_output,
        "suggestions": suggestions,
        "analysis": final_analysis_object,
        "sentiment": { "overall": final_analysis_object["sentiment"] },
        "pipeline": "enhanced_semantic_v8_final"
    }

    return final_output