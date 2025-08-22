# In app/svc/assembler.py

from typing import Dict, Any, List

def build_final_json(
    payload: Dict[str, Any],
    analysis_data: Dict[str, Any],
    suggestions: Dict[str, List[Dict[str, Any]]],
    geo: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Assembles the final JSON using the rich, dynamic analysis data structure.
    """
    # The analysis_data now contains the dynamically generated conversation_state
    conversation_state = analysis_data.get("conversation_state", {
        "topics": {},
        "recent_topics": []
    })

    # Geo object construction
    my_location_parts = payload.get("ui_settings", {}).get("myLocation", ",").split(',')
    their_location_parts = payload.get("scraped_data", {}).get("theirLocationString", ",").split(',')
    geo_output = {
        "userLocation": { "city": my_location_parts[0].strip(), "country": my_location_parts[-1].strip(), "timeOfDay": geo.get("my_time_of_day"), "current_date_time": geo.get("my_local_time") },
        "matchLocation": { "city": their_location_parts[0].strip(), "country": their_location_parts[-1].strip(), "timeOfDay": geo.get("their_time_of_day"), "current_date_time": geo.get("their_local_time") },
        "isVirtual": bool(geo.get("time_difference_hours", 0) != 0 or geo.get("country_difference")),
        "timeZoneDifference": geo.get("time_difference_hours"),
        "countryDifference": geo.get("country_difference")
    }

    # The final, detailed analysis object, separate from the simplified conversation_state
    final_analysis_object = {
        "detected_phases": analysis_data.get("detected_phases", []),
        "detected_tones": analysis_data.get("detected_tones", []),
        "detected_intents": analysis_data.get("detected_intents"),
        "sentiment": analysis_data.get("sentiment_analysis", {}).get("overall", "neutral"),
        "flirtation_score": analysis_data.get("sentiment_analysis", {}).get("flirtation_score", 0.0)
    }

    # Format suggestions into the desired final format: Dict[str, List[str]]
    final_suggestions = {
        category: [s['text'] for s in suggestion_list]
        for category, suggestion_list in suggestions.items()
    }

    # Assemble the final, unified JSON object
    final_output = {
        "matchId": payload.get("matchId"),
        "conversation_state": conversation_state,
        "geo": geo_output,
        "suggestions": final_suggestions,
        "analysis": final_analysis_object,
        "pipeline": "enhanced_hybrid_nlp_v6_retrieval" # Updated pipeline name
    }

    return final_output