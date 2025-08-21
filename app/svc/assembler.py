from typing import Dict, Any, List

def build_final_json(
    payload: Dict[str, Any],
    topics: Dict[str, Any],
    geo: Dict[str, Any],
    suggestions: Dict[str, List[str]],
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Assembles all analysis components into the new, detailed JSON schema.
    """
    # The 'features' dict now contains 'analysis' and 'sentiment' objects.
    analysis = features.get("analysis", {})
    sentiment = features.get("sentiment", {})

    # The 'geo' object from the planner needs to be restructured.
    # Note: The planner doesn't provide city/country separately, so we derive it.
    my_location_parts = payload.get("ui_settings", {}).get("myLocation", ",").split(',')
    their_location_parts = payload.get("scraped_data", {}).get("theirLocationString", ",").split(',')

    geo_output = {
        "userLocation": {
            "city": my_location_parts[0].strip(),
            "country": my_location_parts[-1].strip(),
            "timeOfDay": geo.get("my_time_of_day"),
            "current_date_time": geo.get("my_local_time")
        },
        "matchLocation": {
            "city": their_location_parts[0].strip(),
            "country": their_location_parts[-1].strip(),
            "timeOfDay": geo.get("their_time_of_day"),
            "current_date_time": geo.get("their_local_time")
        },
        "isVirtual": bool(geo.get("time_difference_hours", 0) != 0 or geo.get("country_difference")),
        "timeZoneDifference": geo.get("time_difference_hours"),
        "countryDifference": geo.get("country_difference")
    }

    # The 'topics' object is now the 'conversation_state'.
    conversation_state = {
        "topics": {
            "focus": topics.get("focus", []),
            "avoid": topics.get("avoid", []),
            "neutral": topics.get("neutral", []),
            "sensitive": topics.get("sensitive", []),
            "fetish": topics.get("fetish", []),
            "sexual": topics.get("sexual", [])
        },
        "recent_topics": topics.get("recent_topics", [])
    }

    # The 'suggestions' object from the generator maps directly.
    suggestions_output = {
        "topics": suggestions.get("topics", []),
        "questions": suggestions.get("questions", []),
        "sexual": suggestions.get("sexual", []),
        "intimacy": suggestions.get("intimacy", [])
    }

    # Assemble the final, unified JSON object
    final_output = {
        "matchId": payload.get("matchId"),
        "conversation_state": conversation_state,
        "geo": geo_output,
        "suggestions": suggestions_output,
        "analysis": analysis,
        "sentiment": sentiment,
        "pipeline": "enhanced" # Hardcoded as per the spec
    }

    return final_output
