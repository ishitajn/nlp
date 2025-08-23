# In assembler.py
from typing import Dict, Any, List
from collections import defaultdict

def build_final_json(
    payload: Dict[str, Any],
    analysis_data: Dict[str, Any],
    suggestions: Dict[str, List[str]],
    geo: Dict[str, Any]
) -> Dict[str, Any]:
    
    # Extract data from the different engines
    context = analysis_data.get("contextual_features", {})
    topics = analysis_data.get("identified_topics", [])
    behavior = analysis_data.get("behavioral_analysis", {})

    # --- Reconstruct conversation_state to match spec ---

    # Create topic_mapping from identified_topics
    topic_mapping = defaultdict(lambda: {'keywords': set(), 'messages': []})
    for topic in topics:
        category = topic.get("category", "Uncategorized")
        topic_mapping[category]['keywords'].update(topic.get("keywords", []))
        topic_mapping[category]['messages'].extend(topic.get("messages", []))

    # Convert sets to lists for JSON serialization
    final_topic_mapping = {cat: {'keywords': list(kw), 'messages': msgs} for cat, {'keywords': kw, 'messages': msgs} in topic_mapping.items()}

    # Get recency heatmap (rank 1 is most recent)
    topic_recency_heatmap = context.get("topic_recency", {})
    recent_topics = sorted(topic_recency_heatmap.keys(), key=lambda k: topic_recency_heatmap[k])

    conversation_state = {
        "identified_topics": [
            {
                "topic": t.get("canonical_name"),
                "category": t.get("category")
            } for t in topics
        ],
        "topics": {}, # Meta-topic analysis - This is vague, will leave empty for now.
        "recent_topics": recent_topics,
        "topic_occurrence_heatmap": context.get("topic_saliency", {}),
        "topic_recency_heatmap": topic_recency_heatmap,
        "topic_mapping": final_topic_mapping
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