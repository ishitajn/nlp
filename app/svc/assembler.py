from typing import Dict, Any, List

def build_final_json(
    payload: Dict[str, Any],
    topics: Dict[str, Any],
    geo: Dict[str, Any],
    suggestions: Dict[str, List[str]],
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Assembles all the analysis components into the final unified JSON output.
    This structure is based on the high-level system overview.
    """

    # The final JSON schema needs to be well-defined.
    # Based on the prompt, the output is a "Unified JSON" with keys:
    # - topics
    # - geo
    # - predictions_and_suggestions
    # - analysis (includes sexual/intimacy + robust engagement metrics)

    final_output = {
        "matchId": payload.get("match_id"),
        "analysis_timestamp": payload.get("timestamp"), # We should add a timestamp to the payload

        "topics": {
            "discovered_topics": topics.get("topics", []),
            "turn_topic_map": topics.get("turn_to_topic_map", {})
        },

        "geo": geo,

        "predictions_and_suggestions": {
            "talking_points": suggestions.get("talking_points", []),
            "questions": suggestions.get("questions", []),
            "sexual_intimacy_suggestions": suggestions.get("sexual_intimacy_suggestions", []),
            "next_actions": suggestions.get("next_actions", [])
        },

        "analysis": {
            "sexual_intimacy_context": features.get("sexual_intimacy_flags", {}),
            "engagement_metrics": {
                "match_engagement_level": features.get("match_engagement_level", {}),
                "engagement_signals": features.get("engagement_signals", {})
            },
            "general_probes": features.get("general_probes", {}),
            "conversation_stats": features.get("conversation_stats", {})
        }
    }

    return final_output
