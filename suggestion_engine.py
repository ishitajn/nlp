"""
This engine is responsible for generating flags and instructions for a future generative model,
based on UI settings and conversation analysis.
"""
from typing import Dict, Any

from model import UISettings

def generate_suggestion_flags(
    ui_settings: UISettings,
    analysis_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates a dictionary of flags to guide a future generative model.
    """
    behavioral_analysis = analysis_data.get("behavioral_analysis", {})

    # Start with flags directly from UI settings
    flags = {
        "end_with_question": ui_settings.end_with_question,
        "suggest_new_topic": ui_settings.new_topic,
        "enforce_strict_goal": ui_settings.strict_goal_override,
    }

    # Add flags derived from analysis
    # If engagement is low, strongly suggest a new topic, overriding the UI if necessary.
    if behavioral_analysis.get("suggest_topic_shift"):
        flags["suggest_new_topic"] = True

    # If the match just asked a question, the priority is to answer it, not ask another one.
    if behavioral_analysis.get("match_last_message_has_question"):
        flags["end_with_question"] = False # Prioritize answering

    return flags
