def recommend_actions_fast(analysis: dict) -> dict:
    """
    Recommends UI actions based on the analysis results (fast mode).
    """
    dynamics = analysis.get("conversation_dynamics", {})
    response = analysis.get("response_analysis", {})
    suggestions = analysis.get("suggested_topics", {})

    # Focus topic
    focus_topic = suggestions.get("escalate_topic") or suggestions.get("next_topic")

    # Ask question back
    ask_question_back = not response.get("last_match_response", {}).get("contains_question", False)

    # Escalate flirtation
    flirt_level = dynamics.get("flirtation_level")
    escalate_flirtation = flirt_level in ["medium", "high", "explicit"]

    # Suggested next action based on stage
    stage = dynamics.get("stage")
    if stage == "starting":
        suggestedNextAction = "BUILD_RAPPORT"
    elif stage == "active" and flirt_level in ["high", "explicit"]:
        suggestedNextAction = "PLAN_DATE"
    else:
        suggestedNextAction = "MAINTAIN_ENGAGEMENT"

    # Sexual communication style
    if flirt_level == "explicit":
        sexual_style = "direct_and_explicit"
    elif flirt_level == "high":
        sexual_style = "suggestive_and_romantic"
    else:
        sexual_style = "casual_and_flirty"


    return {
        "focus_topic": focus_topic,
        "ask_question_back": ask_question_back,
        "escalate_flirtation": escalate_flirtation,
        "length": 60,  # Default
        "tone": 75,  # Default
        "linguisticStyle": "casual",  # Default
        "emojiStrategy": "auto",  # Default
        "suggestedNextAction": suggestedNextAction,
        "sexualCommunicationStyle": sexual_style,
        "dateArcPhase": "escalation" if escalate_flirtation else "rapport_building",
        "suggestedResponseStyle": "direct" if dynamics.get("pace") == "fast" else "thoughtful",
    }
