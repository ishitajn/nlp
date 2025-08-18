def generate_prompt_context(analysis: dict) -> dict:
    """
    Generates a dictionary of natural language summaries to be used
    as context for a downstream LLM prompt.
    """
    sentiment = analysis.get("sentiment", {})
    topics = analysis.get("topics", {})
    dynamics = analysis.get("conversation_dynamics", {})
    scraped_data = analysis.get("scraped_data", {})

    # Build Conversation Summary
    conv_summary_parts = []
    if sentiment.get("overall"):
        conv_summary_parts.append(f"The conversation is {sentiment['overall']}")
    if dynamics.get("pace"):
        conv_summary_parts.append(f"it is {dynamics['pace']}-paced")
    if dynamics.get("reciprocity_balance"):
        conv_summary_parts.append(f"and the reciprocity is {dynamics['reciprocity_balance']}.")

    discussed_topics = list(topics.get("map", {}).keys())
    if discussed_topics:
        conv_summary_parts.append(f"The main topics have been {', '.join(discussed_topics)}.")

    if dynamics.get("flirtation_level"):
        conv_summary_parts.append(f"The flirtation level is {dynamics['flirtation_level']}.")

    conversation_summary = " ".join(conv_summary_parts)

    # Build Persona Summaries
    user_persona_summary = f"The user's name is {scraped_data.get('myName', 'User')}. Their profile says: {scraped_data.get('myProfile', 'Not provided')}."
    match_persona_summary = f"The match's name is {scraped_data.get('theirName', 'Match')}. Their profile says: {scraped_data.get('theirProfile', 'Not provided')}."

    return {
        "llm_prompt_context": {
            "conversation_summary": conversation_summary,
            "user_persona_summary": user_persona_summary,
            "match_persona_summary": match_persona_summary
        }
    }
