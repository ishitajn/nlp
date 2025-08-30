"""
This module is responsible for mapping the raw analysis data from the various
engines into the final, structured Pydantic response model. This decouples the
analysis pipeline from the final API contract.
"""
from typing import Dict, Any, List
from model import (
    FinalResponse, ConversationAnalysisResponse, LastMessageAnalysisResponse,
    MemoryResponse, UISettings, AdditionalAnalysis, OldConversationAnalysis,
    OldAnalysis, OldConversationState
)
from constants import DATE_ARC_PHASE_MAP, INTENT_MAP

def _format_question_history(history: List[Dict[str, str]]) -> str:
    """Formats the question history list into a readable string."""
    lines = []
    for item in history:
        role = "Me" if item.get("role") == "user" else "Them"
        lines.append(f'{role}: {item.get("question", "")}')
    return "\n".join(lines)

def _build_additional_analysis(analysis_data: Dict[str, Any]) -> AdditionalAnalysis:
    """Builds the 'additional_analysis' object from the raw analysis data."""
    behavior = analysis_data.get("behavioral_analysis", {})
    context = analysis_data.get("contextual_features", {})
    categorized_topics = analysis_data.get("categorized_topics", {})

    old_conv_analysis = OldConversationAnalysis(
        last_message_from_user=behavior.get("last_message_from_user"),
        last_message_from_match=behavior.get("last_message_from_match"),
        Last_message_from=behavior.get("Last_message_from", "unknown"),
        match_last_message_has_question=behavior.get("match_last_message_has_question", False),
        last_user_greeted=behavior.get("last_user_greeted", False),
        conversation_state=behavior.get("conversation_state", "Unknown"),
        greeting_detected=behavior.get("greeting_detected", False),
        flirtation_indicator=behavior.get("flirtation_indicator", False),
        recent_engagement_score=behavior.get("recent_engagement_score", "low"),
        suggest_topic_shift=behavior.get("suggest_topic_shift", False),
        suggest_greeting=behavior.get("suggest_greeting", False),
        pace=behavior.get("pace", "steady")
    )

    # Calculate flirtation_level for the old analysis block
    has_sexual_topics = bool(categorized_topics.get("sexual"))
    flirtation_level = "very high" if has_sexual_topics else "high" if behavior.get('flirtation_indicator') else "low"

    old_analysis = OldAnalysis(
        sentiment=context.get("sentiment_analysis", {}).get("overall", "neutral"),
        flirtation_level=flirtation_level,
        engagement=behavior.get("recent_engagement_score", "low"),
        pace=behavior.get("pace", "steady"),
        power_dynamics=context.get("power_dynamics", {})
    )

    old_conv_state = OldConversationState(
        topics={
            "focus": categorized_topics.get("focus", []), "avoid": categorized_topics.get("avoid", []),
            "neutral": categorized_topics.get("neutral", []), "sensitive": categorized_topics.get("sensitive", []),
            "romantic": categorized_topics.get("romantic", []), "fetish": categorized_topics.get("fetish", []),
            "sexual": categorized_topics.get("sexual", [])
        },
        recent_topics=analysis_data.get("recent_topics", [])
    )

    return AdditionalAnalysis(
        conversation_analysis=old_conv_analysis,
        analysis=old_analysis,
        conversation_state=old_conv_state
    )

def build_final_response(
    payload: Dict[str, Any],
    analysis_data: Dict[str, Any],
    suggestions: Dict[str, Any],
    geo: Dict[str, Any],
    ui_settings: UISettings
) -> Dict[str, Any]:
    """
    Assembles the final JSON response, mapping and formatting data for the frontend.
    """
    behavior = analysis_data.get("behavioral_analysis", {})
    context = analysis_data.get("contextual_features", {})
    memory_features = context.get("memory_features", {})
    last_message_analysis_data = analysis_data.get("last_message_analysis", {})

    # Map intents and date arc phase
    last_message_analysis_data['intents'] = [INTENT_MAP.get(i, i.upper()) for i in last_message_analysis_data.get('intents', [])]
    raw_phase = memory_features.get("date_arc_phase", "default")
    mapped_phase = DATE_ARC_PHASE_MAP.get(raw_phase, DATE_ARC_PHASE_MAP["default"])

    # Format list-based memory fields into strings
    formatted_memory_data = {
        "date_arc_phase": mapped_phase,
        "inside_jokes": "\n".join(memory_features.get("inside_jokes", [])),
        "avoided_topics": "\n".join(memory_features.get("avoided_topics", [])),
        "question_history": _format_question_history(memory_features.get("question_history", []))
    }

    # Instantiate Pydantic Models for the main response
    last_message_analysis = LastMessageAnalysisResponse(**last_message_analysis_data)
    memory = MemoryResponse(**formatted_memory_data)

    conversation_analysis = ConversationAnalysisResponse(
        conversation_state=behavior.get("conversation_state", "Unknown"),
        suppress_greeting=not behavior.get("suggest_greeting", True),
        last_message_analysis=last_message_analysis,
        memory=memory
    )

    # Build the additional analysis object
    additional_analysis = _build_additional_analysis(analysis_data)

    pipeline_version = "modular_semantic_v17.0_enhanced" if ui_settings.use_enhanced_nlp else "modular_semantic_v17.0"

    debug_data = None
    if ui_settings.debug_mode_enabled:
        debug_data = { "raw_analysis": analysis_data, "geo_features": geo }

    final_response = FinalResponse(
        match_id=payload.get("matchId"),
        response="[Placeholder for generative response]",
        conversation_analysis=conversation_analysis,
        suggestions=suggestions,
        geo=geo,
        additional_analysis=additional_analysis,
        pipeline=pipeline_version,
        debug_data=debug_data
    )

    return final_response.model_dump(by_alias=True, exclude_none=True)
