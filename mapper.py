"""
This module is responsible for mapping the raw analysis data from the various
engines into the final, structured Pydantic response model. This decouples the
analysis pipeline from the final API contract.
"""
from typing import Dict, Any, List
from model import (
    FinalResponse, ConversationAnalysis, LastMessageAnalysis, Memory,
    Engagement, Analysis, UISettings
)
from constants import DATE_ARC_PHASE_MAP, INTENT_MAP

def _format_question_history(history: List[Dict[str, str]]) -> str:
    """Formats the question history list into a readable string."""
    lines = [f"{'Me' if item.get('role') == 'user' else 'Them'}: {item.get('question', '')}" for item in history]
    return "\n".join(lines)

def build_final_response(
    payload: Dict[str, Any],
    analysis_data: Dict[str, Any],
    suggestions: Dict[str, Any],
    geo: Dict[str, Any],
    ui_settings: UISettings
) -> Dict[str, Any]:
    """
    Assembles the final JSON response, mapping and formatting data into the
    final, normalized structure.
    """
    behavior = analysis_data.get("behavioral_analysis", {})
    context = analysis_data.get("contextual_features", {})
    memory_features = context.get("memory_features", {})
    last_message_analysis_data = analysis_data.get("last_message_analysis", {})
    categorized_topics = analysis_data.get("categorized_topics", {})

    # --- Build Nested Pydantic Objects ---

    # Pop 'intents' so it can be transformed and passed separately, avoiding a TypeError.
    intents_raw = last_message_analysis_data.pop('intents', [])
    mapped_intents = [INTENT_MAP.get(i, i.upper()) for i in intents_raw]

    last_message_analysis = LastMessageAnalysis(
        intents=mapped_intents,
        **last_message_analysis_data
    )

    memory = Memory(
        date_arc_phase=DATE_ARC_PHASE_MAP.get(memory_features.get("date_arc_phase", "default"), "VIBING"),
        inside_jokes="\n".join(memory_features.get("inside_jokes", [])),
        avoided_topics=", ".join(memory_features.get("avoided_topics", [])),
        question_history=_format_question_history(memory_features.get("question_history", []))
    )

    engagement = Engagement(
        last_message_from_user=behavior.get("last_message_from_user"),
        last_message_from_match=behavior.get("last_message_from_match"),
        last_message_from=behavior.get("Last_message_from", "unknown"),
        match_last_message_has_question=behavior.get("match_last_message_has_question", False),
        last_user_greeted=behavior.get("last_user_greeted", False),
        greeting_detected=behavior.get("greeting_detected", False),
        flirtation_indicator=behavior.get("flirtation_indicator", False),
        recent_engagement_score=behavior.get("recent_engagement_score", "low"),
        suggest_topic_shift=behavior.get("suggest_topic_shift", False),
        suggest_greeting=behavior.get("suggest_greeting", False),
        pace=behavior.get("pace", "steady")
    )

    flirtation_level = "very high" if bool(categorized_topics.get("sexual")) else "high" if behavior.get('flirtation_indicator') else "low"
    analysis = Analysis(
        sentiment=context.get("sentiment_analysis", {}).get("overall", "neutral"),
        flirtation_level=flirtation_level,
        engagement=behavior.get("recent_engagement_score", "low"),
        pace=behavior.get("pace", "steady"),
        power_dynamics=context.get("power_dynamics", {})
    )

    conversation_analysis = ConversationAnalysis(
        state=behavior.get("conversation_state", "Unknown"),
        suppress_greeting=not behavior.get("suggest_greeting", True),
        last_message_analysis=last_message_analysis,
        memory=memory,
        engagement=engagement,
        analysis=analysis,
        topics=categorized_topics,
        recent_topics=analysis_data.get("recent_topics", [])
    )

    pipeline_version = f"modular_semantic_v18.0_{'enhanced' if ui_settings.use_enhanced_nlp else 'standard'}"

    debug_data = { "raw_analysis": analysis_data, "geo_features": geo } if ui_settings.debug_mode_enabled else None

    final_response = FinalResponse(
        match_id=payload.get("matchId"),
        response="[Placeholder for generative response]",
        conversation_analysis=conversation_analysis,
        suggestions=suggestions,
        geo=geo,
        pipeline=pipeline_version,
        debug_data=debug_data
    )

    return final_response.model_dump(by_alias=True, exclude_none=True)
