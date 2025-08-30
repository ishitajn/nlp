"""
This module is responsible for mapping the raw analysis data from the various
engines into the final, structured Pydantic response model. This decouples the
analysis pipeline from the final API contract.
"""
from typing import Dict, Any, List
from model import FinalResponse, ConversationAnalysisResponse, LastMessageAnalysisResponse, MemoryResponse, UISettings
from constants import DATE_ARC_PHASE_MAP, INTENT_MAP

def _format_question_history(history: List[Dict[str, str]]) -> str:
    """Formats the question history list into a readable string."""
    lines = []
    for item in history:
        role = "Me" if item.get("role") == "user" else "Them"
        lines.append(f'{role}: {item.get("question", "")}')
    return "\n".join(lines)

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

    # --- Map and Format Data ---

    # Map intents to frontend values
    last_message_analysis_data['intents'] = [
        INTENT_MAP.get(intent, intent.upper()) for intent in last_message_analysis_data.get('intents', [])
    ]

    # Map date arc phase
    raw_phase = memory_features.get("date_arc_phase", "default")
    mapped_phase = DATE_ARC_PHASE_MAP.get(raw_phase, DATE_ARC_PHASE_MAP["default"])

    # Format list-based memory fields into strings
    formatted_memory_data = {
        "date_arc_phase": mapped_phase,
        "inside_jokes": "\n".join(memory_features.get("inside_jokes", [])),
        "avoided_topics": "\n".join(memory_features.get("avoided_topics", [])),
        "question_history": _format_question_history(memory_features.get("question_history", []))
    }

    # --- Instantiate Pydantic Models ---
    last_message_analysis = LastMessageAnalysisResponse(**last_message_analysis_data)
    memory = MemoryResponse(**formatted_memory_data)

    conversation_analysis = ConversationAnalysisResponse(
        conversation_state=behavior.get("conversation_state", "Unknown"),
        suppress_greeting=not behavior.get("suggest_greeting", True),
        last_message_analysis=last_message_analysis,
        memory=memory
    )

    pipeline_version = "modular_semantic_v16.0_enhanced" if ui_settings.use_enhanced_nlp else "modular_semantic_v16.0"

    debug_data = None
    if ui_settings.debug_mode_enabled:
        debug_data = { "raw_analysis": analysis_data, "geo_features": geo }

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
