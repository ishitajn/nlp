# In main.py
import asyncio
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware
from typing import Dict, Any

from preprocessor import clean_and_truncate
from analysis_engine import run_full_analysis
from model import AnalyzePayload, FinalResponse, ConversationAnalysisResponse, LastMessageAnalysisResponse, MemoryResponse, UISettings
from planner import compute_geo_time_features
from suggestion_engine import generate_suggestion_flags
from cache import generate_and_check_cache, set_cached_data

fl = open('load.json', 'a+')
fa = open('analysis.json', 'a+')

def build_final_json(
    payload: Dict[str, Any],
    analysis_data: Dict[str, Any],
    geo: Dict[str, Any],
    ui_settings: UISettings
) -> Dict[str, Any]:
    """
    Assembles the final JSON response from all the analysis components into the new nested structure.
    """
    behavior = analysis_data.get("behavioral_analysis", {})
    context = analysis_data.get("contextual_features", {})
    memory_features = context.get("memory_features", {})
    last_message_analysis_data = analysis_data.get("last_message_analysis", {})

    last_message_analysis = LastMessageAnalysisResponse(**last_message_analysis_data)
    memory = MemoryResponse(
        date_arc_phase=memory_features.get("date_arc_phase", "Unknown"),
        inside_jokes=memory_features.get("inside_jokes", []),
        avoided_topics=memory_features.get("avoided_topics", []),
        question_history=memory_features.get("question_history", [])
    )
    conversation_analysis = ConversationAnalysisResponse(
        conversation_state=behavior.get("conversation_state", "Unknown"),
        suppress_greeting=not behavior.get("suggest_greeting", True),
        last_message_analysis=last_message_analysis,
        memory=memory
    )

    pipeline_version = "modular_semantic_v13.0_enhanced" if ui_settings.use_enhanced_nlp else "modular_semantic_v13.0"

    # Prepare debug data if enabled
    debug_data = None
    if ui_settings.debug_mode_enabled:
        suggestion_flags = generate_suggestion_flags(ui_settings, analysis_data)
        debug_data = {
            "raw_analysis": analysis_data,
            "suggestion_flags": suggestion_flags,
            "geo_features": geo
        }

    final_response = FinalResponse(
        match_id=payload.get("matchId"),
        response="[Placeholder for generative response]",
        conversation_analysis=conversation_analysis,
        pipeline=pipeline_version,
        debug_data=debug_data
    )

    return final_response.model_dump(by_alias=True, exclude_none=True)

app = FastAPI(title="Dating Conversation Analyzer", version="13.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

async def run_analysis_pipeline(payload: AnalyzePayload) -> dict:
    """
    Runs the full analysis pipeline for a given conversation payload.
    This version removes the suggestion engine and focuses on analysis.
    """
    ui_settings = payload.ui_settings

    # Caching logic remains the same...
    cached_result, cache_key = await asyncio.to_thread(
        generate_and_check_cache, payload.match_id, ui_settings.use_enhanced_nlp, payload.scraped_data.conversation_history
    )
    if cached_result:
        analysis_results = cached_result[0]
    else:
        cleaned_turns = await asyncio.to_thread(clean_and_truncate, payload.scraped_data.conversation_history)
        if not cleaned_turns: raise HTTPException(status_code=400, detail="Conversation history is empty.")

        analysis_task = asyncio.to_thread(
            run_full_analysis,
            my_profile=ui_settings.my_profile,
            their_profile=payload.scraped_data.their_profile,
            processed_turns=cleaned_turns,
            use_enhanced_nlp=ui_settings.use_enhanced_nlp
        )
        geo_task = asyncio.to_thread(
            compute_geo_time_features, ui_settings.my_location, payload.scraped_data.their_location_string
        ) if ui_settings.geo_context_toggle else asyncio.sleep(0, result={})

        analysis_results, geo_features = await asyncio.gather(analysis_task, geo_task)
        await asyncio.to_thread(set_cached_data, cache_key, analysis_results, None)

    # Geo features might not have been cached, so we compute them if they are missing
    if 'geo_features' not in locals():
        geo_features = await asyncio.to_thread(
            compute_geo_time_features, ui_settings.my_location, payload.scraped_data.their_location_string
        ) if ui_settings.geo_context_toggle else {}

    return await asyncio.to_thread(
        build_final_json,
        payload=payload.model_dump(by_alias=True),
        analysis_data=analysis_results,
        geo=geo_features,
        ui_settings=ui_settings
    )

@app.post("/analyze")
async def analyze_conversation_endpoint(payload: AnalyzePayload):
    fl.write(payload.model_dump_json(indent=4) + '\n,\n')
    r_payload = await run_analysis_pipeline(payload)
    fa.write(payload.model_dump_json(indent=4) + '\n,\n')
    return r_payload

@app.get("/")
async def root():
    return {"message": "Dating Conversation Analyzer v12.0 is running."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)