# In main.py
import asyncio
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware
from typing import Dict, Any

from preprocessor import clean_and_truncate
from analysis_engine import run_full_analysis
from model import AnalyzePayload
from planner import compute_geo_time_features
from suggestion_engine import generate_suggestions

def build_final_json(
    payload: Dict[str, Any],
    analysis_data: Dict[str, Any],
    suggestions: Dict[str, Any],
    geo: Dict[str, Any]
) -> Dict[str, Any]:
    context = analysis_data.get("contextual_features", {})
    categorized_topics = analysis_data.get("categorized_topics", {})
    behavior = analysis_data.get("behavioral_analysis", {})
    recent_topics = analysis_data.get("recent_topics", [])
    final_topics_object = {
        "focus": categorized_topics.get("focus", []), "avoid": categorized_topics.get("avoid", []),
        "neutral": categorized_topics.get("neutral", []), "sensitive": categorized_topics.get("sensitive", []),
        "romantic": categorized_topics.get("romantic", []), "fetish": categorized_topics.get("fetish", []),
        "sexual": categorized_topics.get("sexual", [])
    }
    conversation_state = {"topics": final_topics_object, "recent_topics": recent_topics}
    geo_output = {}
    if geo:
        distance_km, time_diff_hours = geo.get('distance_km'), geo.get('time_difference_hours')
        geo_output = {
            "userLocation": geo.get("my_location", {}), "matchLocation": geo.get("their_location", {}),
            "Time Difference": int(round(time_diff_hours)) if time_diff_hours is not None else None,
            "Distance_mile": int(distance_km * 0.621371) if distance_km is not None else None,
            "is_virtual": geo.get("is_virtual", False)
        }
    final_suggestions = suggestions
    final_suggestions["topic_shift_recommended"] = behavior.get("suggest_topic_shift", False)
    flirtation_indicator = behavior.get('flirtation_indicator', False)
    has_sexual_topics = bool(categorized_topics.get("sexual"))
    flirtation_level = "very high" if has_sexual_topics else "high" if flirtation_indicator else "low"
    final_analysis_object = {
        "sentiment": context.get("sentiment_analysis", {}).get("overall", "neutral"),
        "flirtation_level": flirtation_level,
        "engagement": context.get("engagement_metrics", {}).get("level", "low"),
        "pace": context.get("engagement_metrics", {}).get("pace", "steady"),
        "power_dynamics": context.get("power_dynamics", {})
    }
    return {
        "matchId": payload.get("matchId"), "conversation_state": conversation_state,
        "geo": geo_output, "suggestions": final_suggestions,
        "analysis": final_analysis_object, "conversation_analysis": behavior,
        "pipeline": "modular_semantic_v12.0"
    }

app = FastAPI(title="Dating Conversation Analyzer", version="12.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

async def run_analysis_pipeline(payload: AnalyzePayload) -> dict:
    payload_dict = payload.dict(by_alias=True)
    cleaned_turns = await asyncio.to_thread(clean_and_truncate, payload_dict["scraped_data"]["conversationHistory"])
    if not cleaned_turns: raise HTTPException(status_code=400, detail="Conversation history is empty.")
    
    analysis_task = asyncio.to_thread(run_full_analysis, payload.ui_settings.my_profile, payload.scraped_data.their_profile, cleaned_turns)
    geo_task = asyncio.to_thread(compute_geo_time_features, payload.ui_settings.my_location, payload.scraped_data.their_location_string)
    analysis_results, geo_features = await asyncio.gather(analysis_task, geo_task)
    
    final_suggestions = await asyncio.to_thread(
        generate_suggestions,
        analysis_data=analysis_results
    )
    
    return await asyncio.to_thread(
        build_final_json, payload=payload_dict, analysis_data=analysis_results,
        suggestions=final_suggestions, geo=geo_features
    )

@app.post("/analyze")
async def analyze_conversation_endpoint(payload: AnalyzePayload):
    return await run_analysis_pipeline(payload)

@app.get("/")
async def root():
    return {"message": "Dating Conversation Analyzer v12.0 is running."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)