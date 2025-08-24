import asyncio
import json

import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware

from typing import Dict, Any, List
from collections import defaultdict

from preprocessor import clean_and_truncate
from analysis_engine import run_full_analysis
from model import AnalyzePayload
from planner import compute_geo_time_features
from suggestion_engine import generate_suggestions

def build_final_json(
    payload: Dict[str, Any],
    analysis_data: Dict[str, Any],
    suggestions: Dict[str, str], # Now receives stringified JSON
    geo: Dict[str, Any]
) -> Dict[str, Any]:

    # Extract data from the different engines
    context = analysis_data.get("contextual_features", {})
    # This is the output of the scoring engine
    categorized_topics = analysis_data.get("categorized_topics", {})
    behavior = analysis_data.get("behavioral_analysis", {})

    # --- Build conversation_state to match new spec ---
    topic_recency_heatmap = context.get("topic_recency", {})
    recent_topics = sorted(topic_recency_heatmap.keys(), key=lambda k: topic_recency_heatmap[k])

    # Ensure all keys are present in the topics object, even if empty
    final_topics_object = {
        "focus": categorized_topics.get("focus", []),
        "avoid": categorized_topics.get("avoid", []),
        "neutral": categorized_topics.get("neutral", []),
        "sensitive": categorized_topics.get("sensitive", []),
        "romantic": categorized_topics.get("romantic", []),
        "fetish": categorized_topics.get("fetish", []),
        "sexual": categorized_topics.get("sexual", [])
    }

    conversation_state = {
        "topics": final_topics_object,
        "recent_topics": recent_topics
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
        "Distance": f"{geo.get('distance_km', 'N/A')} km",
        "is_virtual": geo.get("is_virtual", False)
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

app = FastAPI(
    title="Dating Conversation Analyzer",
    description="An ultra-fast, high-accuracy deterministic analysis engine for dating conversations.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def run_analysis_pipeline(payload: AnalyzePayload) -> dict:
    payload_dict = payload.dict(by_alias=True)
    
    cleaned_turns = await asyncio.to_thread(
        clean_and_truncate, payload_dict["scraped_data"]["conversationHistory"]
    )
    if not cleaned_turns:
        raise HTTPException(status_code=400, detail="Conversation history is empty.")

    # --- Run ALL deterministic analysis in parallel ---
    analysis_task = asyncio.to_thread(
        run_full_analysis,
        payload.ui_settings.my_profile,
        payload.scraped_data.their_profile,
        cleaned_turns  # This was the missing piece
    )
    geo_task = asyncio.to_thread(
        compute_geo_time_features,
        payload.ui_settings.my_location,
        payload.scraped_data.their_location_string
    )
    analysis_results, geo_features = await asyncio.gather(analysis_task, geo_task)

    # --- Generate final suggestions ---
    final_suggestions = await asyncio.to_thread(
        generate_suggestions,
        analysis_data=analysis_results,
        conversation_turns=cleaned_turns,
        identified_topics=analysis_results.get('topic_clusters'),
        feedback=payload.feedback
    )

    # --- Assemble final output ---
    final_json = await asyncio.to_thread(
        build_final_json,
        payload=payload_dict,
        analysis_data=analysis_results,
        suggestions=final_suggestions,
        geo=geo_features
    )

    with open('analysis.json', 'a+') as f:
        f.write(json.dumps(final_json, indent=4)+'\n,\n')
    #     f.write(str(final_json))
    return final_json

@app.post("/analyze")
async def analyze_conversation_endpoint(payload: AnalyzePayload):
    with open('load.json', 'a+') as f:
        f.write(payload.model_dump_json(indent=4)+'\n,\n')
    return await run_analysis_pipeline(payload)

@app.get("/")
async def root():
    return {"message": "Dating Conversation Analyzer v3.0 is running."}

if __name__ == "__main__":
    # Assumes this file is in the root directory, and the app folder is a sibling.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)