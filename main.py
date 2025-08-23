import asyncio
import json

import uvicorn
from fastapi import FastAPI, HTTPException

from starlette.middleware.cors import CORSMiddleware

import assembler
import normalizer
from analysis_engine import run_full_analysis
from model import AnalyzePayload
from planner import compute_geo_time_features
from suggestion_engine import generate_suggestions

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
        normalizer.clean_and_truncate, payload_dict["scraped_data"]["conversationHistory"]
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
        conversation_turns=cleaned_turns
    )

    # --- Assemble final output ---
    final_json = await asyncio.to_thread(
        assembler.build_final_json,
        payload=payload_dict,
        analysis_data=analysis_results,
        suggestions=final_suggestions,
        geo=geo_features
    )
    with open('analysis.json', 'a+') as f:
        f.write(json.dumps(final_json, indent=4)+'\n,\n')
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