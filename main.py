"""
Main FastAPI application file.
Handles API endpoints, request orchestration, and service initialization.
"""
import asyncio
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware

from services import Services
from preprocessor import clean_and_truncate
from analysis_engine import run_full_analysis
from model import AnalyzePayload
from suggestion_engine import generate_suggestions
from cache import generate_and_check_cache, set_cached_data
from mapper import build_final_response

# --- App Initialization ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI(title="Dating Conversation Analyzer", version="15.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Initialize the service container at startup.
# This loads all models and configs into memory once.
services = Services()

# --- Analysis Pipeline ---
async def run_analysis_pipeline(payload: AnalyzePayload, services: Services) -> dict:
    """
    Orchestrates the full analysis pipeline for a given conversation payload.
    """
    ui_settings = payload.ui_settings

    # The cache key is now generated inside this function call
    cached_data, cache_key = await asyncio.to_thread(
        generate_and_check_cache,
        match_id=payload.match_id,
        use_enhanced_nlp=ui_settings.use_enhanced_nlp,
        conversation_history=[turn.model_dump() for turn in payload.scraped_data.conversation_history]
    )

    if cached_data:
        analysis_results = cached_data.get("analysis")
        final_suggestions = cached_data.get("suggestions")
        # Geo features are not cached, so they need to be computed
        geo_features = await asyncio.to_thread(
            services.planner.compute_geo_time_features, ui_settings.my_location, payload.scraped_data.their_location_string
        ) if ui_settings.geo_context_toggle else {}
    else:
        cleaned_turns = await asyncio.to_thread(clean_and_truncate, [t.model_dump() for t in payload.scraped_data.conversation_history])
        if not cleaned_turns:
            raise HTTPException(status_code=400, detail="Conversation history is empty.")

        # Run analysis and geo-feature extraction in parallel
        analysis_task = asyncio.to_thread(
            run_full_analysis,
            services=services,
            my_profile=ui_settings.my_profile,
            their_profile=payload.scraped_data.their_profile,
            processed_turns=cleaned_turns,
            use_enhanced_nlp=ui_settings.use_enhanced_nlp
        )
        geo_task = asyncio.to_thread(
            services.planner.compute_geo_time_features, ui_settings.my_location, payload.scraped_data.their_location_string
        ) if ui_settings.geo_context_toggle else asyncio.sleep(0, result={})

        analysis_results, geo_features = await asyncio.gather(analysis_task, geo_task)

        # Generate suggestions based on the analysis
        final_suggestions = await asyncio.to_thread(
            generate_suggestions,
            services=services,
            categorized_topics=analysis_results.get("categorized_topics", {}),
            topic_map=analysis_results.get("topic_map", {}),
            behavioral_analysis=analysis_results.get("behavioral_analysis", {}),
            use_enhanced_nlp=ui_settings.use_enhanced_nlp,
            my_profile=ui_settings.my_profile,
            their_profile=payload.scraped_data.their_profile
        )

        # Cache the combined results
        data_to_cache = {"analysis": analysis_results, "suggestions": final_suggestions}
        await asyncio.to_thread(set_cached_data, cache_key, data_to_cache)

    # Map the final results to the response model
    return await asyncio.to_thread(
        build_final_response,
        payload=payload.model_dump(by_alias=True),
        analysis_data=analysis_results,
        suggestions=final_suggestions,
        geo=geo_features,
        ui_settings=ui_settings
    )

# --- API Endpoints ---
@app.post("/analyze")
async def analyze_conversation_endpoint(payload: AnalyzePayload):
    """The main endpoint to analyze a conversation."""
    return await run_analysis_pipeline(payload, services)

@app.get("/")
async def root():
    """Root endpoint for health checks."""
    return {"message": "Dating Conversation Analyzer v15.0 is running."}

# --- Main Execution ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)