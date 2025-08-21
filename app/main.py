import sys
import os
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Pydantic Models ---
class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    date: str

class ScrapedData(BaseModel):
    my_name: str = Field(..., alias="myName")
    their_name: str = Field(..., alias="theirName")
    their_profile: str = Field(..., alias="theirProfile")
    their_location_string: str = Field(..., alias="theirLocationString")
    conversation_history: List[ConversationTurn] = Field(..., alias="conversationHistory")

class UISettings(BaseModel):
    use_enhanced_nlp: bool = Field(..., alias="useEnhancedNlp")
    my_location: str = Field(..., alias="myLocation")
    my_profile: str = Field(..., alias="myProfile")
    local_model_name: Optional[str] = Field(None, alias="local_model_name")

class AnalyzePayload(BaseModel):
    match_id: str = Field(..., alias="matchId")
    scraped_data: ScrapedData
    ui_settings: UISettings

# --- Import Services ---
from app.svc import normalizer, planner, probes, topics, reranker, assembler
from app.svc.embedder import embedder_service
from app.svc.index import faiss_index_service
from app.svc.generator import suggestion_generator_service

# --- FastAPI App ---
app = FastAPI(
    title="Dating Conversation Analyzer",
    description="An NLP-driven API to analyze dating conversations and provide insights.",
    version="1.0.0"
)

# --- Analysis Pipeline (Now Asynchronous) ---
async def run_analysis_pipeline(payload: AnalyzePayload) -> dict:
    """
    The main analysis pipeline, running blocking IO and CPU-bound tasks in a thread pool.
    """
    payload_dict = payload.dict(by_alias=True)
    payload_dict['timestamp'] = datetime.utcnow().isoformat()

    # 1. Normalize and Clean (Fast, but run in thread for consistency)
    cleaned_turns = await asyncio.to_thread(
        normalizer.clean_and_truncate, payload_dict["scraped_data"]["conversationHistory"]
    )
    if not cleaned_turns:
        raise HTTPException(status_code=400, detail="Conversation history cannot be empty.")
    turn_texts = [t["content"] for t in cleaned_turns]

    # 2. Generate Embeddings (Slow, CPU-bound)
    vectors = await asyncio.to_thread(embedder_service.encode_cached, turn_texts)

    # 3. Add to FAISS Index (Potential disk I/O)
    await asyncio.to_thread(faiss_index_service.ensure_added, cleaned_turns, vectors, payload.match_id)

    # 4. Evaluate Features & Probes (CPU-bound)
    features = await asyncio.to_thread(probes.evaluate_features, cleaned_turns)

    # 5. Discover Topics & Conversation State (CPU-bound)
    n_clusters = min(4, len(cleaned_turns))
    if n_clusters > 1:
        conversation_state = await asyncio.to_thread(
            topics.discover_and_label_topics, cleaned_turns, vectors, n_clusters=n_clusters
        )
    else:
        conversation_state = {
            "focus": [], "avoid": [], "neutral": [], "sensitive": [],
            "fetish": [], "sexual": [], "recent_topics": []
        }

    # 6. Compute Geo/Time Features (Network I/O)
    geo_features = await asyncio.to_thread(
        planner.compute_geo_time_features,
        payload.ui_settings.my_location,
        payload.scraped_data.their_location_string
    )

    # 7. Generate Suggestions with LLM (Slow, CPU-bound)
    raw_suggestions = await asyncio.to_thread(
        suggestion_generator_service.suggest, cleaned_turns, features, conversation_state, geo_features
    )
    final_suggestions = await asyncio.to_thread(reranker.enforce_constraints, raw_suggestions)

    # 8. Assemble Final JSON (Fast, but run in thread for consistency)
    final_json = await asyncio.to_thread(
        assembler.build_final_json,
        payload=payload_dict,
        topics=conversation_state,
        geo=geo_features,
        suggestions=final_suggestions,
        features=features
    )
    return final_json

# --- API Endpoints ---
@app.post("/analyze")
async def analyze_conversation_endpoint(payload: AnalyzePayload):
    """
    Analyzes a dating conversation to provide insights and suggestions.
    """
    if not payload.ui_settings.use_enhanced_nlp:
        raise HTTPException(
            status_code=501,
            detail="The 'fast' analysis mode is not implemented. Please set 'useEnhancedNlp' to true."
        )
    return await run_analysis_pipeline(payload)

@app.get("/")
async def root():
    return {"message": "Dating Conversation Analyzer is running."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
