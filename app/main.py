import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# Add the parent directory to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Pydantic Models for Request Payload
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
    scraped_data: ScrapedData = Field(..., alias="scraped_data")
    ui_settings: UISettings = Field(..., alias="ui_settings")

# FastAPI App
app = FastAPI()

@app.post("/analyze")
async def analyze_conversation(payload: AnalyzePayload):
    """
    Analyzes a dating conversation to provide insights and suggestions.
    """
    if not payload.ui_settings.use_enhanced_nlp:
        # For now, we only implement the enhanced NLP path.
        # A fast, rule-based version could be implemented here.
        raise HTTPException(
            status_code=501,
            detail="The 'fast' analysis mode is not implemented. Please set 'useEnhancedNlp' to true."
        )

    # This is where the full pipeline will be called.
    # For now, just return a confirmation.
    return {"message": "Payload received successfully. Analysis pipeline will be implemented here."}

@app.get("/")
async def root():
    return {"message": "Dating Conversation Analyzer is running."}
