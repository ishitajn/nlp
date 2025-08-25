"""
Defines the Pydantic models for the API request and response structures.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal


# --- Pydantic Models ---
class ConversationTurn(BaseModel):
    """Represents a single turn in the conversation history."""
    role: Literal["user", "assistant"]
    content: str
    date: str

class ScrapedData(BaseModel):
    """Represents the data scraped from the dating platform."""
    my_name: str = Field(..., alias="myName")
    their_name: str = Field(..., alias="theirName")
    their_profile: str = Field(..., alias="theirProfile")
    their_location_string: str = Field(..., alias="theirLocationString")
    conversation_history: List[ConversationTurn] = Field(..., alias="conversationHistory")

class UISettings(BaseModel):
    """Represents the settings configured by the user in the UI."""
    use_enhanced_nlp: bool = Field(..., alias="useEnhancedNlp")
    my_location: str = Field(..., alias="myLocation")
    my_profile: str = Field(..., alias="myProfile")
    local_model_name: Optional[str] = Field(None, alias="local_model_name")

class Feedback(BaseModel):
    """Represents user feedback on a given suggestion."""
    current_topic: str
    chosen_suggestion: str
    action: Literal["chosen", "dismissed"]

class AnalyzePayload(BaseModel):
    """The main payload for the /analyze endpoint."""
    match_id: str = Field(..., alias="matchId")
    scraped_data: ScrapedData
    ui_settings: UISettings
    feedback: Optional[List[Feedback]] = None