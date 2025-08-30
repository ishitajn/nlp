"""
Defines the Pydantic models for the API request and response structures.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal, Dict, Any


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
    """Represents all settings configured by the user in the UI."""
    custom_instruction: Optional[str] = Field(None, alias="customInstruction")
    end_with_question: bool = Field(True, alias="endWithQuestion")
    geo_context_toggle: bool = Field(True, alias="geoContextToggle")
    new_topic: bool = Field(False, alias="newTopic")
    strict_goal_override: bool = Field(False, alias="strictGoalOverride")
    debug_mode_enabled: bool = Field(False, alias="debugModeEnabled")
    linguistic_style: str = Field("default", alias="linguisticStyle")
    flirty_value: int = Field(5, alias="flirtyValue")
    length_value: int = Field(5, alias="lengthValue")
    emoji_strategy: str = Field("contextual", alias="emojiStrategy")
    model_temperature: float = Field(0.7, alias="modelTemperature")
    top_p_value: float = Field(1.0, alias="topPValue")
    use_enhanced_nlp: bool = Field(True, alias="useEnhancedNlp")
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


# --- Pydantic Models for API Response (Matching Frontend Structure) ---

# --- Models for the new 'additional_analysis' section ---
class OldConversationAnalysis(BaseModel):
    last_message_from_user: Optional[str] = None
    last_message_from_match: Optional[str] = None
    Last_message_from: str
    match_last_message_has_question: bool
    last_user_greeted: bool
    conversation_state: str
    greeting_detected: bool
    flirtation_indicator: bool
    recent_engagement_score: str
    suggest_topic_shift: bool
    suggest_greeting: bool
    pace: str

class OldAnalysis(BaseModel):
    sentiment: str
    flirtation_level: str
    engagement: str
    pace: str
    power_dynamics: Dict[str, Any]

class OldConversationState(BaseModel):
    topics: Dict[str, List[str]]
    recent_topics: List[str]

class AdditionalAnalysis(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    conversation_analysis: OldConversationAnalysis = Field(..., alias="conversationAnalysis")
    analysis: OldAnalysis
    conversation_state: OldConversationState = Field(..., alias="conversationState")


class LastMessageAnalysisResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    is_direct_question: bool = Field(..., alias="isDirectQuestion")
    is_low_effort: bool = Field(..., alias="isLowEffort")
    is_sarcastic: bool = Field(..., alias="isSarcastic")
    is_ambiguous: bool = Field(..., alias="isAmbiguous")
    is_vulnerable: bool = Field(..., alias="isVulnerable")
    valence: float
    arousal: float
    intents: List[str]

class MemoryResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    date_arc_phase: str = Field(..., alias="dateArcPhase")
    inside_jokes: str = Field(..., alias="insideJokes")
    avoided_topics: str = Field(..., alias="avoidedTopics")
    question_history: str = Field(..., alias="questionHistory")

class ConversationAnalysisResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    conversation_state: str = Field(..., alias="conversationState")
    suppress_greeting: bool = Field(..., alias="suppressGreeting")
    last_message_analysis: LastMessageAnalysisResponse = Field(..., alias="lastMessageAnalysis")
    memory: MemoryResponse

class FinalResponse(BaseModel):
    """Defines the final, nested response structure for the API."""
    model_config = ConfigDict(populate_by_name=True)

    match_id: str = Field(..., alias="matchId")
    response: str
    conversation_analysis: ConversationAnalysisResponse = Field(..., alias="conversationAnalysis")
    suggestions: Optional[Dict[str, Any]] = None
    geo: Optional[Dict[str, Any]] = None
    additional_analysis: Optional[AdditionalAnalysis] = Field(None, alias="additionalAnalysis")
    pipeline: str
    debug_data: Optional[Dict[str, Any]] = Field(None, alias="debugData")