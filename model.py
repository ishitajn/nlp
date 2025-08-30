"""
Defines the Pydantic models for the API request and response structures.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal, Dict, Any

# --- Request Payload Models ---
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
    local_model_name: Optional[str] = Field(None)

class Feedback(BaseModel):
    current_topic: str
    chosen_suggestion: str
    action: Literal["chosen", "dismissed"]

class AnalyzePayload(BaseModel):
    match_id: str = Field(..., alias="matchId")
    scraped_data: ScrapedData
    ui_settings: UISettings
    feedback: Optional[List[Feedback]] = None

# --- Final, Normalized Response Models ---
class LastMessageAnalysis(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    is_direct_question: bool = Field(..., alias="isDirectQuestion")
    is_low_effort: bool = Field(..., alias="isLowEffort")
    is_sarcastic: bool = Field(..., alias="isSarcastic")
    is_ambiguous: bool = Field(..., alias="isAmbiguous")
    is_vulnerable: bool = Field(..., alias="isVulnerable")
    valence: float
    arousal: float
    intents: List[str]

class Memory(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    date_arc_phase: str = Field(..., alias="dateArcPhase")
    inside_jokes: str = Field(..., alias="insideJokes")
    avoided_topics: str = Field(..., alias="avoidedTopics")
    question_history: str = Field(..., alias="questionHistory")

class Engagement(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    last_message_from_user: Optional[str] = Field(..., alias="lastMessageFromUser")
    last_message_from_match: Optional[str] = Field(..., alias="lastMessageFromMatch")
    last_message_from: str = Field(..., alias="lastMessageFrom")
    match_last_message_has_question: bool = Field(..., alias="matchLastMessageHasQuestion")
    last_user_greeted: bool = Field(..., alias="lastUserGreeted")
    greeting_detected: bool = Field(..., alias="greetingDetected")
    flirtation_indicator: bool = Field(..., alias="flirtationIndicator")
    recent_engagement_score: str = Field(..., alias="recentEngagementScore")
    suggest_topic_shift: bool = Field(..., alias="suggestTopicShift")
    suggest_greeting: bool = Field(..., alias="suggestGreeting")
    pace: str

class Analysis(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    sentiment: str
    flirtation_level: str = Field(..., alias="flirtationLevel")
    engagement: str
    pace: str
    power_dynamics: Dict[str, Any] = Field(..., alias="powerDynamics")

class ConversationAnalysis(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    state: str
    suppress_greeting: bool = Field(..., alias="suppressGreeting")
    last_message_analysis: LastMessageAnalysis = Field(..., alias="lastMessageAnalysis")
    memory: Memory
    engagement: Engagement
    analysis: Analysis
    topics: Dict[str, List[str]]
    recent_topics: List[str] = Field(..., alias="recentTopics")

class FinalResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    match_id: str = Field(..., alias="matchId")
    response: str
    conversation_analysis: ConversationAnalysis = Field(..., alias="conversationAnalysis")
    suggestions: Optional[Dict[str, Any]] = None
    geo: Optional[Dict[str, Any]] = None
    pipeline: str
    debug_data: Optional[Dict[str, Any]] = Field(None, alias="debugData")