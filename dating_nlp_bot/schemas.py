from pydantic import BaseModel, Field
from typing import List, Optional

# --- Pydantic Schemas ---

# Input Schemas
class ConversationMessage(BaseModel):
    role: str
    content: str
    date: str

class ScrapedData(BaseModel):
    myName: str
    theirName: str
    theirProfile: str
    theirLocationString: str
    conversationHistory: List[ConversationMessage]

class UISettings(BaseModel):
    useEnhancedNlp: bool
    myLocation: str
    myProfile: str
    local_model_name: Optional[str] = None

class AnalysisPayload(BaseModel):
    matchId: str
    scraped_data: ScrapedData
    ui_settings: UISettings

# Output Schemas
class SentimentOutput(BaseModel):
    overall: str

from typing import Dict

class TopicsOutput(BaseModel):
    liked: List[str]
    disliked: List[str]
    neutral: List[str]
    sensitive: List[str]
    kinksAndFetishes: List[str]
    pornReferences: List[str]
    map: Dict[str, List[str]]

class SuggestedTopicsOutput(BaseModel):
    next_topic: Optional[str]
    avoid_topic: Optional[str]
    escalate_topic: Optional[str]

class ConversationDynamicsOutput(BaseModel):
    question_detected: bool
    recent_greeting: bool
    pace: str
    stage: str
    is_engaged: bool
    reciprocity_balance: str
    flirtation_level: str
    sexualResponseSuggestion: bool

class LocationContext(BaseModel):
    address: Optional[str]
    city: Optional[str]
    state: Optional[str]
    country: Optional[str]
    timeZone: Optional[str]
    timeOfDay: Optional[str]
    current_date_time: str

class GeoContextOutput(BaseModel):
    userLocation: LocationContext
    matchLocation: LocationContext
    distance_miles: Optional[float]
    timeZoneDifference: Optional[int]
    countryDifference: bool
    isVirtual: bool

class LastMatchResponse(BaseModel):
    contains_question: bool
    related_to_location: bool

class ResponseAnalysisOutput(BaseModel):
    last_response: str
    last_match_response: LastMatchResponse

class RecommendedActionsOutput(BaseModel):
    focus_topic: Optional[str]
    ask_question_back: bool
    escalate_flirtation: bool
    length: int
    tone: int
    linguisticStyle: str
    emojiStrategy: str
    suggestedNextAction: str
    sexualCommunicationStyle: str
    dateArcPhase: str
    suggestedResponseStyle: str

class PredictiveActions(BaseModel):
    suggested_questions: List[str]
    goal_tracking: List[str]
    topic_switch_suggestions: List[str]

class MemoryLayer(BaseModel):
    recent_topics: List[str]

class ConversationBrainOutput(BaseModel):
    predictive_actions: PredictiveActions
    memory_layer: MemoryLayer

class ModelInfo(BaseModel):
    method: str
    model: str

class AnalysisInfo(BaseModel):
    pipeline: str
    sentiment_analysis: ModelInfo
    topic_classification: ModelInfo
    conversation_dynamics: ModelInfo
    response_analysis: ModelInfo
    brain_analysis: ModelInfo

class AnalysisOutput(BaseModel):
    sentiment: SentimentOutput
    topics: TopicsOutput
    suggested_topics: SuggestedTopicsOutput
    conversation_dynamics: ConversationDynamicsOutput
    geoContext: GeoContextOutput
    response_analysis: ResponseAnalysisOutput
    recommended_actions: RecommendedActionsOutput
    conversation_brain: ConversationBrainOutput
    analysis_info: AnalysisInfo
