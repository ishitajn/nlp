from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

from .main import process_payload

app = FastAPI(
    title="Dating Conversation Analyzer API",
    description="An API for NLP-driven analysis of dating conversations.",
    version="2.0.0",
)

# --- Input Schemas (no changes here) ---

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

# --- V2 Output Schemas ---

class SentimentOutput(BaseModel):
    overall: str

class FemaleCentricTopics(BaseModel):
    fashion: List[str] = []
    wellness: List[str] = []
    hobbies: List[str] = []
    social: List[str] = []
    relationships: List[str] = []

class TopicsMap(BaseModel):
    travel: List[str] = []
    food: List[str] = []
    flirt: List[str] = []
    sexual: List[str] = []
    sports: List[str] = []
    career: List[str] = []
    female_centric: FemaleCentricTopics = Field(default_factory=FemaleCentricTopics)

class TopicsOutput(BaseModel):
    liked: List[str]
    disliked: List[str]
    neutral: List[str]
    sensitive: List[str]
    kinksAndFetishes: List[str]
    pornReferences: List[str]
    map: TopicsMap

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

class LLMPromptContext(BaseModel):
    conversation_summary: str
    user_persona_summary: str
    match_persona_summary: str

class AnalysisOutput(BaseModel):
    sentiment: SentimentOutput
    topics: TopicsOutput
    suggested_topics: SuggestedTopicsOutput
    conversation_dynamics: ConversationDynamicsOutput
    geoContext: GeoContextOutput
    response_analysis: ResponseAnalysisOutput
    recommended_actions: RecommendedActionsOutput
    conversation_brain: ConversationBrainOutput
    llm_prompt_context: LLMPromptContext

@app.post("/analyze", response_model=AnalysisOutput)
def analyze_conversation(payload: AnalysisPayload):
    """
    Analyzes a dating conversation based on the provided payload.
    """
    payload_dict = payload.model_dump()
    result = process_payload(payload_dict)
    return result
