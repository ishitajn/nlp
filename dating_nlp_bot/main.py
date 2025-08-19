import json
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# Import all analyzers and recommenders
from dating_nlp_bot.analyzers import (
    sentiment_analyzer,
    topic_classifier,
    conversation_dynamics,
    geo_time_analyzer,
    response_analysis,
    brain_analyzer,
    prompt_generator,
)
from dating_nlp_bot.recommender import topic_suggester, action_recommender

# --- FastAPI App Definition ---
app = FastAPI(
    title="Dating Conversation Analyzer API",
    description="An API for NLP-driven analysis of dating conversations.",
    version="3.0.0",
)

# --- Pydantic Schemas ---
# Input
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

# Output
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
    liked: List[str]; disliked: List[str]; neutral: List[str]
    sensitive: List[str]; kinksAndFetishes: List[str]; pornReferences: List[str]
    map: TopicsMap
class SuggestedTopicsOutput(BaseModel):
    next_topic: Optional[str]; avoid_topic: Optional[str]; escalate_topic: Optional[str]
class ConversationDynamicsOutput(BaseModel):
    question_detected: bool; recent_greeting: bool; pace: str; stage: str
    is_engaged: bool; reciprocity_balance: str; flirtation_level: str
    sexualResponseSuggestion: bool
class LocationContext(BaseModel):
    address: Optional[str]; city: Optional[str]; state: Optional[str]; country: Optional[str]
    timeZone: Optional[str]; timeOfDay: Optional[str]; current_date_time: str
class GeoContextOutput(BaseModel):
    userLocation: LocationContext; matchLocation: LocationContext
    distance_miles: Optional[float]; timeZoneDifference: Optional[int]
    countryDifference: bool; isVirtual: bool
class LastMatchResponse(BaseModel):
    contains_question: bool; related_to_location: bool
class ResponseAnalysisOutput(BaseModel):
    last_response: str; last_match_response: LastMatchResponse
class RecommendedActionsOutput(BaseModel):
    focus_topic: Optional[str]; ask_question_back: bool; escalate_flirtation: bool
    length: int; tone: int; linguisticStyle: str; emojiStrategy: str
    suggestedNextAction: str; sexualCommunicationStyle: str; dateArcPhase: str
    suggestedResponseStyle: str
class PredictiveActions(BaseModel):
    suggested_questions: List[str]; goal_tracking: List[str]; topic_switch_suggestions: List[str]
class MemoryLayer(BaseModel):
    recent_topics: List[str]
class ConversationBrainOutput(BaseModel):
    predictive_actions: PredictiveActions; memory_layer: MemoryLayer
class LLMPromptContext(BaseModel):
    conversation_summary: str; user_persona_summary: str; match_persona_summary: str
class AnalysisOutput(BaseModel):
    sentiment: SentimentOutput; topics: TopicsOutput; suggested_topics: SuggestedTopicsOutput
    conversation_dynamics: ConversationDynamicsOutput; geoContext: GeoContextOutput
    response_analysis: ResponseAnalysisOutput; recommended_actions: RecommendedActionsOutput
    conversation_brain: ConversationBrainOutput; llm_prompt_context: LLMPromptContext

# --- Core Processing Logic ---
def run_fast_pipeline(payload: dict) -> dict:
    scraped_data = payload.get("scraped_data", {})
    ui_settings = payload.get("ui_settings", {})
    conversation_history = scraped_data.get("conversationHistory", [])
    my_location = ui_settings.get("myLocation", "")
    their_location = scraped_data.get("theirLocationString", "")
    scraped_data['myProfile'] = ui_settings.get("myProfile", "")
    sentiment = sentiment_analyzer.analyze_sentiment_fast(conversation_history)
    topics = topic_classifier.classify_topics_fast(conversation_history)
    dynamics = conversation_dynamics.analyze_dynamics_fast(conversation_history)
    geo_context = geo_time_analyzer.analyze_geo_time(my_location, their_location)
    response = response_analysis.analyze_response_fast(conversation_history)
    analysis_results = {"sentiment": sentiment, "topics": topics, "conversation_dynamics": dynamics, "response_analysis": response, "scraped_data": scraped_data}
    suggested_topics = topic_suggester.suggest_topics_fast(topics, dynamics)
    analysis_results["suggested_topics"] = suggested_topics
    recommended_actions = action_recommender.recommend_actions_fast(analysis_results)
    conversation_brain = brain_analyzer.analyze_brain_fast(analysis_results)
    llm_prompt_context = prompt_generator.generate_prompt_context(analysis_results)
    return {"sentiment": sentiment, "topics": topics, "suggested_topics": suggested_topics, "conversation_dynamics": dynamics, "geoContext": geo_context, "response_analysis": response, "recommended_actions": recommended_actions, "conversation_brain": conversation_brain, **llm_prompt_context}

def run_enhanced_pipeline(payload: dict) -> dict:
    scraped_data = payload.get("scraped_data", {})
    ui_settings = payload.get("ui_settings", {})
    conversation_history = scraped_data.get("conversationHistory", [])
    my_location = ui_settings.get("myLocation", "")
    their_location = scraped_data.get("theirLocationString", "")
    scraped_data['myProfile'] = ui_settings.get("myProfile", "")
    sentiment = sentiment_analyzer.analyze_sentiment_enhanced(conversation_history)
    topics = topic_classifier.classify_topics_enhanced(conversation_history)
    dynamics = conversation_dynamics.analyze_dynamics_enhanced(conversation_history)
    geo_context = geo_time_analyzer.analyze_geo_time(my_location, their_location)
    response = response_analysis.analyze_response_fast(conversation_history)
    analysis_results = {"sentiment": sentiment, "topics": topics, "conversation_dynamics": dynamics, "response_analysis": response, "scraped_data": scraped_data}
    suggested_topics = topic_suggester.suggest_topics_fast(topics, dynamics)
    analysis_results["suggested_topics"] = suggested_topics
    recommended_actions = action_recommender.recommend_actions_fast(analysis_results)
    conversation_brain = brain_analyzer.analyze_brain_enhanced(conversation_history, analysis_results)
    llm_prompt_context = prompt_generator.generate_prompt_context(analysis_results)
    return {"sentiment": sentiment, "topics": topics, "suggested_topics": suggested_topics, "conversation_dynamics": dynamics, "geoContext": geo_context, "response_analysis": response, "recommended_actions": recommended_actions, "conversation_brain": conversation_brain, **llm_prompt_context}

def process_payload(payload: dict) -> dict:
    use_enhanced = payload.get("ui_settings", {}).get("useEnhancedNlp", False)
    if use_enhanced:
        return run_enhanced_pipeline(payload)
    else:
        return run_fast_pipeline(payload)

# --- API Endpoint ---
@app.post("/analyze", response_model=AnalysisOutput)
def analyze_conversation(payload: AnalysisPayload):
    payload_dict = payload.model_dump()
    return process_payload(payload_dict)
