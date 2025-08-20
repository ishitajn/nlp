import shelve
import hashlib
import json
from dating_nlp_bot.analyzers import (
    sentiment_analyzer,
    topic_classifier,
    conversation_dynamics,
    geo_time_analyzer,
    response_analysis,
    brain_analyzer,
)
from dating_nlp_bot.recommender import topic_suggester, action_recommender

CACHE_DB = "analysis_cache.db"

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
    return {"sentiment": sentiment, "topics": topics, "suggested_topics": suggested_topics, "conversation_dynamics": dynamics, "geoContext": geo_context, "response_analysis": response, "recommended_actions": recommended_actions, "conversation_brain": conversation_brain}

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
    return {"sentiment": sentiment, "topics": topics, "suggested_topics": suggested_topics, "conversation_dynamics": dynamics, "geoContext": geo_context, "response_analysis": response, "recommended_actions": recommended_actions, "conversation_brain": conversation_brain}

def process_payload(payload: dict) -> dict:
    match_id = payload.get("matchId")
    conversation_history = payload.get("scraped_data", {}).get("conversationHistory", [])

    if not match_id or not conversation_history:
        # Cannot cache without matchId or history, so run pipeline directly
        use_enhanced = payload.get("ui_settings", {}).get("useEnhancedNlp", False)
        if use_enhanced:
            return run_enhanced_pipeline(payload)
        else:
            return run_fast_pipeline(payload)

    # Create a hash of the conversation history to detect changes
    history_str = json.dumps(conversation_history, sort_keys=True)
    history_hash = hashlib.sha256(history_str.encode()).hexdigest()
    cache_key = f"{match_id}_{history_hash}"

    with shelve.open(CACHE_DB) as cache:
        if cache_key in cache:
            return cache[cache_key]

        use_enhanced = payload.get("ui_settings", {}).get("useEnhancedNlp", False)
        if use_enhanced:
            result = run_enhanced_pipeline(payload)
        else:
            result = run_fast_pipeline(payload)

        cache[cache_key] = result
        return result
