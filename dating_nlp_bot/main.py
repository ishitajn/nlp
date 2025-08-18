import json
from .analyzers import (
    sentiment_analyzer,
    topic_classifier,
    conversation_dynamics,
    geo_time_analyzer,
    response_analysis,
    brain_analyzer,
)
from .recommender import topic_suggester, action_recommender

def run_fast_pipeline(payload: dict) -> dict:
    """
    Runs the fast NLP pipeline.
    """
    scraped_data = payload.get("scraped_data", {})
    ui_settings = payload.get("ui_settings", {})
    conversation_history = scraped_data.get("conversationHistory", [])
    my_location = ui_settings.get("myLocation", "")
    their_location = scraped_data.get("theirLocationString", "")

    # Run analyzers
    sentiment = sentiment_analyzer.analyze_sentiment_fast(conversation_history)
    topics = topic_classifier.classify_topics_fast(conversation_history)
    dynamics = conversation_dynamics.analyze_dynamics_fast(conversation_history)
    geo_context = geo_time_analyzer.analyze_geo_time(my_location, their_location)
    response = response_analysis.analyze_response_fast(conversation_history)

    # Run recommenders
    suggested_topics = topic_suggester.suggest_topics_fast(topics, dynamics)

    analysis_for_recommender = {"conversation_dynamics": dynamics, "response_analysis": response, "suggested_topics": suggested_topics}
    recommended_actions = action_recommender.recommend_actions_fast(analysis_for_recommender)

    # Run brain analyzer
    analysis_for_brain = {"topics": topics, "conversation_dynamics": dynamics, "suggested_topics": suggested_topics}
    conversation_brain = brain_analyzer.analyze_brain_fast(analysis_for_brain)

    return {
        "sentiment": sentiment, "topics": topics, "suggested_topics": suggested_topics,
        "conversation_dynamics": dynamics, "geoContext": geo_context,
        "response_analysis": response, "recommended_actions": recommended_actions,
        "conversation_brain": conversation_brain,
    }

def run_enhanced_pipeline(payload: dict) -> dict:
    """
    Runs the enhanced NLP pipeline.
    """
    scraped_data = payload.get("scraped_data", {})
    ui_settings = payload.get("ui_settings", {})
    conversation_history = scraped_data.get("conversationHistory", [])
    my_location = ui_settings.get("myLocation", "")
    their_location = scraped_data.get("theirLocationString", "")

    # Run enhanced analyzers
    sentiment = sentiment_analyzer.analyze_sentiment_enhanced(conversation_history)
    topics = topic_classifier.classify_topics_enhanced(conversation_history)
    dynamics = conversation_dynamics.analyze_dynamics_enhanced(conversation_history)
    geo_context = geo_time_analyzer.analyze_geo_time(my_location, their_location)
    response = response_analysis.analyze_response_enhanced(conversation_history)

    # Run recommenders
    suggested_topics = topic_suggester.suggest_topics_fast(topics, dynamics)

    analysis_for_recommender = {"conversation_dynamics": dynamics, "response_analysis": response, "suggested_topics": suggested_topics}
    recommended_actions = action_recommender.recommend_actions_enhanced(analysis_for_recommender)

    # Run brain analyzer
    analysis_for_brain = {"topics": topics, "conversation_dynamics": dynamics, "suggested_topics": suggested_topics}
    conversation_brain = brain_analyzer.analyze_brain_enhanced(conversation_history, analysis_for_brain)

    return {
        "sentiment": sentiment, "topics": topics, "suggested_topics": suggested_topics,
        "conversation_dynamics": dynamics, "geoContext": geo_context,
        "response_analysis": response, "recommended_actions": recommended_actions,
        "conversation_brain": conversation_brain,
    }

def process_payload(payload: dict) -> dict:
    """
    Processes the input payload and routes to the correct pipeline.
    """
    use_enhanced = payload.get("ui_settings", {}).get("useEnhancedNlp", False)
    if use_enhanced:
        return run_enhanced_pipeline(payload)
    else:
        return run_fast_pipeline(payload)

if __name__ == '__main__':
    sample_payload = {
        "matchId": "12345",
        "scraped_data": {
            "myName": "Jules",
            "theirName": "Alex",
            "theirProfile": "Loves hiking and dogs. Looking for a connection.",
            "theirLocationString": "San Francisco, USA",
            "conversationHistory": [
                {"role": "assistant", "content": "Hey, how's it going?", "date": "2024-01-01T10:00:00"},
                {"role": "user", "content": "Hi! I'm doing great, thanks for asking. I saw you like hiking, me too!", "date": "2024-01-01T10:05:00"},
                {"role": "assistant", "content": "Oh nice! We should go sometime. What's your favorite trail?", "date": "2024-01-01T10:10:00"}
            ]
        },
        "ui_settings": {
            "useEnhancedNlp": False,
            "myLocation": "New York, USA",
            "myProfile": "Software engineer, loves coffee and coding.",
            "local_model_name": None
        }
    }
    result = process_payload(sample_payload)
    print(json.dumps(result, indent=2))
