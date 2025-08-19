from .analyzers import (
    sentiment_analyzer,
    topic_classifier,
    conversation_dynamics,
    geo_time_analyzer,
    response_analysis,
    brain_analyzer,
    prompt_generator,
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

    scraped_data['myProfile'] = ui_settings.get("myProfile", "")

    # Run analyzers
    sentiment = sentiment_analyzer.analyze_sentiment_fast(conversation_history)
    topics = topic_classifier.classify_topics_fast(conversation_history)
    dynamics = conversation_dynamics.analyze_dynamics_fast(conversation_history)
    geo_context = geo_time_analyzer.analyze_geo_time(my_location, their_location)
    response = response_analysis.analyze_response_fast(conversation_history)

    analysis_results = {
        "sentiment": sentiment, "topics": topics, "conversation_dynamics": dynamics,
        "response_analysis": response, "scraped_data": scraped_data
    }

    # Run recommenders
    suggested_topics = topic_suggester.suggest_topics_fast(topics, dynamics)
    analysis_results["suggested_topics"] = suggested_topics
    recommended_actions = action_recommender.recommend_actions_fast(analysis_results)

    # Run brain and prompt generators
    conversation_brain = brain_analyzer.analyze_brain_fast(analysis_results)
    llm_prompt_context = prompt_generator.generate_prompt_context(analysis_results)

    return {
        "sentiment": sentiment, "topics": topics, "suggested_topics": suggested_topics,
        "conversation_dynamics": dynamics, "geoContext": geo_context,
        "response_analysis": response, "recommended_actions": recommended_actions,
        "conversation_brain": conversation_brain, **llm_prompt_context
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

    scraped_data['myProfile'] = ui_settings.get("myProfile", "")

    # Run enhanced analyzers
    sentiment = sentiment_analyzer.analyze_sentiment_enhanced(conversation_history)
    topics = topic_classifier.classify_topics_enhanced(conversation_history)
    dynamics = conversation_dynamics.analyze_dynamics_enhanced(conversation_history)
    geo_context = geo_time_analyzer.analyze_geo_time(my_location, their_location)
    # Explicitly use the fast version as there is no enhanced alternative
    response = response_analysis.analyze_response_fast(conversation_history)

    analysis_results = {
        "sentiment": sentiment, "topics": topics, "conversation_dynamics": dynamics,
        "response_analysis": response, "scraped_data": scraped_data
    }

    # Run recommenders
    suggested_topics = topic_suggester.suggest_topics_fast(topics, dynamics)
    analysis_results["suggested_topics"] = suggested_topics
    # Explicitly use the fast version as there is no enhanced alternative
    recommended_actions = action_recommender.recommend_actions_fast(analysis_results)

    # Run brain and prompt generators
    conversation_brain = brain_analyzer.analyze_brain_enhanced(conversation_history, analysis_results)
    llm_prompt_context = prompt_generator.generate_prompt_context(analysis_results)

    return {
        "sentiment": sentiment, "topics": topics, "suggested_topics": suggested_topics,
        "conversation_dynamics": dynamics, "geoContext": geo_context,
        "response_analysis": response, "recommended_actions": recommended_actions,
        "conversation_brain": conversation_brain, **llm_prompt_context
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
