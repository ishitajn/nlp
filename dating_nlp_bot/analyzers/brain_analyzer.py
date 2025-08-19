
SUGGESTED_QUESTIONS = {
    "travel": ["What's the most adventurous trip you've ever taken?", "If you could travel anywhere tomorrow, where would you go?", "Do you prefer relaxing on a beach or exploring a new city?"],
    "food": ["What's your go-to comfort food?", "Are you more of a cook or a take-out person?", "What's the best meal you've had recently?"],
    "sports": ["What's your favorite way to stay active?", "Are you a fan of any sports teams?", "Do you enjoy working out in a gym or outdoors?"],
    "career": ["What do you enjoy most about your job?", "What are your long-term career goals?", "What's the most interesting project you've worked on?"],
    "hobbies": ["What's a hobby you've always wanted to try?", "How do you like to unwind after a long week?", "What's a movie or book that has stuck with you?"],
    "default": ["What's something that made you smile recently?", "What are you most looking forward to this week?", "What's a fun fact about you?"]
}

def analyze_brain_fast(analysis: dict) -> dict:
    """
    Generates the conversation brain output (fast mode).
    """
    topics = analysis.get("topics", {})
    dynamics = analysis.get("conversation_dynamics", {})
    suggestions = analysis.get("suggested_topics", {})

    # Memory Layer
    recent_topics = list(topics.get("map", {}).keys())
    memory_layer = {"recent_topics": recent_topics}

    # Predictive Actions
    # Suggested Questions
    suggested_questions = []
    next_topic = suggestions.get("next_topic")
    if next_topic and next_topic in SUGGESTED_QUESTIONS:
        suggested_questions.extend(SUGGESTED_QUESTIONS[next_topic][:2])
    else:
        suggested_questions.extend(SUGGESTED_QUESTIONS["default"][:2])

    # Goal Tracking
    goal_tracking = []
    stage = dynamics.get("stage")
    flirt_level = dynamics.get("flirtation_level")
    if stage == "starting":
        goal_tracking.append("Build rapport and find common interests.")
    elif stage == "active":
        goal_tracking.append("Maintain engagement and deepen the connection.")
        if flirt_level in ["medium", "high"]:
            goal_tracking.append("Gauge flirtation comfort level.")
        if flirt_level in ["high", "explicit"]:
            goal_tracking.append("Move to in-person meeting.")

    # Topic Switch Suggestions
    topic_switch_suggestions = [suggestions.get("next_topic")] if suggestions.get("next_topic") else []

    predictive_actions = {
        "suggested_questions": suggested_questions,
        "goal_tracking": goal_tracking,
        "topic_switch_suggestions": topic_switch_suggestions
    }

    return {
        "predictive_actions": predictive_actions,
        "memory_layer": memory_layer,
    }

from ..models.llm_generator import LLMGenerator

def analyze_brain_enhanced(conversation_history: list[dict], analysis: dict) -> dict:
    """
    Generates the conversation brain output (enhanced mode).
    """
    try:
        llm_generator = LLMGenerator()
        return llm_generator.generate(conversation_history, analysis)
    except Exception as e:
        # If the LLM fails for any reason, fallback to the fast (rule-based) method
        print(f"LLM generation failed: {e}. Falling back to fast brain analysis.")
        return analyze_brain_fast(analysis)
