import re
from dating_nlp_bot.model_loader import get_models
from dating_nlp_bot.utils.suggestions import SUGGESTED_QUESTIONS

models = get_models()

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
    liked_topics = topics.get("liked", [])

    for topic in liked_topics:
        if topic in SUGGESTED_QUESTIONS:
            suggested_questions.extend(SUGGESTED_QUESTIONS[topic])

    # Fallback to generic suggestions if no liked topics matched
    if not suggested_questions:
        next_topic = suggestions.get("next_topic")
        if next_topic and next_topic in SUGGESTED_QUESTIONS:
            suggested_questions.extend(SUGGESTED_QUESTIONS[next_topic])
        else:
            suggested_questions.extend(SUGGESTED_QUESTIONS["default"])

    # Goal Tracking
    goal_tracking = []
    stage = dynamics.get("stage")
    flirt_level = dynamics.get("flirtation_level")
    if stage == "starting":
        goal_tracking.append("Build rapport and find common interests.")
    elif stage == "active":
        goal_tracking.append("Maintain engagement and deepen the connection.")
        if flirt_level in ["medium", "high", "explicit"]:
            goal_tracking.append("Gauge flirtation comfort level.")
        if flirt_level in ["high", "explicit"]:
            goal_tracking.append("Move to in-person meeting.")

    topic_switch_suggestions = [suggestions.get("next_topic")] if suggestions.get("next_topic") else []

    predictive_actions = {
        "suggested_questions": suggested_questions,
        "goal_tracking": goal_tracking,
        "topic_switch_suggestions": topic_switch_suggestions
    }

    return { "predictive_actions": predictive_actions, "memory_layer": memory_layer }

def analyze_brain_enhanced(conversation_history: list[dict], analysis: dict) -> dict:
    """
    Generates the conversation brain output using a text generation model (enhanced mode).
    """
    text_generator = models.text_generator_enhanced
    if not text_generator:
        return analyze_brain_fast(analysis)

    # Prepare the context for the model
    topics = analysis.get("topics", {}).get("neutral", [])
    scraped_data = analysis.get("scraped_data", {})
    my_profile = scraped_data.get("myProfile", "")
    their_profile = scraped_data.get("theirProfile", "")

    # Format last 10 messages for context
    last_messages = conversation_history[-10:]
    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in last_messages])

    prompt = (
        f"Analyze the following dating conversation and provide actionable advice.\n\n"
        f"My Profile: '{my_profile}'\n"
        f"Their Profile: '{their_profile}'\n"
        f"Recent Conversation:\n{formatted_history}\n\n"
        f"Main Topics: {', '.join(topics)}\n\n"
        f"Based on the context, generate a response with the following headers. Under each header, provide a bulleted list of items as requested:\n\n"
        f"Suggested Questions: (provide a list of 3 creative questions to ask next)\n"
        f"Goal Tracking: (provide a list of 2 potential conversational goals)\n"
        f"Topic Switch Suggestions: (provide a list of 2 new topics to steer the conversation towards)\n"
    )

    try:
        generated_text = text_generator(prompt, max_new_tokens=250, num_return_sequences=1)[0]['generated_text']

        # More robust parsing logic
        parsed_output = {}
        # Split the text by the known headers
        sections = re.split(r'\n(?=Suggested Questions:|Goal Tracking:|Topic Switch Suggestions:)', generated_text)

        for section in sections:
            section = section.strip()
            if section.startswith("Suggested Questions:"):
                lines = section.replace("Suggested Questions:", "").strip().split('\n')
                parsed_output['questions'] = [line.strip().lstrip('- ') for line in lines if line.strip()]
            elif section.startswith("Goal Tracking:"):
                lines = section.replace("Goal Tracking:", "").strip().split('\n')
                parsed_output['goals'] = [line.strip().lstrip('- ') for line in lines if line.strip()]
            elif section.startswith("Topic Switch Suggestions:"):
                lines = section.replace("Topic Switch Suggestions:", "").strip().split('\n')
                parsed_output['switches'] = [line.strip().lstrip('- ') for line in lines if line.strip()]

        suggested_questions = parsed_output.get('questions', [])
        goal_tracking = parsed_output.get('goals', [])
        topic_switch_suggestions = parsed_output.get('switches', [])

    except Exception as e:
        # If model or parsing fails, return empty lists
        suggested_questions = []
        goal_tracking = []
        topic_switch_suggestions = []

    # Memory Layer (from fast analysis)
    fast_brain = analyze_brain_fast(analysis)
    memory_layer = fast_brain.get("memory_layer", {})

    predictive_actions = {
        "suggested_questions": suggested_questions,
        "goal_tracking": goal_tracking,
        "topic_switch_suggestions": topic_switch_suggestions
    }

    return {"predictive_actions": predictive_actions, "memory_layer": memory_layer}
