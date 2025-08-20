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
    full_text = " ".join([message.get("content", "") for message in conversation_history])
    topics = analysis.get("topics", {}).get("neutral", [])
    scraped_data = analysis.get("scraped_data", {})
    my_profile = scraped_data.get("myProfile", "")
    their_profile = scraped_data.get("theirProfile", "")

    prompt = (
        f"Analyze the following dating conversation and provide actionable advice.\n\n"
        f"My Profile: '{my_profile}'\n"
        f"Their Profile: '{their_profile}'\n"
        f"Conversation (last 1000 chars): '{full_text[-1000:]}'\n"
        f"Main Topics: {', '.join(topics)}\n\n"
        f"Generate a response in the following format exactly:\n"
        f"Suggested Questions:\n- Question 1\n- Question 2\n- Question 3\n\n"
        f"Goal Tracking:\n- Goal 1\n- Goal 2\n\n"
        f"Topic Switch Suggestions:\n- Suggestion 1\n- Suggestion 2\n"
    )

    try:
        generated_text = text_generator(prompt, max_new_tokens=250, num_return_sequences=1)[0]['generated_text']

        # Basic parsing of the generated text
        questions = re.findall(r"Suggested Questions:\n(.*?)\n\n", generated_text, re.DOTALL)
        goals = re.findall(r"Goal Tracking:\n(.*?)\n\n", generated_text, re.DOTALL)
        switches = re.findall(r"Topic Switch Suggestions:\n(.*?)$", generated_text, re.DOTALL)

        suggested_questions = [q.strip() for q in questions[0].split('\n') if q.strip()] if questions else []
        goal_tracking = [g.strip() for g in goals[0].split('\n') if g.strip()] if goals else []
        topic_switch_suggestions = [s.strip() for s in switches[0].split('\n') if s.strip()] if switches else []

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
