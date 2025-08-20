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

    prompt = (
        f"This is an analysis of a dating conversation. "
        f"The conversation so far: '{full_text[-1000:]}' (last 1000 chars). "
        f"The main topics discussed are: {', '.join(topics)}. "
        f"Based on this, generate some brief, actionable advice for the user. "
        f"Provide a list of 3 suggested questions to ask next. "
        f"Provide a list of 2 potential goals for the conversation. "
        f"Provide a list of 2 topic switch suggestions. "
        f"Format the output clearly with headers like 'Suggested Questions:', 'Goal Tracking:', and 'Topic Switch Suggestions:'"
    )

    try:
        generated_text = text_generator(prompt, max_length=250, num_return_sequences=1)[0]['generated_text']

        # Basic parsing of the generated text
        questions = re.findall(r"Suggested Questions:\n(.*?)\n\n", generated_text, re.DOTALL)
        goals = re.findall(r"Goal Tracking:\n(.*?)\n\n", generated_text, re.DOTALL)
        switches = re.findall(r"Topic Switch Suggestions:\n(.*?)$", generated_text, re.DOTALL)

        suggested_questions = [q.strip() for q in questions[0].split('\n') if q.strip()] if questions else ["Ask about their day."]
        goal_tracking = [g.strip() for g in goals[0].split('\n') if g.strip()] if goals else ["Keep the conversation flowing."]
        topic_switch_suggestions = [s.strip() for s in switches[0].split('\n') if s.strip()] if switches else ["Talk about hobbies."]

    except Exception as e:
        # If model fails, fallback to a simpler set of suggestions
        suggested_questions = ["How is your week going?", "What do you do for fun?"]
        goal_tracking = ["Build rapport."]
        topic_switch_suggestions = ["Ask about their hobbies or weekend plans."]

    # Memory Layer (from fast analysis)
    fast_brain = analyze_brain_fast(analysis)
    memory_layer = fast_brain.get("memory_layer", {})

    predictive_actions = {
        "suggested_questions": suggested_questions,
        "goal_tracking": goal_tracking,
        "topic_switch_suggestions": topic_switch_suggestions
    }

    return {"predictive_actions": predictive_actions, "memory_layer": memory_layer}
