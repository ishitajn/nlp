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
    This version uses multiple focused prompts for more reliable results.
    """
    text_generator = models.text_generator_enhanced
    if not text_generator:
        return analyze_brain_fast(analysis)

    # Prepare the shared context for the prompts
    topics = analysis.get("topics", {}).get("neutral", [])
    scraped_data = analysis.get("scraped_data", {})
    my_profile = scraped_data.get("myProfile", "")
    their_profile = scraped_data.get("theirProfile", "")
    last_messages = conversation_history[-10:]
    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in last_messages])

    base_prompt_context = (
        f"You are an AI assistant analyzing a dating conversation.\n"
        f"My Profile: '{my_profile}'\n"
        f"Their Profile: '{their_profile}'\n"
        f"Recent Conversation:\n{formatted_history}\n\n"
    )

    def generate_suggestions(prompt, max_tokens=50):
        try:
            full_prompt = base_prompt_context + prompt
            # The 'text2text-generation' pipeline output is just the generated text
            suggestions_text = text_generator(full_prompt, max_new_tokens=max_tokens, num_return_sequences=1)[0]['generated_text']
            # Split into lines and remove any leading hyphens/bullets
            return [line.strip().lstrip('-* ') for line in suggestions_text.split('\n') if line.strip()]
        except Exception:
            return []

    # Generate each section with a focused prompt
    suggested_questions = generate_suggestions("Based on the conversation, list exactly 3 creative questions to ask next:", max_tokens=60)
    goal_tracking = generate_suggestions("Based on the context, list exactly 2 potential conversational goals:", max_tokens=40)
    topic_switch_suggestions = generate_suggestions("Based on the context, list exactly 2 new topics to steer the conversation towards:", max_tokens=40)

    # Memory Layer (from fast analysis)
    fast_brain = analyze_brain_fast(analysis)
    memory_layer = fast_brain.get("memory_layer", {})

    predictive_actions = {
        "suggested_questions": suggested_questions,
        "goal_tracking": goal_tracking,
        "topic_switch_suggestions": topic_switch_suggestions
    }

    return {"predictive_actions": predictive_actions, "memory_layer": memory_layer}
