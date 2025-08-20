import re
import json
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
    Generates the conversation brain output by prompting a small LLM for a structured JSON object.
    """
    llm = models.text_generator_enhanced
    if not llm:
        return analyze_brain_fast(analysis)

    # Prepare the context
    topics = analysis.get("topics", {}).get("neutral", [])
    scraped_data = analysis.get("scraped_data", {})
    my_profile = scraped_data.get("myProfile", "")
    their_profile = scraped_data.get("theirProfile", "")

    # Dynamically build conversation history to fit within context window
    MAX_CONTEXT_TOKENS = 512
    # Leave a buffer for model to generate response
    PROMPT_TOKEN_LIMIT = MAX_CONTEXT_TOKENS - 256

    # Base prompt without history
    base_prompt = (
        f"<|system|>\n"
        f"You are a helpful dating assistant. Your task is to analyze a dating conversation and provide actionable advice. "
        f"Based on the user's profile, the match's profile, and the recent conversation history, generate a JSON object with three keys: "
        f"'suggested_questions' (a list of 3 creative questions to ask next), "
        f"'goal_tracking' (a list of 2 potential conversational goals), and "
        f"'topic_switch_suggestions' (a list of 2 new topics to steer the conversation towards). "
        f"The user's role is 'user', and the match's role is 'assistant'. "
        f"Ensure your output is a single, valid JSON object and nothing else.\n"
        f"<|user|>\n"
        f"My Profile: {my_profile}\n"
        f"Their Profile: {their_profile}\n"
    )

    formatted_history = []
    current_token_count = len(llm.tokenize(base_prompt))

    for message in reversed(conversation_history):
        message_str = f"<{message['role']}>: {message['content']}\n"
        message_token_count = len(llm.tokenize(message_str))

        if current_token_count + message_token_count <= PROMPT_TOKEN_LIMIT:
            formatted_history.insert(0, message_str)
            current_token_count += message_token_count
        else:
            break

    history_str = "".join(formatted_history)
    prompt = f"{base_prompt}Conversation History:\n{history_str}<|assistant|>\n"

    predictive_actions = {
        "suggested_questions": [],
        "goal_tracking": [],
        "topic_switch_suggestions": []
    }

    try:
        # Generate text from the LLM
        llm_output = llm(prompt, max_new_tokens=256)

        # Extract JSON from the output string
        # Models sometimes wrap JSON in markdown backticks
        json_match = re.search(r'```json\n(.*)\n```', llm_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Or it might just be the JSON object itself
            json_str = llm_output[llm_output.find('{'):llm_output.rfind('}')+1]

        parsed_json = json.loads(json_str)

        # Populate predictive_actions from the parsed JSON
        predictive_actions["suggested_questions"] = parsed_json.get("suggested_questions", [])
        predictive_actions["goal_tracking"] = parsed_json.get("goal_tracking", [])
        predictive_actions["topic_switch_suggestions"] = parsed_json.get("topic_switch_suggestions", [])

    except (json.JSONDecodeError, AttributeError, IndexError, TypeError) as e:
        # If model output is not valid JSON or parsing fails, predictive_actions will remain empty
        print(f"Error parsing LLM output for brain analyzer: {e}")

    # Memory Layer (from fast analysis)
    fast_brain = analyze_brain_fast(analysis)
    memory_layer = fast_brain.get("memory_layer", {})

    return {"predictive_actions": predictive_actions, "memory_layer": memory_layer}
