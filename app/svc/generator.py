import os
import re
from llama_cpp import Llama
from typing import Dict, Any, List

def _create_system_prompt() -> str:
    """Creates a simpler system prompt for the LLM."""
    return """You are a sophisticated dating assistant AI. Your task is to provide insightful and actionable suggestions based on conversation data.
Please provide suggestions for the following categories: Topics, Questions, Intimacy.
If the conversation's flirtation level is not 'low', also provide suggestions for the Sexual category.
Use headings for each category, for example: 'Topic:', 'Question:', 'Sexual:', 'Intimacy:'.
Provide up to two suggestions for each category, each on a new line starting with a hyphen.
"""

def _create_user_prompt(turns: List[Dict[str, Any]], features: Dict[str, Any], topics: Dict[str, Any], geo: Dict[str, Any]) -> str:
    """Packs all analysis data into a single string for the user message."""
    conversation_history = "\n".join([f"{t.get('role', 'unknown')}: {t.get('content', '')}" for t in turns])
    analysis_context = features.get("analysis", {})
    engagement_level = analysis_context.get("match_engaged", "N/A")
    flirtation_level = analysis_context.get("flirtation_level", "N/A")
    recent_topics_str = ", ".join(topics.get('recent_topics', [])) or "N/A"

    return f"""Based on the following data, please provide helpful suggestions.

**Conversation Analysis:**
- Engagement Level: {engagement_level}
- Flirtation Level: {flirtation_level}
- Recent Topics: {recent_topics_str}
- Location Context: The user and match are {geo.get('distance_km', 'N/A')} km apart.

**Conversation History:**
{conversation_history}
"""

def _parse_llm_output_flexibly(raw_text: str) -> Dict[str, List[str]]:
    """
    Parses the raw text output from the LLM using a flexible, state-based approach.
    It looks for headings and bullet points.
    """
    suggestions = {"topics": [], "questions": [], "sexual": [], "intimacy": []}

    # Normalize headings to map to our internal keys
    category_map = {
        "topic": "topics",
        "suggestion": "topics", # The model seems to use this interchangeably
        "question": "questions",
        "sexual": "sexual",
        "intimacy": "intimacy"
    }

    current_category = None

    for line in raw_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check if the line is a heading
        match = re.match(r'^([a-zA-Z]+):', line)
        if match:
            heading = match.group(1).lower()
            if heading in category_map:
                current_category = category_map[heading]
                continue # Move to the next line after identifying a heading

        # Check if the line is a suggestion item
        if line.startswith('-'):
            if current_category:
                suggestion_text = line.lstrip('- ').strip()
                # Avoid adding empty or nonsensical suggestions
                if len(suggestion_text) > 5:
                    suggestions[current_category].append(suggestion_text)

    return suggestions

class SuggestionGenerator:
    def __init__(self, model_path: str = "/home/adwise/Workspace/Models/TheBloke_TinyLlama-1.1B-Chat-v1.0-GGUF_tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"):
        self.model = None
        if os.path.exists(model_path):
            self.model = Llama(model_path=model_path, n_ctx=2048, n_threads=2, verbose=False)
        else:
            print(f"Warning: Model file not found at {model_path}. Suggestion generator will be disabled.")

    def suggest(self, turns, features, topics, geo) -> Dict[str, List[str]]:
        if not self.model:
            return {"topics": [], "questions": [], "sexual": [], "intimacy": []}

        try:
            system_prompt = _create_system_prompt()
            user_prompt = _create_user_prompt(turns, features, topics, geo)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.7,
            )
            raw_text = response['choices'][0]['message']['content']
            return _parse_llm_output_flexibly(raw_text)
        except Exception as e:
            print(f"Error during suggestion generation: {e}")
            return {"topics": [], "questions": [], "sexual": [], "intimacy": []}

suggestion_generator_service = SuggestionGenerator()

def get_suggestion_generator():
    return suggestion_generator_service
