import os
import json
from llama_cpp import Llama
from typing import Dict, Any, List

def _create_system_prompt() -> str:
    """Creates the system prompt with instructions to generate JSON."""
    return """You are a sophisticated dating assistant AI. Your task is to provide insightful and actionable suggestions based on conversation data.
Analyze the user and conversation data provided.
Your output MUST be a single, valid JSON object and nothing else. Do not include any text, explanations, or markdown before or after the JSON.
The JSON object must have the following structure:
{
  "topics": ["<suggestion>", "<suggestion>"],
  "questions": ["<suggestion>", "string"],
  "sexual": ["<suggestion>", "string"],
  "intimacy": ["<suggestion>", "string"]
}
- Generate up to two suggestions for each category.
- The "sexual" category should only contain suggestions if the conversation's flirtation level is NOT "low". If it is low, return an empty list for "sexual".
- All suggestions must be strings.
"""

def _create_user_prompt(turns: List[Dict[str, Any]], features: Dict[str, Any], topics: Dict[str, Any], geo: Dict[str, Any]) -> str:
    """Packs all analysis data into a single string for the user message."""
    conversation_history = "\n".join([f"{t.get('role', 'unknown')}: {t.get('content', '')}" for t in turns])
    analysis_context = features.get("analysis", {})
    engagement_level = analysis_context.get("match_engaged", "N/A")
    flirtation_level = analysis_context.get("flirtation_level", "N/A")
    recent_topics_str = ", ".join(topics.get('recent_topics', [])) or "N/A"

    return f"""Here is the data to analyze:
**Conversation Analysis:**
- Engagement Level: {engagement_level}
- Flirtation Level: {flirtation_level}
- Recent Topics: {recent_topics_str}
- Location Context: The user and match are {geo.get('distance_km', 'N/A')} km apart.

**Conversation History:**
{conversation_history}

Now, provide your analysis in the specified JSON format.
"""

def _parse_json_output(raw_text: str) -> Dict[str, List[str]]:
    """
    Parses the raw text output from the LLM, expecting a JSON object.
    """
    try:
        # Find the JSON block, which might be enclosed in markdown backticks
        if "```json" in raw_text:
            json_str = raw_text.split("```json")[1].split("```")[0]
        elif "```" in raw_text:
            json_str = raw_text.split("```")[1].split("```")[0]
        else:
            json_str = raw_text

        data = json.loads(json_str.strip())

        # Validate structure
        expected_keys = {"topics", "questions", "sexual", "intimacy"}
        if not isinstance(data, dict) or not expected_keys.issubset(data.keys()):
            print(f"Warning: LLM output is valid JSON but has wrong structure. Output: {data}")
            return {"topics": [], "questions": [], "sexual": [], "intimacy": []}

        return data
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error decoding LLM JSON output: {e}. Raw text: '{raw_text}'")
        return {"topics": [], "questions": [], "sexual": [], "intimacy": []}

class SuggestionGenerator:
    def __init__(self, model_path: str = "/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"):
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
                temperature=0.5,
            )
            raw_text = response['choices'][0]['message']['content']
            return _parse_json_output(raw_text)
        except Exception as e:
            print(f"Error during suggestion generation: {e}")
            return {"topics": [], "questions": [], "sexual": [], "intimacy": []}

suggestion_generator_service = SuggestionGenerator()

def get_suggestion_generator():
    return suggestion_generator_service
