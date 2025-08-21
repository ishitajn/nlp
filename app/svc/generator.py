import os
from llama_cpp import Llama
from typing import Dict, Any, List

# --- Prompt Engineering (Refactored for new schema) ---

def pack_context(turns: List[Dict[str, Any]], features: Dict[str, Any], topics: Dict[str, Any], geo: Dict[str, Any]) -> str:
    """
    Packs all analysis data into a single string prompt for the LLM,
    tailored to the new, detailed output schema.
    """
    conversation_history = "\n".join([f"{t.get('role', 'unknown')}: {t.get('content', '')}" for t in turns])

    # Extracting relevant analysis data for the prompt context
    analysis_context = features.get("analysis", {})
    engagement_level = analysis_context.get("match_engaged", "N/A")
    flirtation_level = analysis_context.get("flirtation_level", "N/A")

    prompt = f"""
You are a sophisticated dating assistant AI. Your task is to provide insightful and actionable suggestions to a user based on their conversation history and analysis data.

**Conversation Analysis:**
- Engagement Level: {engagement_level}
- Flirtation Level: {flirtation_level}
- Recent Topics:\n- {topics_str if (topics_str := "\n- ".join(topics.get('recent_topics', []))) else "N/A"}
- Location Context: The user and match are {geo.get('distance_km', 'N/A')} km apart.

**Conversation History:**
{conversation_history}

**Your Task:**
Based on all the context, generate helpful, concise suggestions.
Provide up to two of each category. Do not add any extra text or explanations.
Format your output *exactly* as follows, with each suggestion on a new line:

SUGGESTED_TOPIC: [A specific topic to continue or switch to.]
SUGGESTED_QUESTION: [A relevant, open-ended question to ask.]
SUGGESTED_SEXUAL: [A playful or flirty line. Only generate if engagement and flirtation are NOT "low".]
SUGGESTED_INTIMACY: [A suggestion to build emotional closeness, like sharing a personal story.]
"""
    return prompt.strip()


# --- LLM Suggestion Generator ---

class SuggestionGenerator:
    """
    Uses a TinyLLM to generate conversational suggestions.
    """
    def __init__(self, model_path: str = "/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"):
        self.model = None
        if os.path.exists(model_path):
            self.model = Llama(model_path=model_path, n_ctx=2048, n_threads=2, verbose=False)
        else:
            print(f"Warning: Model file not found at {model_path}. Suggestion generator will use placeholders.")

    def suggest(self, context_pack: str) -> Dict[str, List[str]]:
        """
        Generates suggestions from the LLM based on the provided context prompt.
        """
        if not self.model:
            return self._get_placeholder_suggestions()

        output = self.model(prompt=context_pack, max_tokens=256, temperature=0.7, echo=False)
        raw_text = output['choices'][0]['text']
        return self._parse_llm_output(raw_text)

    def _parse_llm_output(self, text: str) -> Dict[str, List[str]]:
        """
        Parses the raw text output from the LLM into a structured dictionary
        matching the new schema's 'suggestions' object.
        """
        suggestions = {"topics": [], "questions": [], "sexual": [], "intimacy": []}
        lines = text.strip().split('\n')
        for line in lines:
            if line.startswith("SUGGESTED_TOPIC:"):
                suggestions["topics"].append(line.replace("SUGGESTED_TOPIC:", "").strip())
            elif line.startswith("SUGGESTED_QUESTION:"):
                suggestions["questions"].append(line.replace("SUGGESTED_QUESTION:", "").strip())
            elif line.startswith("SUGGESTED_SEXUAL:"):
                suggestions["sexual"].append(line.replace("SUGGESTED_SEXUAL:", "").strip())
            elif line.startswith("SUGGESTED_INTIMACY:"):
                suggestions["intimacy"].append(line.replace("SUGGESTED_INTIMACY:", "").strip())
        return suggestions

    def _get_placeholder_suggestions(self) -> Dict[str, List[str]]:
        """
        Returns placeholder suggestions matching the new schema.
        """
        return {
            "topics": ["Talk about favorite travel destinations.", "Discuss a recent movie you both might have seen."],
            "questions": ["What's something you're excited about right now?", "Is there a skill you're currently learning?"],
            "sexual": [], # Should be empty unless context is appropriate
            "intimacy": ["Share a funny story from your childhood."]
        }

# Global instance
suggestion_generator_service = SuggestionGenerator()

def get_suggestion_generator():
    return suggestion_generator_service
