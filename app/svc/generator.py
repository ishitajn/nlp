import os
from llama_cpp import Llama
from typing import Dict, Any, List

# --- Prompt Engineering ---

def pack_context(turns: List[Dict[str, Any]], features: Dict[str, Any], topics: Dict[str, Any], geo: Dict[str, Any]) -> str:
    """
    Packs all the analysis data into a single string to be used as a prompt for the LLM.
    """
    # This prompt is crucial for getting good results from the LLM.
    # It needs to be clear, concise, and give the model a specific role and task.

    conversation_history = "\n".join([f"{t['role']}: {t['content']}" for t in turns])

    prompt = f"""
You are a dating assistant AI. Your goal is to help the user have better conversations.
Analyze the following conversation context and generate helpful, concise suggestions.

**Conversation Context:**
- Engagement Level: {features.get('match_engagement_level', {}).get('level', 'N/A')}
- Detected Flirtation: {features.get('sexual_intimacy_flags', {}).get('flirtation_detected', 'N/A')}
- Detected Topics: {', '.join([t['label'] for t in topics.get('topics', [])])}
- Location Info: The user and match are {geo.get('distance_km', 'N/A')} km apart.

**Full Conversation History:**
{conversation_history}

**Your Task:**
Based on the context, provide the following, each on a new line. Do not add any extra text or explanations.
Format your output exactly as follows:
TALKING_POINT: [Suggest a specific, context-aware talking point or observation.]
TALKING_POINT: [Suggest another talking point.]
QUESTION: [Suggest a relevant, open-ended question to ask.]
QUESTION: [Suggest another question.]
SEXUAL_INTIMACY_SUGGESTION: [Suggest a playful or flirty line, ONLY if engagement is high and flirtation is detected.]
NEXT_ACTION: [Suggest a concrete next action, like "Suggest meeting for coffee" or "Ask for their number".]
"""
    return prompt.strip()


# --- LLM Suggestion Generator ---

class SuggestionGenerator:
    """
    Uses a TinyLLM to generate conversational suggestions.
    """
    def __init__(self, model_path: str = "/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"):
        """
        Initializes the generator and loads the GGUF model.
        """
        self.model = None
        if os.path.exists(model_path):
            self.model = Llama(
                model_path=model_path,
                n_ctx=2048,  # Context window size
                n_threads=2, # As per spec
                verbose=False
            )
        else:
            print(f"Warning: Model file not found at {model_path}. Suggestion generator will be disabled.")

    def suggest(self, context_pack: str) -> Dict[str, List[str]]:
        """
        Generates suggestions based on the provided context pack (prompt).
        """
        if not self.model:
            return self._get_placeholder_suggestions()

        # Generate text from the LLM
        output = self.model(
            prompt=context_pack,
            max_tokens=256,
            temperature=0.7,
            stop=["\n\n"], # Stop generation after a double newline
            echo=False
        )

        raw_text = output['choices'][0]['text']
        return self._parse_llm_output(raw_text)

    def _parse_llm_output(self, text: str) -> Dict[str, List[str]]:
        """
        Parses the raw text output from the LLM into a structured dictionary.
        """
        suggestions = {
            "talking_points": [],
            "questions": [],
            "sexual_intimacy_suggestions": [],
            "next_actions": []
        }
        lines = text.strip().split('\n')
        for line in lines:
            if line.startswith("TALKING_POINT:"):
                suggestions["talking_points"].append(line.replace("TALKING_POINT:", "").strip())
            elif line.startswith("QUESTION:"):
                suggestions["questions"].append(line.replace("QUESTION:", "").strip())
            elif line.startswith("SEXUAL_INTIMACY_SUGGESTION:"):
                suggestions["sexual_intimacy_suggestions"].append(line.replace("SEXUAL_INTIMACY_SUGGESTION:", "").strip())
            elif line.startswith("NEXT_ACTION:"):
                suggestions["next_actions"].append(line.replace("NEXT_ACTION:", "").strip())
        return suggestions

    def _get_placeholder_suggestions(self) -> Dict[str, List[str]]:
        """
        Returns placeholder suggestions for when the LLM is not available.
        """
        return {
            "talking_points": ["Mention something from their profile.", "Comment on a shared interest."],
            "questions": ["What are you passionate about?", "What's the best part of your week so far?"],
            "sexual_intimacy_suggestions": [],
            "next_actions": ["Keep the conversation going."]
        }

# Global instance
suggestion_generator_service = SuggestionGenerator()

def get_suggestion_generator():
    return suggestion_generator_service
