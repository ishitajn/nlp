from ctransformers import AutoModelForCausalLM
import json
from dating_nlp_bot import config

class LLMGenerator:
    def __init__(self, model_name=config.LLM_GENERATOR_MODEL):
        # For ctransformers, we specify the model repository and the specific GGUF file
        self.llm = AutoModelForCausalLM.from_pretrained(
            "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            model_type="llama",
            context_length=2048
        )

    def generate(self, conversation_history: list[dict], analysis: dict) -> dict:
        """
        Generates the conversation_brain content using a light LLM.
        """
        prompt = self._create_prompt(conversation_history, analysis)

        raw_output = self.llm(
            prompt,
            max_new_tokens=256,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2 # Added to discourage repetition
        )

        return self._parse_output(raw_output)

    def _create_prompt(self, conversation_history: list[dict], analysis: dict) -> str:
        """
        Creates a detailed prompt for the LLM.
        """
        conv_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])

        prompt = f"""
<|system|>
You are a dating conversation analyst. Your task is to generate a JSON object for the "conversation_brain" based on the provided conversation history and analysis. The JSON object should contain "predictive_actions" and "memory_layer".

- `suggested_questions`: Generate 3 open-ended questions relevant to the recent topics.
- `goal_tracking`: Generate 2-3 strategic goals for the user's next message.
- `topic_switch_suggestions`: Suggest 3 topics to switch to.
- `recent_topics`: List the main topics discussed recently.

Here is an example:
Conversation:
user: Hey, I saw you like hiking. Me too!
assistant: Awesome! What's your favorite trail? I love the one up by the lake.
user: Oh, I know that one! It's beautiful. I also love cooking, especially Italian food.

Analysis:
- Topics: {{"hiking": ["hiking"], "food": ["cooking", "food"]}}
- Stage: starting
- Flirtation: low

JSON output:
```json
{{
    "predictive_actions": {{
        "suggested_questions": ["What's your favorite thing to cook?", "Besides hiking, what other outdoor activities do you enjoy?", "What kind of food do you enjoy cooking the most?"],
        "goal_tracking": ["Build rapport by sharing a personal story.", "Find more shared hobbies and interests."],
        "topic_switch_suggestions": ["travel", "movies", "music"]
    }},
    "memory_layer": {{
        "recent_topics": ["hiking", "food"]
    }}
}}
```

Now, analyze the following conversation and generate the JSON output.

Conversation:
{conv_str}

Analysis:
- Topics: {analysis.get("topics", {}).get("map", {})}
- Stage: {analysis.get("conversation_dynamics", {}).get("stage")}
- Flirtation: {analysis.get("conversation_dynamics", {}).get("flirtation_level")}

JSON output:
<|assistant|>
```json
"""
        return prompt

    def _parse_output(self, llm_response: str) -> dict:
        """
        Parses the LLM's string output to extract the JSON object.
        """
        try:
            # The model should generate a JSON string, optionally in a markdown block.
            if "```json" in llm_response:
                json_str = llm_response.split("```json")[1].split("```")[0].strip()
            else:
                json_str = llm_response

            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            # Fallback in case the LLM output is not as expected
            return {
                "predictive_actions": {
                    "suggested_questions": [],
                    "goal_tracking": [],
                    "topic_switch_suggestions": []
                },
                "memory_layer": {
                    "recent_topics": []
                }
            }
