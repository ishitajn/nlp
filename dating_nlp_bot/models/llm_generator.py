import torch
from transformers import pipeline
import json
from .. import config

class LLMGenerator:
    def __init__(self, model_name=config.LLM_GENERATOR_MODEL):
        self.pipe = pipeline("text-generation",
                             model=model_name,
                             torch_dtype=torch.bfloat16,
                             device_map="auto")

    def generate(self, conversation_history: list[dict], analysis: dict) -> dict:
        """
        Generates the conversation_brain content using a light LLM.
        """
        prompt = self._create_prompt(conversation_history, analysis)

        raw_output = self.pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

        llm_response = raw_output[0]['generated_text']

        return self._parse_output(llm_response)

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
- Topics: hiking, cooking, food
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
        "recent_topics": ["hiking", "cooking"]
    }}
}}
```

Now, analyze the following conversation and generate the JSON output.

Conversation:
{conv_str}

Analysis:
- Topics: {analysis.get("topics", {}).get("map", {}).keys()}
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
            # The response will contain the prompt, we need to find the json part
            json_str = llm_response.split("```json")[1].split("```")[0].strip()
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
