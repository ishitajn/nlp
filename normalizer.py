# In app/svc/normalizer.py

import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """
    Performs basic text cleaning on a single message.
    - Strips leading/trailing whitespace.
    - Normalizes whitespace sequences to a single space.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def clean_and_truncate(conversation_history: List[Dict[str, Any]], max_turns: int = 20) -> List[Dict[str, Any]]:
    """
    Cleans and truncates the conversation history to the most recent turns.

    Args:
        conversation_history: The list of conversation turns from the payload.
        max_turns: The maximum number of recent turns to keep for analysis.

    Returns:
        A new list containing the cleaned and truncated conversation turns.
    """
    if not conversation_history:
        return []

    # 1. Truncate conversation to the most recent turns
    truncated_history = conversation_history[-max_turns:]

    # 2. Clean the text content of each turn
    cleaned_history = []
    for turn in truncated_history:
        # Ensure the turn is a dictionary and has the expected keys
        if isinstance(turn, dict) and "content" in turn:
            cleaned_turn = turn.copy()  # Work on a copy
            cleaned_turn["content"] = clean_text(cleaned_turn["content"])
            cleaned_history.append(cleaned_turn)

    return cleaned_history