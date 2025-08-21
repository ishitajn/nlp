import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """
    Performs basic text cleaning on a single message.
    - Strips leading/trailing whitespace.
    - Normalizes whitespace sequences to a single space.

    Future enhancements could include:
    - Removing metadata like signatures or timestamps.
    - Normalizing emojis or removing repeated characters.
    """
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def clean_and_truncate(conversation_history: List[Dict[str, Any]], max_turns: int = 15) -> List[Dict[str, Any]]:
    """
    Cleans and truncates the conversation history.

    Args:
        conversation_history: The list of conversation turns from the payload.
        max_turns: The maximum number of recent turns to keep.

    Returns:
        A new list containing the cleaned and truncated conversation turns.
    """
    # 1. Truncate conversation to the most recent turns
    truncated_history = conversation_history[-max_turns:]

    # 2. Clean the text content of each turn
    cleaned_history = []
    for turn in truncated_history:
        cleaned_turn = turn.copy()  # Work on a copy
        if "content" in cleaned_turn and isinstance(cleaned_turn["content"], str):
            cleaned_turn["content"] = clean_text(cleaned_turn["content"])
        cleaned_history.append(cleaned_turn)

    return cleaned_history
