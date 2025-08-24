# In preprocessor.py
import re
from services import nlp

# Custom stopwords, including common chat slang and conversational filler
CUSTOM_STOPWORDS = {
    'lol', 'haha', 'hehe', 'ok', 'okay', 'yeah', 'yes', 'no', 'nah', 'ya', 'yep', 'nope',
    'im', 'u', 'r', 'ur', 'y', 'tho', 'btw', 'omg', 'idk', 'tbh', 'imo', 'irl', 'fr',
    'hey', 'hi', 'hello', 'sup', 'yo', 'wassup',
    'like', 'actually', 'basically', 'really', 'gonna', 'wanna', 'gotta', 'kinda',
    'um', 'uh', 'er', 'hmm', 'hmmm',
    'a', 'an', 'the', 'is', 'in', 'at', 'on', 'for', 'to', 'of'
}

# A mapping of common emojis to text for normalization
EMOJI_NORMALIZATION_MAP = {
    '❤️': 'love', '❤': 'love', '😍': 'love eyes', '😂': 'laughing tears', '😭': 'crying',
    '😊': 'smiling', '😉': 'wink', '😘': 'kiss', '👍': 'thumbs up', '🔥': 'fire', '💯': 'hundred',
    '🤔': 'thinking', '🤷': 'shrug', '🤷‍♀️': 'shrug', '🤷‍♂️': 'shrug'
}

def normalize_emojis(text: str) -> str:
    """
    Replaces common emojis in a string with their text representations.
    """
    for emoji, replacement in EMOJI_NORMALIZATION_MAP.items():
        text = text.replace(emoji, f" {replacement} ")
    return text

def preprocess_text(text: str, normalize_emojis_flag: bool = False) -> str:
    """
    Cleans, tokenizes, removes stopwords, and lemmatizes a string of text using spaCy.
    This version preserves emojis by default and can optionally normalize them to text.

    Args:
        text (str): The input text.
        normalize_emojis_flag (bool): If True, common emojis are converted to text.

    Returns:
        str: A cleaned string.
    """
    if not isinstance(text, str) or not nlp:
        return ""

    # Optional: Normalize emojis before other processing
    if normalize_emojis_flag:
        text = normalize_emojis(text)

    # Remove URLs and mentions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)

    # Process the text with spaCy
    doc = nlp(text.lower())

    # Lemmatize and remove stopwords and punctuation
    lemmatized_tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_ not in CUSTOM_STOPWORDS and len(token.lemma_) > 1
    ]

    return " ".join(lemmatized_tokens)


# --- Functions from normalizer.py ---

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

def clean_and_truncate(conversation_history: list, max_turns: int = 20) -> list:
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
