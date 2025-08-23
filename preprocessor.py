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
