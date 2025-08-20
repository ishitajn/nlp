import re
import string

def clean_text(text: str) -> str:
    """
    Cleans the input text by converting to lowercase and removing punctuation,
    but preserves question marks.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove all punctuation except for the question mark
    punctuation_to_remove = string.punctuation.replace("?", "")
    text = re.sub(f"[{re.escape(punctuation_to_remove)}]", "", text)
    return text
