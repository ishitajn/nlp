import re
import string

def clean_text(text: str) -> str:
    """
    Cleans the input text by converting to lowercase and removing punctuation.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text
