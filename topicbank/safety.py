import re
from langdetect import detect, LangDetectException

def safe_language(t, allowed=("en",)):
    try:
        return detect(t) in allowed
    except LangDetectException:
        return False

def hard_block(t, ban_patterns):
    for pat in ban_patterns:
        if re.search(pat, t, re.I):
            return True
    return False

def explicit_level(t, lvl2_keys, lvl3_keys):
    if re.search(lvl3_keys, t, re.I):
        return 3
    if re.search(lvl2_keys, t, re.I):
        return 2
    # mild/flirty
    if re.search(r"\b(kiss|flirt|chemistry|turn\s*on|sexy)\b", t, re.I):
        return 1
    return 0
