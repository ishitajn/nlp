import re
from dating_nlp_bot.utils.text_preprocessing import clean_text

LOCATION_KEYWORDS = ["location", "city", "country", "place", "area", "neighborhood"]
QUESTION_WORDS = [
    "who", "what", "where", "when", "why", "how", "do", "is", "are", "did",
    "does", "will", "would", "can", "could", "should", "have", "has", "am",
    "was", "were", "shall"
]
QUESTION_PATTERN = re.compile(r"^\s*(" + "|".join(QUESTION_WORDS) + r")\b", re.IGNORECASE)

def analyze_response_fast(conversation_history: list[dict]) -> dict:
    """
    Analyzes the last responses from the user and the match (fast mode).
    """
    if not conversation_history:
        return {
            "last_response": "none",
            "last_match_response": {
                "contains_question": False,
                "related_to_location": False,
            },
        }

    # Determine who sent the last message
    last_response_role = conversation_history[-1]['role']
    last_response = "user" if last_response_role == "user" else "match"

    # Find the last message from the match
    last_match_message = next((msg for msg in reversed(conversation_history) if msg['role'] == 'assistant'), None)

    # Analyze last match response
    contains_question = False
    related_to_location = False
    if last_match_message:
        content = clean_text(last_match_message.get("content", ""))
        if '?' in content or QUESTION_PATTERN.search(content):
            contains_question = True
        if any(loc_kw in content for loc_kw in LOCATION_KEYWORDS):
            related_to_location = True

    return {
        "last_response": last_response,
        "last_match_response": {
            "contains_question": contains_question,
            "related_to_location": related_to_location,
        },
    }
