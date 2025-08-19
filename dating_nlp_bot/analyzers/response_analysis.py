from datetime import datetime, timedelta
from ..utils.text_preprocessing import clean_text

LOCATION_KEYWORDS = ["location", "city", "country", "place", "area", "neighborhood"]

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
    last_match_message = None
    for msg in reversed(conversation_history):
        if msg['role'] == 'assistant':
            last_match_message = msg
            break

    # Analyze last match response
    contains_question = False
    related_to_location = False
    if last_match_message:
        content = clean_text(last_match_message.get("content", ""))
        contains_question = '?' in content
        if any(loc_kw in content for loc_kw in LOCATION_KEYWORDS):
            related_to_location = True

    return {
        "last_response": last_response,
        "last_match_response": {
            "contains_question": contains_question,
            "related_to_location": related_to_location,
        },
    }
