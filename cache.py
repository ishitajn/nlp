# In cache.py
import hashlib
import json
from typing import Dict, Any, Optional, Tuple

# In-memory cache dictionaries
analysis_cache: Dict[str, Dict[str, Any]] = {}
suggestion_cache: Dict[str, Dict[str, Any]] = {}

def _generate_cache_key(match_id: str, use_enhanced_nlp: bool, conversation_history: list) -> str:
    """Generates a consistent SHA-256 hash for the given inputs."""
    # Serialize the conversation history to a consistent JSON string
    conv_str = json.dumps(conversation_history, sort_keys=True)

    # Create a base string with all components
    base_string = f"{match_id}-{use_enhanced_nlp}-{conv_str}"

    # Return the SHA-256 hash of the base string
    return hashlib.sha256(base_string.encode('utf-8')).hexdigest()

def get_cached_data(key: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Retrieves cached analysis and suggestions.
    Returns a tuple of (analysis, suggestions) if both are found, otherwise None.
    """
    analysis = analysis_cache.get(key)
    suggestions = suggestion_cache.get(key)

    if analysis and suggestions:
        return analysis, suggestions
    return None

def set_cached_data(key: str, analysis_data: Dict[str, Any], suggestion_data: Dict[str, Any]):
    """Stores analysis and suggestion data in the cache."""
    analysis_cache[key] = analysis_data
    suggestion_cache[key] = suggestion_data

def generate_and_check_cache(
    match_id: str,
    use_enhanced_nlp: bool,
    conversation_history: list
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Generates a key and checks the cache for existing data."""
    cache_key = _generate_cache_key(match_id, use_enhanced_nlp, conversation_history)
    return get_cached_data(cache_key), cache_key
