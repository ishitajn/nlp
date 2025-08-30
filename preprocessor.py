"""
This module contains functions for cleaning and preprocessing text, as well as
extracting key phrases (topics) from conversation turns.
"""
import re
import os
import json
import requests
import contractions
from typing import List, Dict

# --- SlangHandler for Dynamic Slang Lookup ---
class SlangHandler:
    """Handles checking for slang terms via the Urban Dictionary API with caching."""
    def __init__(self, cache_path: str, timeout: int = 2):
        self.api_url = "https://api.urbandictionary.com/v0/define"
        self.cache_path = cache_path
        self.timeout = timeout
        self.cache: Dict[str, bool] = self._load_cache()

    def _load_cache(self) -> Dict[str, bool]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f: return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError): return {}
        return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w') as f: json.dump(self.cache, f)

    def is_known_slang(self, term: str) -> bool:
        term = term.lower()
        if term in self.cache: return self.cache[term]
        try:
            response = requests.get(self.api_url, params={"term": term}, timeout=self.timeout)
            response.raise_for_status()
            is_slang = bool(response.json().get("list"))
            self.cache[term] = is_slang
            self._save_cache()
            return is_slang
        except (requests.RequestException, ValueError):
            self.cache[term] = False
            return False

# --- Helper Functions for Phrase Extraction ---
def _is_noise(phrase: str, doc, slang_handler: SlangHandler) -> bool:
    """Determines if a phrase is likely conversational noise."""
    # Note: NOISE_TERMS could also be externalized to config
    noise_terms = {'hmmmm', 'mine', 'mind', 'faves', 'a bit lol'}
    phrase_lower = phrase.lower()
    if phrase_lower in noise_terms: return True

    tokens = [token for token in doc if token.text.lower() in phrase_lower]
    if tokens and all(token.pos_ in {'PRON', 'DET', 'AUX', 'PART', 'INTJ'} for token in tokens): return True

    # Don't filter out short slang terms
    if len(phrase.split()) <= 2 and slang_handler.is_known_slang(phrase): return False

    return False

def _shorten_phrase(phrase: str, kw_extractor) -> str:
    """Shortens a long phrase to its most essential keywords using YAKE."""
    if len(phrase.split()) <= 3: return phrase
    keywords = kw_extractor.extract_keywords(phrase)
    return keywords[0][0] if keywords else phrase

# --- Main Preprocessing Functions ---
def extract_canonical_phrases(text: str, nlp, kw_extractor, slang_handler: SlangHandler) -> List[str]:
    """Extracts key phrases from text using NLP, returning canonical forms."""
    if not text: return []
    
    text = contractions.fix(text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text) # Normalize repeated characters
    
    doc = nlp(text.lower())
    
    candidate_phrases = [chunk.text for chunk in doc.noun_chunks]
    filtered_phrases = [p for p in candidate_phrases if not _is_noise(p, doc, slang_handler) and len(p) > 3]
    canonical_phrases = [_shorten_phrase(p, kw_extractor) for p in filtered_phrases]
    
    return list(dict.fromkeys(canonical_phrases)) # Return unique phrases while preserving order

def clean_text(text: str) -> str:
    """Removes extra whitespace from a string."""
    if not isinstance(text, str): return ""
    return re.sub(r'\s+', ' ', text).strip()

def clean_and_truncate(conversation_history: list, max_turns: int = 20) -> list:
    """Cleans the content of each turn and truncates the conversation history."""
    if not conversation_history: return []

    truncated_history = conversation_history[-max_turns:]
    cleaned_history = [
        {**turn, "content": clean_text(turn.get("content", ""))}
        for turn in truncated_history
        if isinstance(turn, dict) and "content" in turn
    ]
    return cleaned_history