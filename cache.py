# In cache.py
import hashlib
import json
import sqlite3
from typing import Dict, Any, Optional, Tuple
import os

# Define the path to the SQLite database
DB_PATH = "data/embedding_cache.sqlite"
TABLE_NAME = "analysis_cache"

def _initialize_cache_db():
    """Initializes the cache table in the SQLite database if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            cache_key TEXT PRIMARY KEY,
            analysis_data TEXT NOT NULL,
            suggestion_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()

# Initialize the database when the module is loaded
_initialize_cache_db()

def _generate_cache_key(match_id: str, use_enhanced_nlp: bool, conversation_history: list) -> str:
    """Generates a consistent SHA-256 hash for the given inputs."""
    conv_str = json.dumps(conversation_history, sort_keys=True)
    base_string = f"{match_id}-{use_enhanced_nlp}-{conv_str}"
    return hashlib.sha256(base_string.encode('utf-8')).hexdigest()

def get_cached_data(key: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Retrieves cached analysis and suggestions from the SQLite database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT analysis_data, suggestion_data FROM {TABLE_NAME} WHERE cache_key = ?", (key,))
            row = cursor.fetchone()
            if row:
                analysis = json.loads(row[0])
                suggestions = json.loads(row[1])
                return analysis, suggestions
    except (sqlite3.Error, json.JSONDecodeError) as e:
        print(f"Error getting cached data: {e}")
    return None

def set_cached_data(key: str, analysis_data: Dict[str, Any], suggestion_data: Dict[str, Any]):
    """Stores analysis and suggestion data in the SQLite cache."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            analysis_str = json.dumps(analysis_data)
            suggestion_str = json.dumps(suggestion_data)
            cursor.execute(f"""
            INSERT OR REPLACE INTO {TABLE_NAME} (cache_key, analysis_data, suggestion_data)
            VALUES (?, ?, ?)
            """, (key, analysis_str, suggestion_str))
            conn.commit()
    except sqlite3.Error as e:
        print(f"Error setting cached data: {e}")

def generate_and_check_cache(
    match_id: str,
    use_enhanced_nlp: bool,
    conversation_history: list
) -> Tuple[Optional[Tuple[Dict[str, Any], Dict[str, Any]]], str]:
    """Generates a key and checks the cache for existing data."""
    cache_key = _generate_cache_key(match_id, use_enhanced_nlp, conversation_history)
    cached_data = get_cached_data(cache_key)
    return cached_data, cache_key
