"""
This module handles caching of analysis results to improve performance.
It uses an SQLite database for persistence.
"""
import hashlib
import json
import sqlite3
import os
import numpy as np
from typing import Dict, Any, Optional, List

# --- Constants ---
DB_PATH = "data/analysis_cache.sqlite"
TABLE_NAME = "results_cache"

# --- JSON Encoder for Numpy ---
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- Database Initialization ---
def _initialize_cache_db():
    """Initializes the cache table, dropping the old one if schema changed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Check if the table exists and has the old schema (3 columns)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE_NAME,))
        if cursor.fetchone():
            cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
            num_cols = len(cursor.fetchall())
            if num_cols <= 3: # Old schema had cache_key, analysis_data, suggestion_data
                cursor.execute(f"DROP TABLE {TABLE_NAME}")

        # Create the new table with a simpler schema
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            cache_key TEXT PRIMARY KEY,
            cached_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()

# --- Cache Key Generation ---
def _generate_cache_key(
    match_id: str,
    use_enhanced_nlp: bool,
    conversation_history: List[Dict[str, Any]],
    max_turns_for_key: int = 20
) -> str:
    """
    Generates a consistent SHA-256 hash.
    The hash is based on the most recent turns of the conversation, making it
    more robust to minor changes in older parts of the history.
    """
    # Use only the most recent turns for the key
    relevant_history = conversation_history[-max_turns_for_key:]

    # Extract only content for a more stable key
    content_list = [turn.get('content', '') for turn in relevant_history]

    conv_str = json.dumps(content_list, sort_keys=True)
    base_string = f"{match_id}-{use_enhanced_nlp}-{conv_str}"
    return hashlib.sha256(base_string.encode('utf-8')).hexdigest()

# --- Public Cache Interface ---
def get_cached_data(key: str) -> Optional[Dict[str, Any]]:
    """Retrieves cached data from the SQLite database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT cached_data FROM {TABLE_NAME} WHERE cache_key = ?", (key,))
            row = cursor.fetchone()
            if row: return json.loads(row[0])
    except (sqlite3.Error, json.JSONDecodeError) as e:
        print(f"Error getting cached data: {e}")
    return None

def set_cached_data(key: str, data_to_cache: Dict[str, Any]):
    """Stores data in the SQLite cache as a single JSON blob."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            data_str = json.dumps(data_to_cache, cls=NumpyEncoder)
            cursor.execute(f"INSERT OR REPLACE INTO {TABLE_NAME} (cache_key, cached_data) VALUES (?, ?)", (key, data_str))
            conn.commit()
    except sqlite3.Error as e:
        print(f"Error setting cached data: {e}")

def generate_and_check_cache(
    match_id: str,
    use_enhanced_nlp: bool,
    conversation_history: List[Dict[str, Any]]
) -> (Optional[Dict[str, Any]], str):
    """Generates a key and checks the cache for existing data."""
    # Note: Pydantic models from payload must be converted to dicts before being passed here
    serializable_history = [turn if isinstance(turn, dict) else turn.model_dump() for turn in conversation_history]
    cache_key = _generate_cache_key(match_id, use_enhanced_nlp, serializable_history)
    cached_data = get_cached_data(cache_key)
    return cached_data, cache_key

# Initialize the database when the module is loaded
_initialize_cache_db()
