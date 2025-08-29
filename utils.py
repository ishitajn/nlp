"""
Shared utility functions for the conversation analysis pipeline.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

def parse_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """
    Robustly parses a timestamp string into a timezone-aware datetime object.
    Handles multiple common formats, including ISO 8601 with and without 'Z'.
    """
    if not ts_str or not isinstance(ts_str, str): return None
    if ts_str.endswith('Z'): ts_str = ts_str[:-1] + '+00:00'
    formats_to_try = ["%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
    for fmt in formats_to_try:
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except (ValueError, TypeError): continue
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except (ValueError, TypeError):
        logging.warning(f"Could not parse timestamp: {ts_str}")
    return None
