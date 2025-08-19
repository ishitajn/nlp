from datetime import datetime
import pytz
from timezonefinder import TimezoneFinder

def get_timezone(lat: float, lon: float) -> str | None:
    """
    Gets the timezone for a given latitude and longitude.
    """
    tf = TimezoneFinder()
    return tf.timezone_at(lng=lon, lat=lat)

def get_time_of_day(timezone_str: str) -> str:
    """
    Gets the time of day (Morning, Afternoon, Evening, Night) for a given timezone.
    """
    try:
        user_tz = pytz.timezone(timezone_str)
        user_time = datetime.now(user_tz)
        hour = user_time.hour
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"
    except pytz.UnknownTimeZoneError:
        return "Unknown"

def get_timezone_difference(tz1_str: str, tz2_str: str) -> int:
    """
    Calculates the time difference in hours between two timezones.
    """
    try:
        tz1 = pytz.timezone(tz1_str)
        tz2 = pytz.timezone(tz2_str)
        now = datetime.utcnow()
        offset1 = tz1.utcoffset(now)
        offset2 = tz2.utcoffset(now)
        return int((offset1 - offset2).total_seconds() / 3600)
    except pytz.UnknownTimeZoneError:
        return 0
