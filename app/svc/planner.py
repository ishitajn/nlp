from geopy.geocoders import Nominatim
from geopy.distance import great_circle
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz
from typing import Dict, Any, Optional

# Initialize services once to be reused
geolocator = Nominatim(user_agent="dating_conv_analyzer")
tf = TimezoneFinder()

def get_location_details(location_string: str) -> Optional[Dict[str, Any]]:
    """
    Geocodes a location string to get its latitude, longitude, and timezone.
    """
    try:
        location = geolocator.geocode(location_string)
        if location:
            timezone_str = tf.timezone_at(lng=location.longitude, lat=location.latitude)
            return {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "timezone": timezone_str
            }
    except (GeocoderTimedOut, GeocoderUnavailable):
        # In a real app, log this error
        return None
    return None

def compute_geo_time_features(my_location_str: str, their_location_str: str) -> Dict[str, Any]:
    """
    Computes geographical and time-based features between two locations.
    """
    my_details = get_location_details(my_location_str)
    their_details = get_location_details(their_location_str)

    distance_km = None
    time_difference_hours = None
    my_time = None
    their_time = None
    time_of_day_my_location = "unknown"
    time_of_day_their_location = "unknown"

    if my_details and their_details:
        # Calculate distance
        my_coords = (my_details["latitude"], my_details["longitude"])
        their_coords = (their_details["latitude"], their_details["longitude"])
        distance_km = great_circle(my_coords, their_coords).kilometers

        # Calculate time difference
        if my_details["timezone"] and their_details["timezone"]:
            try:
                now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)

                my_tz = pytz.timezone(my_details["timezone"])
                my_time_obj = now_utc.astimezone(my_tz)
                my_time = my_time_obj.strftime('%Y-%m-%d %H:%M:%S')
                time_of_day_my_location = get_time_of_day(my_time_obj.hour)

                their_tz = pytz.timezone(their_details["timezone"])
                their_time_obj = now_utc.astimezone(their_tz)
                their_time = their_time_obj.strftime('%Y-%m-%d %H:%M:%S')
                time_of_day_their_location = get_time_of_day(their_time_obj.hour)

                time_difference_hours = (my_time_obj.utcoffset().total_seconds() - their_time_obj.utcoffset().total_seconds()) / 3600
            except pytz.UnknownTimeZoneError:
                # Handle cases where timezone string is not valid
                pass

    return {
        "distance_km": distance_km,
        "time_difference_hours": time_difference_hours,
        "my_local_time": my_time,
        "their_local_time": their_time,
        "my_time_of_day": time_of_day_my_location,
        "their_time_of_day": time_of_day_their_location,
        "country_difference": my_location_str.split(',')[-1].strip() != their_location_str.split(',')[-1].strip() if my_details and their_details else None
    }

def get_time_of_day(hour: int) -> str:
    """Categorizes the hour into a time of day."""
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"
