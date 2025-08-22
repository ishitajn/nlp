# In app/svc/planner.py

from geopy.geocoders import Nominatim
from geopy.distance import great_circle
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError
from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

geolocator = Nominatim(user_agent="dating_conv_analyzer_v3", timeout=10)
tf = TimezoneFinder()

def get_location_details(location_string: str) -> Optional[Dict[str, Any]]:
    if not location_string or not isinstance(location_string, str):
        return None
    try:
        location = geolocator.geocode(location_string)
        if location:
            try:
                timezone_str = tf.timezone_at(lng=location.longitude, lat=location.latitude)
                return {
                    "latitude": location.latitude, "longitude": location.longitude,
                    "timezone": timezone_str
                }
            except Exception as e:
                logging.error(f"Error finding timezone for {location_string}: {e}")
                return None
    except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError) as e:
        logging.error(f"Geocoding service error for '{location_string}': {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during geocoding for '{location_string}': {e}")
        return None
    logging.warning(f"Geocoding failed for: {location_string}")
    return None

def compute_geo_time_features(my_location_str: str, their_location_str: str) -> Dict[str, Any]:
    my_details = get_location_details(my_location_str)
    their_details = get_location_details(their_location_str)

    geo_features = {
        "distance_km": None, "time_difference_hours": None, "my_local_time": None,
        "their_local_time": None, "my_time_of_day": "unknown", "their_time_of_day": "unknown",
        "country_difference": None
    }

    if my_details and their_details:
        my_coords = (my_details["latitude"], my_details["longitude"])
        their_coords = (their_details["latitude"], their_details["longitude"])
        geo_features["distance_km"] = great_circle(my_coords, their_coords).kilometers

        my_country = my_location_str.split(',')[-1].strip()
        their_country = their_location_str.split(',')[-1].strip()
        geo_features["country_difference"] = my_country != their_country

        if my_details.get("timezone") and their_details.get("timezone"):
            try:
                now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
                my_tz = pytz.timezone(my_details["timezone"])
                my_time_obj = now_utc.astimezone(my_tz)
                geo_features["my_local_time"] = my_time_obj.strftime('%Y-%m-%d %H:%M:%S')
                geo_features["my_time_of_day"] = get_time_of_day(my_time_obj.hour)

                their_tz = pytz.timezone(their_details["timezone"])
                their_time_obj = now_utc.astimezone(their_tz)
                geo_features["their_local_time"] = their_time_obj.strftime('%Y-%m-%d %H:%M:%S')
                geo_features["their_time_of_day"] = get_time_of_day(their_time_obj.hour)

                time_diff_seconds = my_time_obj.utcoffset().total_seconds() - their_time_obj.utcoffset().total_seconds()
                geo_features["time_difference_hours"] = time_diff_seconds / 3600
            except pytz.UnknownTimeZoneError as e:
                logging.error(f"Unknown timezone error: {e}")
    return geo_features

def get_time_of_day(hour: int) -> str:
    if 5 <= hour < 12: return "morning"
    elif 12 <= hour < 17: return "afternoon"
    elif 17 <= hour < 21: return "evening"
    else: return "night"