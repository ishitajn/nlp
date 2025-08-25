# In planner.py
import functools
from geopy.geocoders import Nominatim
from geopy.distance import great_circle
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError
from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

geolocator = Nominatim(user_agent="dating_conv_analyzer_v_final", timeout=10)
tf = TimezoneFinder()

@functools.lru_cache(maxsize=128)
def get_location_details(location_string: str) -> Optional[Dict[str, Any]]:
    if not location_string or not isinstance(location_string, str): return None
    try:
        location = geolocator.geocode(location_string, addressdetails=True)
        if location and location.raw.get('address'):
            address = location.raw['address']
            city = address.get('city', address.get('town', address.get('village', address.get('county'))))
            state = address.get('state')
            country = address.get('country')

            # --- FALLBACK LOGIC ---
            if city:
                city_state = city
                if state and city and state != city:
                    city_state = f"{city}, {state}"
            elif country:
                city_state = country # Fallback to country name
            else:
                city_state = location_string # Fallback to the original string if all else fails

            timezone_str = tf.timezone_at(lng=location.longitude, lat=location.latitude)

            return {
                "latitude": location.latitude, "longitude": location.longitude,
                "timezone": timezone_str, "city_state": city_state, "country": country
            }
    except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError) as e:
        logging.error(f"Geocoding service error for '{location_string}': {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during geocoding for '{location_string}': {e}")
    return None

def get_time_of_day(hour: int) -> str:
    if hour < 5: return 'late night'
    elif hour < 8: return 'early morning'
    elif hour < 12: return 'morning'
    elif hour < 14: return 'afternoon'
    elif hour < 17: return 'late afternoon'
    elif hour < 19: return 'evening'
    elif hour < 22: return 'late evening'
    else: return "night"

def compute_geo_time_features(my_location_str: str, their_location_str: str) -> Dict[str, Any]:
    my_details = get_location_details(my_location_str)
    their_details = get_location_details(their_location_str)

    geo_features = { "my_location": {}, "their_location": {}, "distance_km": None, "time_difference_hours": None }

    if my_details:
        geo_features["my_location"]["city_state"] = my_details.get("city_state")
        geo_features["my_location"]["country"] = my_details.get("country")
        if my_details.get("timezone"):
            try:
                my_tz = pytz.timezone(my_details["timezone"])
                my_time_obj = datetime.now(my_tz)
                geo_features["my_location"]["current_time"] = my_time_obj.isoformat()
                geo_features["my_location"]["time_of_day"] = get_time_of_day(my_time_obj.hour)
                geo_features["my_location"]["day_of_week"] = my_time_obj.strftime('%A')
                geo_features["my_location"]["is_weekend"] = my_time_obj.weekday() >= 5  # Saturday or Sunday
                geo_features["my_location"]["timezone"] = str(my_tz)
            except pytz.UnknownTimeZoneError:
                geo_features["my_location"]["timezone"] = "Unknown"

    if their_details:
        geo_features["their_location"]["city_state"] = their_details.get("city_state")
        geo_features["their_location"]["country"] = their_details.get("country")
        if their_details.get("timezone"):
            try:
                their_tz = pytz.timezone(their_details["timezone"])
                their_time_obj = datetime.now(their_tz)
                geo_features["their_location"]["current_time"] = their_time_obj.isoformat()
                geo_features["their_location"]["time_of_day"] = get_time_of_day(their_time_obj.hour)
                geo_features["their_location"]["day_of_week"] = their_time_obj.strftime('%A')
                geo_features["their_location"]["is_weekend"] = their_time_obj.weekday() >= 5
                geo_features["their_location"]["timezone"] = str(their_tz)
            except pytz.UnknownTimeZoneError:
                geo_features["their_location"]["timezone"] = "Unknown"

    if my_details and their_details and my_details.get("latitude") and their_details.get("latitude"):
        my_coords = (my_details["latitude"], my_details["longitude"])
        their_coords = (their_details["latitude"], their_details["longitude"])
        geo_features["distance_km"] = round(great_circle(my_coords, their_coords).kilometers, 2)

        if my_details.get("timezone") and their_details.get("timezone"):
            try:
                my_offset = datetime.now(pytz.timezone(my_details["timezone"])).utcoffset().total_seconds()
                their_offset = datetime.now(pytz.timezone(their_details["timezone"])).utcoffset().total_seconds()
                geo_features["time_difference_hours"] = round((my_offset - their_offset) / 3600, 2)
            except pytz.UnknownTimeZoneError:
                geo_features["time_difference_hours"] = None

    # --- Add is_virtual flag ---
    is_virtual = False
    dist = geo_features.get("distance_km")
    time_diff = geo_features.get("time_difference_hours")
    if dist is not None and dist > 161: # 100 miles is ~161 km
        is_virtual = True
    if time_diff is not None and abs(time_diff) > 2:
        is_virtual = True
    geo_features["is_virtual"] = is_virtual

    return geo_features