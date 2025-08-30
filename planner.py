"""
This module provides a service for handling geo-location and time-based features.
"""
import functools
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import pytz
from geopy.geocoders import Nominatim
from geopy.distance import great_circle
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError
from timezonefinder import TimezoneFinder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PlannerService:
    """A service class to handle all geo-spatial and time-related calculations."""
    def __init__(self):
        self.geolocator = Nominatim(user_agent="dating_conv_analyzer_v15", timeout=10)
        self.tf = TimezoneFinder()

    @functools.lru_cache(maxsize=128)
    def get_location_details(self, location_string: str) -> Optional[Dict[str, Any]]:
        """Geocodes a location string and returns a dictionary of details."""
        if not location_string or not isinstance(location_string, str): return None
        try:
            location = self.geolocator.geocode(location_string, addressdetails=True)
            if location and location.raw.get('address'):
                address = location.raw['address']
                city = address.get('city', address.get('town', address.get('village', address.get('county'))))
                state = address.get('state')
                country = address.get('country')

                city_state = city if city else country if country else location_string
                if state and city and state != city: city_state = f"{city}, {state}"

                timezone_str = self.tf.timezone_at(lng=location.longitude, lat=location.latitude)
                return {
                    "latitude": location.latitude, "longitude": location.longitude,
                    "timezone": timezone_str, "city_state": city_state, "country": country
                }
        except (GeocoderTimedOut, GeocoderUnavailable, GeocoderServiceError) as e:
            logging.error(f"Geocoding service error for '{location_string}': {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during geocoding for '{location_string}': {e}")
        return None

    def get_time_of_day(self, hour: int) -> str:
        """Categorizes the hour of the day into a human-readable string."""
        if hour < 5: return 'late night'
        if hour < 8: return 'early morning'
        if hour < 12: return 'morning'
        if hour < 14: return 'afternoon'
        if hour < 17: return 'late afternoon'
        if hour < 19: return 'evening'
        if hour < 22: return 'late evening'
        return "night"

    def compute_geo_time_features(self, my_location_str: str, their_location_str: str) -> Dict[str, Any]:
        """Computes all geo and time features for the user and their match."""
        my_details = self.get_location_details(my_location_str)
        their_details = self.get_location_details(their_location_str)

        geo_features = { "my_location": {}, "their_location": {}, "distance_km": None, "time_difference_hours": None }

        if my_details: self._populate_location_time_features(geo_features["my_location"], my_details)
        if their_details: self._populate_location_time_features(geo_features["their_location"], their_details)

        if my_details and their_details:
            my_coords = (my_details["latitude"], my_details["longitude"])
            their_coords = (their_details["latitude"], their_details["longitude"])
            geo_features["distance_km"] = round(great_circle(my_coords, their_coords).kilometers, 2)

            if my_details.get("timezone") and their_details.get("timezone"):
                try:
                    my_offset = datetime.now(pytz.timezone(my_details["timezone"])).utcoffset().total_seconds()
                    their_offset = datetime.now(pytz.timezone(their_details["timezone"])).utcoffset().total_seconds()
                    geo_features["time_difference_hours"] = round((my_offset - their_offset) / 3600, 2)
                except pytz.UnknownTimeZoneError:
                    pass # Keep it None

        dist = geo_features.get("distance_km")
        time_diff = geo_features.get("time_difference_hours")
        geo_features["is_virtual"] = (dist is not None and dist > 161) or (time_diff is not None and abs(time_diff) > 2)

        return geo_features

    def _populate_location_time_features(self, feature_dict: Dict, details: Dict):
        """Helper to populate time features for a given location."""
        feature_dict["city_state"] = details.get("city_state")
        feature_dict["country"] = details.get("country")
        if details.get("timezone"):
            try:
                tz = pytz.timezone(details["timezone"])
                time_obj = datetime.now(tz)
                feature_dict["current_time"] = time_obj.isoformat()
                feature_dict["time_of_day"] = self.get_time_of_day(time_obj.hour)
                feature_dict["day_of_week"] = time_obj.strftime('%A')
                feature_dict["is_weekend"] = time_obj.weekday() >= 5
                feature_dict["timezone"] = str(tz)
            except pytz.UnknownTimeZoneError:
                feature_dict["timezone"] = "Unknown"