from geopy.geocoders import Nominatim
from geopy.distance import great_circle
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

def get_location_details(location_string: str) -> dict | None:
    """
    Gets location details (latitude, longitude, address, city, state, country) from a location string.
    """
    try:
        geolocator = Nominatim(user_agent="dating_nlp_bot_v2")
        location = geolocator.geocode(location_string, addressdetails=True, timeout=10)
        if location:
            address = location.raw.get('address', {})
            return {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "address": location.address,
                "city": address.get('city', address.get('town', address.get('village'))),
                "state": address.get('state'),
                "country": address.get('country'),
            }
    except (GeocoderTimedOut, GeocoderUnavailable):
        return None
    return None

def calculate_distance(coords1: tuple[float, float], coords2: tuple[float, float]) -> float:
    """
    Calculates the distance in miles between two sets of coordinates.
    """
    return great_circle(coords1, coords2).miles
