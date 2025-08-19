from dating_nlp_bot.utils import location_utils, time_utils
from datetime import datetime
import pytz

def analyze_geo_time(my_location_str: str, their_location_str: str) -> dict:
    """
    Analyzes geographical and time-based context.
    """
    user_loc = location_utils.get_location_details(my_location_str)
    match_loc = location_utils.get_location_details(their_location_str)

    user_context = {"address": None, "city": None, "state": None, "country": None, "timeZone": None, "timeOfDay": None, "current_date_time": ""}
    match_context = {"address": None, "city": None, "state": None, "country": None, "timeZone": None, "timeOfDay": None, "current_date_time": ""}
    distance = None
    tz_diff = None
    country_diff = False
    is_virtual = False

    if user_loc:
        user_tz_str = time_utils.get_timezone(user_loc['latitude'], user_loc['longitude'])
        user_time_str = ""
        if user_tz_str:
            try:
                user_tz = pytz.timezone(user_tz_str)
                user_time_str = datetime.now(user_tz).isoformat()
            except pytz.UnknownTimeZoneError:
                user_time_str = ""

        user_context = {
            "address": user_loc['address'], "city": user_loc['city'], "state": user_loc['state'],
            "country": user_loc['country'], "timeZone": user_tz_str,
            "timeOfDay": time_utils.get_time_of_day(user_tz_str) if user_tz_str else None,
            "current_date_time": user_time_str
        }

    if match_loc:
        match_tz_str = time_utils.get_timezone(match_loc['latitude'], match_loc['longitude'])
        match_time_str = ""
        if match_tz_str:
            try:
                match_tz = pytz.timezone(match_tz_str)
                match_time_str = datetime.now(match_tz).isoformat()
            except pytz.UnknownTimeZoneError:
                match_time_str = ""

        match_context = {
            "address": match_loc['address'], "city": match_loc['city'], "state": match_loc['state'],
            "country": match_loc['country'], "timeZone": match_tz_str,
            "timeOfDay": time_utils.get_time_of_day(match_tz_str) if match_tz_str else None,
            "current_date_time": match_time_str
        }

    if user_loc and match_loc:
        user_coords = (user_loc['latitude'], user_loc['longitude'])
        match_coords = (match_loc['latitude'], match_loc['longitude'])
        distance = location_utils.calculate_distance(user_coords, match_coords)

        if user_context['timeZone'] and match_context['timeZone']:
            tz_diff = time_utils.get_timezone_difference(user_context['timeZone'], match_context['timeZone'])

        country_diff = user_context['country'] != match_context['country']
        is_virtual = distance > 100 or country_diff

    return {
        "userLocation": user_context,
        "matchLocation": match_context,
        "distance_miles": distance,
        "timeZoneDifference": tz_diff,
        "countryDifference": country_diff,
        "isVirtual": is_virtual,
    }
