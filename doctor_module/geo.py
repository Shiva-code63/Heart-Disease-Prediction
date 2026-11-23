"""Location module: distance maths and free address search (Nominatim)."""
import math
import threading
import time
from functools import lru_cache

import requests

from .config import COUNTRY_CODES, USER_AGENT

_lock, _last_call = threading.Lock(), [0.0]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def round_coords(lat, lon, decimals=3):
    """~110 m precision: all we ever send to an outside service."""
    return round(float(lat), decimals), round(float(lon), decimals)


@lru_cache(maxsize=256)
def geocode(text):
    """City / area / address -> (lat, lon, label) using Nominatim.
    Policy: max 1 request per second, identifying User-Agent, cache results, no autocomplete."""
    text = (text or "").strip()
    if len(text) < 3:
        return None
    with _lock:                                   # keep to <= 1 request per second
        wait = 1.05 - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
    r = requests.get("https://nominatim.openstreetmap.org/search", timeout=10,
                     params={"q": text, "format": "jsonv2", "limit": 1, "countrycodes": COUNTRY_CODES},
                     headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    hits = r.json()
    if not hits:
        return None
    return float(hits[0]["lat"]), float(hits[0]["lon"]), hits[0].get("display_name", text)
