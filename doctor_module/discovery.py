"""Doctor discovery module: nearby hospitals / clinics / doctor facilities from OpenStreetMap (Overpass).
IMPORTANT: these are FACILITIES. An OSM listing never verifies any individual doctor."""
import requests

from .config import RADIUS_KM, USER_AGENT
from .geo import haversine_km, round_coords

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def build_query(lat, lon, radius_km=RADIUS_KM, limit=150):
    lat, lon = round_coords(lat, lon)             # privacy: rounded coordinates only
    r = int(radius_km * 1000)
    return ('[out:json][timeout:25];('
            f'nwr["amenity"~"^(hospital|clinic|doctors)$"](around:{r},{lat},{lon});'
            f'nwr["healthcare"~"^(hospital|clinic|doctor|centre)$"](around:{r},{lat},{lon});'
            f');out center tags {limit};')


def parse_elements(elements, user_lat, user_lon, radius_km=RADIUS_KM):
    out, seen = [], set()
    for el in elements:
        tags = el.get("tags") or {}
        name = (tags.get("name") or "").strip()
        point = el if "lat" in el else el.get("center")
        if not name or not point:
            continue
        key = (el.get("type"), el.get("id"))
        if key in seen:
            continue
        seen.add(key)
        dist = haversine_km(user_lat, user_lon, point["lat"], point["lon"])
        if dist > radius_km:
            continue
        addr = ", ".join(tags[k] for k in ("addr:housename", "addr:housenumber", "addr:street", "addr:suburb",
                                           "addr:city", "addr:postcode") if tags.get(k))
        out.append({
            "name": name, "kind": tags.get("amenity") or tags.get("healthcare") or "facility",
            "cardiology_tag": "cardio" in (tags.get("healthcare:speciality") or "").lower(),
            "lat": point["lat"], "lon": point["lon"], "distance_km": dist, "address": addr,
            "phone": tags.get("phone") or tags.get("contact:phone") or "",
            "website": tags.get("website") or tags.get("contact:website") or "",
            "osm_url": f"https://www.openstreetmap.org/{el.get('type')}/{el.get('id')}"})
    out.sort(key=lambda f: (not f["cardiology_tag"], f["distance_km"]))
    return out


def fetch_facilities(lat, lon, radius_km=RADIUS_KM):
    lat, lon = round_coords(lat, lon)
    r = requests.post(OVERPASS_URL, data={"data": build_query(lat, lon, radius_km)},
                      headers={"User-Agent": USER_AGENT}, timeout=40)
    if r.status_code == 429:
        raise RuntimeError("The free OpenStreetMap service is busy (rate limit). Please try again in a minute.")
    if r.status_code != 200:
        raise RuntimeError(f"OpenStreetMap service returned status {r.status_code}. Please try again later.")
    return parse_elements(r.json().get("elements", []), lat, lon, radius_km)
