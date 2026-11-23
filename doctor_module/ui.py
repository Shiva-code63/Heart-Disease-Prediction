"""Streamlit section: 'Recommended Cardiologists Near You'. Only calls the other modules."""
from html import escape as esc

import folium
import streamlit as st

from . import db
from .config import PRIOR_MEAN, RADIUS_KM
from .discovery import fetch_facilities
from .geo import geocode, round_coords
from .ranking import rank_doctors

NMC_URL = "https://www.nmc.org.in/information-desk/indian-medical-register/"
CSS = """<style>
.dc {border-radius:16px; padding:16px 18px; margin:0 0 12px; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.12);}
.dc.ok {border-color:rgba(46,204,113,.55); background:rgba(46,204,113,.07);}
.dc h4 {margin:0 0 4px; font-size:1.08rem;}
.dc .meta {font-size:.9rem; opacity:.88; line-height:1.55;}
.badge {display:inline-block; padding:2px 10px; border-radius:20px; font-size:.78rem; font-weight:700; margin-left:8px;}
.badge.ok {background:rgba(46,204,113,.22); color:#7dffb0;} .badge.no {background:rgba(255,193,7,.20); color:#ffd866;}
.badge.osm {background:rgba(66,165,245,.22); color:#9fd0ff;}
.dc a {color:#9fd0ff;}
</style>"""


def _link(url):
    return url if url and url.lower().startswith(("http://", "https://")) else ""


def _contact(phone, website):
    bits = []
    if phone:
        bits.append(f"📞 {esc(phone)}")
    if _link(website):
        bits.append(f'🔗 <a href="{esc(_link(website))}" target="_blank" rel="noopener">Website / appointments</a>')
    return " · ".join(bits)


def _doctor_card(d):
    ok = d["verification_status"] == "verified"
    rating = (f"⭐ {float(d['avg_rating']):.1f} from {d['review_count']} review(s) <i>(collected by this app)</i>"
              if d["review_count"] else "No patient reviews yet")
    proof = (f"Registration <b>{esc(d['registration_no'] or '')}</b> ({esc(d['registration_council'] or '')}), checked "
             f"{d['verified_on']} by {esc(d['verified_by'] or '')}. <a href='{NMC_URL}' target='_blank' rel='noopener'>"
             "Check it yourself</a>" if ok else "Registration has <b>not</b> been checked by us. Please verify it yourself on the official register.")
    badge = '<span class="badge ok">✅ Verified registration</span>' if ok else '<span class="badge no">⚠️ Not verified</span>'
    st.markdown(f"""<div class="dc{' ok' if ok else ''}"><h4>{esc(d['name'])}{badge}</h4><div class="meta">
{esc(d['specialization'])} · {esc(d['hospital'] or '')}<br>📍 {d['distance_km']:.1f} km away · {esc(d['address'] or '')}<br>
{rating}<br>{proof}<br>{_contact(d['phone'], d['website'])}</div></div>""", unsafe_allow_html=True)


def _facility_card(f):
    star = '<span class="badge ok">tagged cardiology on OpenStreetMap</span>' if f["cardiology_tag"] else ""
    st.markdown(f"""<div class="dc"><h4>{esc(f['name'])}<span class="badge osm">🏥 {esc(f['kind'])} · OpenStreetMap listing</span>{star}</h4>
<div class="meta">📍 {f['distance_km']:.1f} km away · {esc(f['address'])}<br>{_contact(f['phone'], f['website'])}
<a href="{esc(f['osm_url'])}" target="_blank" rel="noopener">View on OpenStreetMap</a></div></div>""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner="Looking for hospitals and clinics near you...")
def _facilities(lat, lon):
    return fetch_facilities(lat, lon)


def _map(lat, lon, verified, others, facilities):
    m = folium.Map(location=[lat, lon], zoom_start=11)
    folium.Circle([lat, lon], radius=RADIUS_KM * 1000, color="#E5546C", weight=1, fill=False).add_to(m)
    folium.Marker([lat, lon], tooltip="You (approximate)", icon=folium.Icon(color="red", icon="user", prefix="fa")).add_to(m)
    for items, color, label in ((verified, "green", "Verified doctor"), (others, "gray", "Doctor (not verified)")):
        for d in items:
            folium.Marker([d["lat"], d["lon"]], tooltip=f"{label}: {d['name']} ({d['distance_km']:.1f} km)",
                          icon=folium.Icon(color=color, icon="plus-sign")).add_to(m)
    for f in facilities[:40]:
        folium.CircleMarker([f["lat"], f["lon"]], radius=6, color="#1E88E5", fill=True,
                            tooltip=f"{f['name']} ({f['distance_km']:.1f} km)").add_to(m)
    st.iframe(m.get_root().render(), height=450)
    st.caption("Map © OpenStreetMap contributors. Red circle = 30 km search radius. Green = verified doctor, grey = unverified "
               "doctor, blue dots = hospitals/clinics from OpenStreetMap.")


def _get_location():
    """Browser location (asks the patient's permission) or a manually typed city / area / address."""
    loc = st.session_state.get("doc_loc")
    st.write("Allow your browser to share your location, or type your city / area below. "
             "Your location is used only for this search and is not saved.")
    c1, c2 = st.columns([1, 3])
    with c1:
        try:
            from streamlit_geolocation import streamlit_geolocation
            g = streamlit_geolocation()
        except Exception:
            g = None
    if isinstance(g, dict) and g.get("latitude") is not None and g.get("longitude") is not None:
        loc = {"lat": float(g["latitude"]), "lon": float(g["longitude"]), "label": "your current location (browser)"}
        st.session_state["doc_loc"] = loc
    with c2:
        text = st.text_input("Or enter your city, area or address", key="doc_addr", placeholder="e.g. Sector 62, Noida")
        if st.button("Search this place") and text.strip():
            try:
                hit = geocode(text.strip())
            except Exception:
                hit = None
                st.error("Could not reach the free OpenStreetMap address service. Please try again in a moment.")
            if hit:
                loc = {"lat": hit[0], "lon": hit[1], "label": hit[2]}
                st.session_state["doc_loc"] = loc
            elif hit is None:
                st.warning("We could not find that place. Try adding the city or state name.")
    return loc


def _review_form(verified):
    if not verified:
        return
    with st.expander("⭐ Rate a doctor you consulted (needs a single-use consultation code)"):
        st.caption("After your visit, the doctor's clinic can give you a one-time code. Each code allows one review for "
                   "that doctor only. We never ask for any health details.")
        names = {f"{d['name']} ({d['hospital']})": d["id"] for d in verified}
        who = st.selectbox("Doctor", list(names))
        code = st.text_input("Consultation code")
        stars = st.slider("Rating", 1, 5, 5)
        comment = st.text_area("Comment (optional, max 500 characters)", max_chars=500)
        if st.button("Submit review"):
            ok, msg = db.submit_review(names[who], code, stars, comment)
            (st.success if ok else st.error)(msg)


def render_nearby_cardiologists(prob=None):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="sec-title">🩺 Recommended Cardiologists Near You</div>', unsafe_allow_html=True)
    st.info("The prediction above is an indicative estimate, **not a diagnosis**. Only a qualified doctor can evaluate "
            "your heart health. A ✅ badge means a person checked the doctor's registration on the official register on "
            "the date shown. Ratings come only from patients of this app and do not guarantee quality of care.")
    loc = _get_location()
    if not loc:
        return
    lat, lon = round_coords(loc["lat"], loc["lon"])      # rounded: the only form that leaves this server
    st.success(f"Searching within {RADIUS_KM} km of {loc['label']}")

    prior = db.global_mean_rating() or PRIOR_MEAN
    verified, others = rank_doctors(db.doctors_with_stats(), lat, lon, prior)
    facilities, error = [], None
    try:
        facilities = _facilities(lat, lon)
    except Exception as e:
        error = str(e)

    st.markdown("##### ✅ Verified cardiologists")
    if verified:
        for d in verified:
            _doctor_card(d)
    else:
        st.write(f"We don't have any verified cardiologists within {RADIUS_KM} km of this location yet. "
                 "We only show a verified badge after checking the official register, and we never list made-up doctors.")
    if others:
        with st.expander(f"Doctors we have not verified yet ({len(others)})"):
            for d in others:
                _doctor_card(d)

    st.markdown("##### 🏥 Hospitals and clinics nearby")
    if error:
        st.warning(error)
    elif facilities:
        st.caption("These are facility listings from OpenStreetMap (community-edited, may be incomplete or outdated). "
                   "They do not identify or verify individual doctors. Call ahead to confirm cardiology services.")
        for f in facilities[:8]:
            _facility_card(f)
    else:
        st.write("No hospitals or clinics were found in OpenStreetMap within this radius.")

    _map(lat, lon, verified, others, facilities)
    _review_form(verified)
