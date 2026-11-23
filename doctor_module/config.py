import os

RADIUS_KM = 30                       # search radius around the patient
COUNTRY_CODES = "in"                 # restrict address search to India
# Nominatim / Overpass require an identifying User-Agent. EDIT THIS with your own contact e-mail.
USER_AGENT = "HeartRiskStudentProject/1.0 (educational project; contact: davbennett07@gmail.com)"
# SQLite works out of the box. For MySQL set e.g.
#   DATABASE_URL=mysql+pymysql://user:password@localhost/heart_doctors?charset=utf8mb4
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///doctors.db")

PRIOR_MEAN = 4.0                     # assumed average rating until the app has its own reviews
PRIOR_STRENGTH = 20                  # "m": reviews needed before a doctor's own average counts for half
W_RATING, W_PROXIMITY = 0.65, 0.35   # weights of the ranking score (verified doctors are listed first)
