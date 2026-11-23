"""Ranking module: Bayesian average rating + proximity. Verified doctors are always listed first."""
from .config import PRIOR_MEAN, PRIOR_STRENGTH, RADIUS_KM, W_PROXIMITY, W_RATING
from .geo import haversine_km


def bayesian_average(avg, n, prior_mean=PRIOR_MEAN, m=PRIOR_STRENGTH):
    """Pulls doctors with few reviews towards the prior mean, so 5.0 from 2 reviews cannot beat 4.6 from 200."""
    if not n or avg is None:
        return prior_mean
    return (n * float(avg) + m * prior_mean) / (n + m)


def score(bayes_avg, distance_km, radius_km=RADIUS_KM):
    rating_part = (bayes_avg - 1) / 4                       # 1..5 stars -> 0..1
    proximity = max(0.0, 1 - distance_km / radius_km)       # 0 km -> 1, edge of radius -> 0
    return W_RATING * rating_part + W_PROXIMITY * proximity


def rank_doctors(docs, user_lat, user_lon, prior_mean=PRIOR_MEAN, radius_km=RADIUS_KM):
    ranked = []
    for d in docs:
        if d.get("lat") is None or d.get("lon") is None:
            continue
        dist = haversine_km(user_lat, user_lon, d["lat"], d["lon"])
        if dist > radius_km:
            continue
        b = bayesian_average(d.get("avg_rating"), d.get("review_count"), prior_mean)
        ranked.append({**d, "distance_km": dist, "bayes_rating": b, "score": score(b, dist, radius_km)})
    verified = sorted((x for x in ranked if x["verification_status"] == "verified"), key=lambda x: -x["score"])
    others = sorted((x for x in ranked if x["verification_status"] != "verified"), key=lambda x: -x["score"])
    return verified, others
