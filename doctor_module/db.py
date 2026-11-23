"""Verification + reviews module. SQLAlchemy, so the same code runs on SQLite (default) or MySQL."""
import datetime as dt
import hashlib
import secrets

from sqlalchemy import (CheckConstraint, Column, Date, DateTime, Float, ForeignKey, Integer, MetaData, String,
                        Table, Text, UniqueConstraint, create_engine, func, insert, select, update)

from .config import DATABASE_URL

metadata = MetaData()
doctors = Table(
    "doctors", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(150), nullable=False),
    Column("specialization", String(120), nullable=False),
    Column("registration_no", String(60)),
    Column("registration_council", String(120)),
    Column("hospital", String(200)),
    Column("address", String(300)),
    Column("lat", Float), Column("lon", Float),
    Column("phone", String(40)), Column("website", String(300)),
    # 'verified' is only ever set by verify_doctor(), after a person checked the official register
    Column("verification_status", String(20), nullable=False, default="unverified"),
    Column("verified_on", Date), Column("verified_by", String(120)), Column("verification_note", Text),
    Column("created_at", DateTime, default=dt.datetime.utcnow),
    UniqueConstraint("registration_no", "registration_council", name="uq_registration"))
consultation_codes = Table(
    "consultation_codes", metadata,
    Column("id", Integer, primary_key=True),
    Column("doctor_id", Integer, ForeignKey("doctors.id"), nullable=False),
    Column("code_hash", String(64), nullable=False, unique=True),
    Column("created_at", DateTime, default=dt.datetime.utcnow), Column("used_at", DateTime))
reviews = Table(
    "reviews", metadata,
    Column("id", Integer, primary_key=True),
    Column("doctor_id", Integer, ForeignKey("doctors.id"), nullable=False),
    Column("code_id", Integer, ForeignKey("consultation_codes.id"), nullable=False, unique=True),  # 1 code = 1 review
    Column("rating", Integer, nullable=False), Column("comment", Text),
    Column("source", String(20), nullable=False, default="app"),   # ratings collected by THIS app
    Column("created_at", DateTime, default=dt.datetime.utcnow),
    CheckConstraint("rating BETWEEN 1 AND 5", name="ck_rating"))

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        metadata.create_all(_engine)
    return _engine


def _hash(code):
    return hashlib.sha256(code.strip().upper().encode()).hexdigest()


def add_doctor(**f):
    for need in ("name", "specialization"):
        if not (f.get(need) or "").strip():
            raise ValueError(f"'{need}' is required")
    with get_engine().begin() as c:
        return c.execute(insert(doctors).values(verification_status="unverified", **f)).inserted_primary_key[0]


def verify_doctor(doctor_id, verified_by, note):
    """Mark a doctor verified. Refuses unless every detail was recorded AND a person describes the check."""
    with get_engine().begin() as c:
        d = c.execute(select(doctors).where(doctors.c.id == doctor_id)).mappings().first()
        if not d:
            raise ValueError("No such doctor")
        missing = [k for k in ("name", "specialization", "registration_no", "registration_council", "hospital")
                   if not (d[k] or "").strip()]
        if missing:
            raise ValueError("Cannot verify. Missing: " + ", ".join(missing))
        if not (verified_by or "").strip() or len((note or "").strip()) < 15:
            raise ValueError("Say who checked it, and describe the check "
                             "(e.g. 'NMC Indian Medical Register search on 2026-11-14: name and number match, status active').")
        c.execute(update(doctors).where(doctors.c.id == doctor_id).values(
            verification_status="verified", verified_on=dt.date.today(),
            verified_by=verified_by.strip(), verification_note=note.strip()))


def issue_consultation_code(doctor_id):
    """Single-use code that the doctor / clinic hands to a real patient after a consultation."""
    code = secrets.token_hex(4).upper()
    with get_engine().begin() as c:
        c.execute(insert(consultation_codes).values(doctor_id=doctor_id, code_hash=_hash(code)))
    return code


def submit_review(doctor_id, code, rating, comment=""):
    try:
        rating = int(rating)
    except (TypeError, ValueError):
        return False, "Rating must be a number from 1 to 5."
    comment = (comment or "").strip()
    if not 1 <= rating <= 5:
        return False, "Rating must be between 1 and 5."
    if len(comment) > 500:
        return False, "Please keep the comment under 500 characters."
    with get_engine().begin() as c:
        row = c.execute(select(consultation_codes).where(
            consultation_codes.c.code_hash == _hash(code or ""), consultation_codes.c.doctor_id == doctor_id,
            consultation_codes.c.used_at.is_(None))).mappings().first()
        if not row:
            return False, "That consultation code is invalid for this doctor, or was already used."
        c.execute(insert(reviews).values(doctor_id=doctor_id, code_id=row["id"], rating=rating,
                                         comment=comment, source="app"))
        c.execute(update(consultation_codes).where(consultation_codes.c.id == row["id"]).values(used_at=dt.datetime.utcnow()))
    return True, "Thank you. Your review was added and rankings were updated."


def doctors_with_stats():
    """Every doctor plus live review count / average. Computed on read, so rankings are always current."""
    agg = (select(reviews.c.doctor_id, func.count().label("n"), func.avg(reviews.c.rating).label("avg"))
           .group_by(reviews.c.doctor_id).subquery())
    q = (select(doctors, func.coalesce(agg.c.n, 0).label("review_count"), agg.c.avg.label("avg_rating"))
         .select_from(doctors.outerjoin(agg, doctors.c.id == agg.c.doctor_id)))
    with get_engine().connect() as c:
        return [dict(r) for r in c.execute(q).mappings()]


def global_mean_rating():
    with get_engine().connect() as c:
        v = c.execute(select(func.avg(reviews.c.rating))).scalar()
    return float(v) if v is not None else None
