"""Curator tool: add real doctors, verify them against the official register, issue review codes.

  python manage_doctors.py add --name "Dr. Full Name" --specialization Cardiology --reg-no 12345 \
        --council "Delhi Medical Council" --hospital "Hospital name" --address "Full address" [--phone ..] [--website ..]
  python manage_doctors.py verify 1 --by "Your name" --note "NMC IMR search 2026-11-14: name and number match, active"
  python manage_doctors.py code 1          (prints a single-use review code for doctor 1)
  python manage_doctors.py list
Only add doctors you have really found (hospital website, official directory). Never invent entries.
"""
import argparse

from doctor_module import db
from doctor_module.geo import geocode


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    a = s.add_parser("add")
    for k in ("name", "specialization", "reg-no", "council", "hospital", "address", "phone", "website"):
        a.add_argument("--" + k, default="")
    a.add_argument("--lat", type=float)
    a.add_argument("--lon", type=float)
    v = s.add_parser("verify"); v.add_argument("id", type=int); v.add_argument("--by", required=True); v.add_argument("--note", required=True)
    c = s.add_parser("code"); c.add_argument("id", type=int)
    s.add_parser("list")
    args = p.parse_args()

    if args.cmd == "add":
        lat, lon = args.lat, args.lon
        if lat is None and args.address:
            hit = geocode(args.address)
            if not hit:
                raise SystemExit("Could not locate that address; pass --lat and --lon.")
            lat, lon = hit[0], hit[1]
        new_id = db.add_doctor(name=args.name, specialization=args.specialization, registration_no=args.reg_no or None,
                               registration_council=args.council or None, hospital=args.hospital, address=args.address,
                               lat=lat, lon=lon, phone=args.phone, website=args.website)
        print(f"Added doctor #{new_id} as UNVERIFIED. Check the official register, then run: verify {new_id} ...")
    elif args.cmd == "verify":
        db.verify_doctor(args.id, args.by, args.note)
        print("Marked as verified.")
    elif args.cmd == "code":
        print("Single-use consultation code (give it to the patient once):", db.issue_consultation_code(args.id))
    else:
        for d in db.doctors_with_stats():
            print(f"#{d['id']} {d['name']} | {d['specialization']} | {d['hospital']} | {d['verification_status']} | "
                  f"reviews: {d['review_count']}")


if __name__ == "__main__":
    main()
