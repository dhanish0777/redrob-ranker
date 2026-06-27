"""
honeypots.py
============
Detects impossible profiles (honeypots) using internal contradictions.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

# Reference date for tenure math.
REFERENCE_DATE = date(2026, 6, 1)

HIGH_PROFICIENCY = {"advanced", "expert"}


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        y, m, d = (int(x) for x in s.split("-"))
        return date(y, m, d)
    except (ValueError, AttributeError):
        return None


def _months_between(a: date, b: date) -> int:
    """Whole months from a to b (b assumed >= a). Negative if b < a."""
    return (b.year - a.year) * 12 + (b.month - a.month)


def detect_inconsistencies(cand: dict) -> list[str]:
    """
    Return a list of flag names for every hard contradiction found.
    An empty list means the profile is internally consistent.
    """
    flags: list[str] = []
    profile = cand.get("profile", {})
    history = cand.get("career_history", []) or []
    education = cand.get("education", []) or []
    skills = cand.get("skills", []) or []

    yoe = float(profile.get("years_of_experience", 0) or 0)

    # Check per-role date logic
    role_spans_months = []
    earliest_start: Optional[date] = None
    latest_end: Optional[date] = None
    for role in history:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date"))
        is_current = bool(role.get("is_current"))
        dur = int(role.get("duration_months", 0) or 0)

        # is_current must agree with end_date.
        if is_current and role.get("end_date") not in (None, ""):
            flags.append("current_role_has_end_date")
        if (not is_current) and role.get("end_date") in (None, ""):
            flags.append("past_role_missing_end_date")

        if start and end and end < start:
            flags.append("end_before_start")

        # Ensure duration_months roughly matches the date span.
        eff_end = end or (REFERENCE_DATE if is_current else None)
        if start and eff_end:
            span = _months_between(start, eff_end)
            if span >= 0:
                role_spans_months.append(span)
                if abs(span - dur) > 3 and dur > 0:
                    flags.append("duration_date_mismatch")
            if earliest_start is None or start < earliest_start:
                earliest_start = start
            if latest_end is None or eff_end > latest_end:
                latest_end = eff_end

    # Compare career span with stated years of experience
    if earliest_start and latest_end:
        career_span_years = _months_between(earliest_start, latest_end) / 12.0
        # Add slack for gaps/rounding.
        if career_span_years > yoe + 4:
            flags.append("career_span_exceeds_yoe")
        # Check for impossible experience claims.
        if yoe > career_span_years + 4:
            flags.append("yoe_exceeds_career_span")

    # Check if a single role is longer than total career span.
    if role_spans_months and earliest_start and latest_end:
        total_span = _months_between(earliest_start, latest_end)
        if max(role_spans_months) > total_span + 1:
            flags.append("role_longer_than_career")

    # Detect high-proficiency skills with zero usage.
    zero_dur_high_prof = sum(
        1
        for s in skills
        if s.get("proficiency") in HIGH_PROFICIENCY
        and int(s.get("duration_months", 0) or 0) == 0
    )
    if zero_dur_high_prof >= 3:
        flags.append("many_zero_duration_expert_skills")
    elif zero_dur_high_prof >= 1:
        flags.append("some_zero_duration_expert_skills")  # weak signal

    # Check education year logic
    for edu in education:
        sy = edu.get("start_year")
        ey = edu.get("end_year")
        if isinstance(sy, int) and isinstance(ey, int) and ey < sy:
            flags.append("education_end_before_start")
            break

    return flags


# Weights for different contradiction flags.
_FLAG_WEIGHTS = {
    "current_role_has_end_date": 0.5,
    "past_role_missing_end_date": 0.5,
    "end_before_start": 1.0,
    "duration_date_mismatch": 0.4,
    "career_span_exceeds_yoe": 0.7,
    "yoe_exceeds_career_span": 0.9,
    "role_longer_than_career": 1.0,
    "many_zero_duration_expert_skills": 1.0,
    "some_zero_duration_expert_skills": 0.25,
    "education_end_before_start": 0.6,
}

# Threshold for treating a candidate as a honeypot.
HONEYPOT_THRESHOLD = 1.0


def consistency_report(cand: dict) -> dict:
    """
    Generate a consistency report for one candidate.
    """
    flags = detect_inconsistencies(cand)
    weight = sum(_FLAG_WEIGHTS.get(f, 0.3) for f in flags)
    # Map penalty to [0, 1].
    penalty = min(1.0, weight / 2.0)
    return {
        "flags": flags,
        "penalty_weight": round(weight, 3),
        "consistency_penalty": round(penalty, 3),
        "is_honeypot": weight >= HONEYPOT_THRESHOLD,
    }


if __name__ == "__main__":
    import sys
    from loader import load_candidates, default_candidate_path

    path = sys.argv[1] if len(sys.argv) > 1 else default_candidate_path()
    cands = load_candidates(path)

    n_honey = 0
    n_any_flag = 0
    examples = []
    for c in cands:
        rep = consistency_report(c)
        if rep["flags"]:
            n_any_flag += 1
        if rep["is_honeypot"]:
            n_honey += 1
            if len(examples) < 10:
                examples.append((c["candidate_id"], rep["flags"]))

    print(f"Scanned {len(cands):,} candidates")
    print(f"  with >=1 consistency flag : {n_any_flag:,}")
    print(f"  flagged as honeypot       : {n_honey:,} "
          f"({100*n_honey/max(1,len(cands)):.3f}%)")
    if examples:
        print("\nExample honeypots:")
        for cid, fl in examples:
            print(f"  {cid}: {fl}")
