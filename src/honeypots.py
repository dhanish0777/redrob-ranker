"""
honeypots.py
============
Detects "impossible" profiles -- the ~80 honeypot candidates the spec forces
to relevance tier 0. The README and submission_spec describe them as:

  * "8 years of experience at a company founded 3 years ago"
  * "'expert' proficiency in 10 skills with 0 years used"

We can't see company founding dates, but every honeypot betrays itself through
*internal contradictions* in the structured fields. We encode each contradiction
as an independent, individually-defensible check, then combine.

DESIGN PHILOSOPHY (important for the interview):
  - HIGH PRECISION over recall. A false positive here pushes a *real* strong
    candidate down our ranking, which directly costs NDCG. A false negative
    (a missed honeypot) only matters if that honeypot would otherwise rank in
    our top 100 -- and an incoherent profile usually scores low anyway. So we
    only fire on clear, hard logical impossibilities, never on "looks weird".
  - We emit BOTH a hard boolean (is_honeypot) for the safety-net filter AND a
    graded consistency_penalty in [0,1] we can fold into the score, so the
    behaviour degrades gracefully instead of being a single brittle gate.
  - Honeypots are forced to tier 0 in ground truth, so a good scorer should
    avoid them naturally; this detector is a belt-and-suspenders guarantee
    against the >10% honeypot-rate disqualification at Stage 3.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

# Reference "today" for tenure math. The dataset's latest activity dates sit in
# mid-2026, so this anchors career-span calculations for still-current roles.
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

    # ---- 1. Per-role date logic ------------------------------------------
    role_spans_months = []
    earliest_start: Optional[date] = None
    latest_end: Optional[date] = None
    for role in history:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date"))
        is_current = bool(role.get("is_current"))
        dur = int(role.get("duration_months", 0) or 0)

        # is_current must agree with whether end_date is set.
        if is_current and role.get("end_date") not in (None, ""):
            flags.append("current_role_has_end_date")
        if (not is_current) and role.get("end_date") in (None, ""):
            flags.append("past_role_missing_end_date")

        if start and end and end < start:
            flags.append("end_before_start")

        # duration_months should roughly match the date span (allow +/- 2 mo
        # for rounding, and skip when a date is missing).
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

    # ---- 2. Career span vs stated years of experience --------------------
    # The wall-clock span of someone's career cannot be wildly larger than
    # their stated YOE (you can't have a 20-year span with 6 years experience).
    if earliest_start and latest_end:
        career_span_years = _months_between(earliest_start, latest_end) / 12.0
        # Generous slack (3 yrs) for gaps/rounding; only fire on gross mismatch.
        if career_span_years > yoe + 4:
            flags.append("career_span_exceeds_yoe")
        # Conversely, claiming far more experience than the career timeline
        # allows is the "8 yrs at a 3-yr-old company" tell.
        if yoe > career_span_years + 4:
            flags.append("yoe_exceeds_career_span")

    # ---- 3. A single role longer than the whole career is impossible -----
    if role_spans_months and earliest_start and latest_end:
        total_span = _months_between(earliest_start, latest_end)
        if max(role_spans_months) > total_span + 1:
            flags.append("role_longer_than_career")

    # ---- 4. High-proficiency skills with zero usage ----------------------
    # "expert in 10 skills with 0 years used" -- a skill you've never spent
    # time on cannot be advanced/expert. One could be a typo; several is a tell.
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

    # ---- 5. (removed) skill-duration-vs-career --------------------------
    # We tested a rule flagging skills whose duration_months exceeds the
    # candidate's career length. In THIS dataset, skill.duration_months is
    # generated independently of years_of_experience (it's noise), so the rule
    # produced ~14% false positives -- including burying our strongest real
    # candidate (a Pinecone skill listed at 88 months). Dropped on evidence.
    # Lesson kept in code on purpose: rules must be validated against the data,
    # not assumed from the schema.

    # ---- 6. Education year logic -----------------------------------------
    for edu in education:
        sy = edu.get("start_year")
        ey = edu.get("end_year")
        if isinstance(sy, int) and isinstance(ey, int) and ey < sy:
            flags.append("education_end_before_start")
            break

    return flags


# Weights let weak signals contribute a little without, on their own, branding
# a real candidate a honeypot. Hard impossibilities carry most of the mass.
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

# A candidate is treated as a hard honeypot once accumulated weight crosses this.
# Tuned so a single soft flag never trips it, but any one hard impossibility
# (weight 1.0) or two corroborating medium flags does.
HONEYPOT_THRESHOLD = 1.0


def consistency_report(cand: dict) -> dict:
    """
    Full report for one candidate:
      flags            -> list of contradiction names
      penalty_weight   -> summed flag weight
      consistency_penalty -> in [0,1], saturating; 0 = clean, 1 = clearly impossible
      is_honeypot      -> bool, the hard safety-net gate
    """
    flags = detect_inconsistencies(cand)
    weight = sum(_FLAG_WEIGHTS.get(f, 0.3) for f in flags)
    # Saturating map to [0,1]; 2.0 of accumulated weight => fully penalised.
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
