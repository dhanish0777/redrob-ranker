"""
score.py
========
Deterministic core ranker.
"""

from __future__ import annotations

import math

from features import (
    role_relevance, SKILL_WEIGHTS, CV_SPEECH_SKILLS, NLP_IR_SKILLS,
    location_score, months_inactive, short_tenure_fraction,
    has_product_experience, tokenize_skills, is_services_company,
)
from honeypots import consistency_report

# Component weights.
W_ROLE, W_SKILL, W_EXP, W_LOC, W_NICE = 0.40, 0.30, 0.15, 0.10, 0.05


# Calculate skill trust based on evidence.
def _skill_trust(skill: dict, assessment_scores: dict) -> float:
    name = (skill.get("name", "") or "").lower()
    endorse = int(skill.get("endorsements", 0) or 0)
    months = int(skill.get("duration_months", 0) or 0)

    # Calculate endorsement evidence.
    e = min(1.0, endorse / 10.0)
    # Calculate duration evidence.
    d = min(1.0, months / 24.0)
    trust = 0.35 + 0.4 * e + 0.25 * d  # floor 0.35 so a real skill isn't zeroed

    # Use assessment score if available.
    for k, v in (assessment_scores or {}).items():
        if k.lower() == name:
            trust = max(trust, 0.4 + 0.6 * (float(v) / 100.0))
            break
    return min(1.0, trust)


def skill_score(cand: dict) -> float:
    """Calculate skill score."""
    skills = cand.get("skills", []) or []
    assess = cand.get("redrob_signals", {}).get("skill_assessment_scores", {})
    pos = 0.0
    neg = 0.0
    for s in skills:
        name = (s.get("name", "") or "").lower()
        w = SKILL_WEIGHTS.get(name)
        if w is None:
            continue
        if w >= 0:
            pos += w * _skill_trust(s, assess)
        else:
            neg += -w
    # Normalize the score.
    raw = pos - 0.5 * neg
    return max(0.0, min(1.0, raw / 3.0))


def role_score(cand: dict) -> float:
    """Calculate role score."""
    profile = cand.get("profile", {})
    history = cand.get("career_history", []) or []
    cur = role_relevance(profile.get("current_title", ""))

    if not history:
        return cur
    # Weight roles by recency.
    weighted, wsum = 0.0, 0.0
    for i, r in enumerate(history):
        w = 0.6 ** i
        weighted += w * role_relevance(r.get("title", ""))
        wsum += w
    hist = weighted / wsum if wsum else 0.0
    # Blend current and history scores.
    return min(1.0, 0.6 * cur + 0.4 * hist)


def experience_score(cand: dict) -> float:
    """Calculate experience score."""
    yoe = float(cand.get("profile", {}).get("years_of_experience", 0) or 0)
    # Calculate experience band multiplier.
    band = math.exp(-((yoe - 7.0) ** 2) / (2 * 3.0 ** 2))  # 1.0 at 7y, ~0.6 at 3y/11y
    # Check for product experience.
    prod = 1.0 if has_product_experience(cand.get("career_history", [])) else 0.7
    return max(0.0, min(1.0, band * prod))


def nice_to_have_score(cand: dict) -> float:
    """Calculate nice-to-have score."""
    names = set(tokenize_skills(cand.get("skills", [])))
    bonus = 0.0
    for k in ("lora", "qlora", "peft", "fine-tuning llms", "xgboost",
              "lightgbm", "learning to rank", "spark", "kafka"):
        if k in names:
            bonus += 0.15
    gh = cand.get("redrob_signals", {}).get("github_activity_score", -1)
    if gh and gh > 30:
        bonus += 0.2
    return min(1.0, bonus)


def behavioral_modifier(cand: dict) -> float:
    """Calculate behavioral modifier."""
    sig = cand.get("redrob_signals", {})
    m = 1.0
    # Open to work.
    m *= 1.0 if sig.get("open_to_work_flag") else 0.8
    # Calculate recency multiplier.
    inactive = months_inactive(sig.get("last_active_date", ""))
    if inactive <= 1:
        m *= 1.05
    elif inactive <= 3:
        m *= 1.0
    elif inactive <= 6:
        m *= 0.85
    else:
        m *= 0.65
    # Responsiveness to recruiters.
    rr = sig.get("recruiter_response_rate", 0) or 0
    m *= 0.75 + 0.35 * min(1.0, rr / 0.7)  # 0.75 at 0%, ~1.1 at 70%+
    # Interview reliability.
    ic = sig.get("interview_completion_rate", 0) or 0
    if ic > 0:
        m *= 0.9 + 0.15 * min(1.0, ic)
    # Profile completeness (light).
    pc = sig.get("profile_completeness_score", 0) or 0
    m *= 0.92 + 0.08 * min(1.0, pc / 100.0)
    # Notice period modifier.
    np = sig.get("notice_period_days")
    if isinstance(np, (int, float)):
        if np <= 30:
            m *= 1.0
        elif np <= 60:
            m *= 0.97
        elif np <= 90:
            m *= 0.93
        else:
            m *= 0.88
    return max(0.4, min(1.15, m))


def penalty_multiplier(cand: dict) -> tuple[float, list[str]]:
    """Calculate penalty multiplier."""
    reasons = []
    mult = 1.0
    profile = cand.get("profile", {})
    history = cand.get("career_history", []) or []
    names = set(tokenize_skills(cand.get("skills", [])))

    # Check for services-only career.
    if history and not has_product_experience(history):
        mult *= 0.45
        reasons.append("services-only career (no product-company experience)")

    # Check for CV/speech focus without NLP/IR.
    cv = names & CV_SPEECH_SKILLS
    nlp = names & NLP_IR_SKILLS
    if len(cv) >= 2 and not nlp:
        mult *= 0.55
        reasons.append("CV/speech-heavy skillset without NLP/IR exposure")

    # Check for LLM wrappers without core AI skills.
    ai_core = nlp | (names & {"machine learning", "pytorch", "tensorflow",
                              "scikit-learn", "xgboost", "lightgbm", "mlops"})
    wrapper = names & {"langchain", "llamaindex", "prompt engineering"}
    if wrapper and not ai_core:
        mult *= 0.7
        reasons.append("AI experience appears limited to LLM-wrapper tooling")

    # Check for frequent short tenures.
    if (short_tenure_fraction(history) >= 0.6 and len(history) >= 3
            and role_score(cand) < 0.8):
        mult *= 0.85
        reasons.append("frequent short tenures (possible title-chasing)")

    return mult, reasons


def score_candidate(cand: dict) -> dict:
    """Score a candidate."""
    rep = consistency_report(cand)
    comp = {
        "role": role_score(cand),
        "skill": skill_score(cand),
        "exp": experience_score(cand),
        "loc": location_score(cand.get("profile", {}),
                              cand.get("redrob_signals", {}).get("willing_to_relocate", False)),
        "nice": nice_to_have_score(cand),
    }
    base = (W_ROLE * comp["role"] + W_SKILL * comp["skill"] + W_EXP * comp["exp"]
            + W_LOC * comp["loc"] + W_NICE * comp["nice"])
    bmod = behavioral_modifier(cand)
    pmult, preasons = penalty_multiplier(cand)

    final = base * bmod * pmult * (1.0 - rep["consistency_penalty"])
    if rep["is_honeypot"]:
        final *= 0.01

    return {
        "candidate_id": cand["candidate_id"],
        "score": final,
        "base": base,
        "components": comp,
        "behavioral_modifier": bmod,
        "penalty_multiplier": pmult,
        "penalty_reasons": preasons,
        "is_honeypot": rep["is_honeypot"],
        "honeypot_flags": rep["flags"],
    }


if __name__ == "__main__":
    import sys
    from loader import load_candidates, default_candidate_path

    path = sys.argv[1] if len(sys.argv) > 1 else default_candidate_path()
    cands = load_candidates(path)
    scored = sorted((score_candidate(c) for c in cands),
                    key=lambda r: (-r["score"], r["candidate_id"]))
    print(f"Scored {len(scored)} candidates. Top 15:\n")
    print(f"{'rank':>4} {'cid':<14} {'score':>6}  {'role':>4} {'skill':>5} "
          f"{'exp':>4} {'bmod':>4} {'pen':>4}  flags/penalties")
    for i, r in enumerate(scored[:15], 1):
        c = r["components"]
        note = ", ".join(r["penalty_reasons"]) or ("HONEYPOT" if r["is_honeypot"] else "")
        print(f"{i:>4} {r['candidate_id']:<14} {r['score']:.3f}  "
              f"{c['role']:.2f} {c['skill']:.2f} {c['exp']:.2f} "
              f"{r['behavioral_modifier']:.2f} {r['penalty_multiplier']:.2f}  {note}")
