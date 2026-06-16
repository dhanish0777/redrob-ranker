"""
score.py
========
The deterministic core ranker. Produces a composite score in roughly [0, 1.2]
per candidate from JD-aligned components, a multiplicative behavioral modifier,
explicit hard penalties for the JD's do-not-want list, and a honeypot guard.

Composite shape (defendable design):
    base   = w_role*role + w_skill*skill + w_exp*exp + w_loc*loc + w_nice*nice
    final  = base * behavioral_modifier * penalty_multiplier * (1 - consistency)
    honeypots: final is crushed to ~0 so they can never reach the top 100.

Weights put role + skills first because the JD's "absolutely need" list is
entirely role/skill, and because role is our strongest defense against the
keyword-stuffer traps. These are the knobs we tune in Phase 3 against the
local gold set -- they are intentionally explicit, not learned, so every
ranking decision can be explained at Stage 5.
"""

from __future__ import annotations

import math

from features import (
    role_relevance, SKILL_WEIGHTS, CV_SPEECH_SKILLS, NLP_IR_SKILLS,
    location_score, months_inactive, short_tenure_fraction,
    has_product_experience, tokenize_skills, is_services_company,
)
from honeypots import consistency_report

# ---- Component weights (sum to 1.0 for the base) --------------------------
W_ROLE, W_SKILL, W_EXP, W_LOC, W_NICE = 0.40, 0.30, 0.15, 0.10, 0.05


# ---------------------------------------------------------------------------
# Skill trust: a skill only counts as much as the evidence behind it.
# This is the second line of defense against keyword stuffing -- listing
# "Pinecone" with 0 endorsements / 0 months / no assessment barely moves the
# needle, while a well-endorsed, long-used, assessed skill counts fully.
# ---------------------------------------------------------------------------
def _skill_trust(skill: dict, assessment_scores: dict) -> float:
    name = (skill.get("name", "") or "").lower()
    endorse = int(skill.get("endorsements", 0) or 0)
    months = int(skill.get("duration_months", 0) or 0)

    # Endorsement evidence: saturating, 10+ endorsements ~ full.
    e = min(1.0, endorse / 10.0)
    # Duration evidence: saturating, 24+ months ~ full.
    d = min(1.0, months / 24.0)
    trust = 0.35 + 0.4 * e + 0.25 * d  # floor 0.35 so a real skill isn't zeroed

    # A Redrob assessment score is hard evidence; let it override upward.
    for k, v in (assessment_scores or {}).items():
        if k.lower() == name:
            trust = max(trust, 0.4 + 0.6 * (float(v) / 100.0))
            break
    return min(1.0, trust)


def skill_score(cand: dict) -> float:
    """Trust-weighted match of candidate skills against the JD taxonomy -> [0,1]."""
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
            neg += -w  # negatives aren't trust-discounted; CV/speech presence is itself the signal
    # Normalize: ~3 strong core skills (trust~0.8) -> near 1.0. Saturating.
    raw = pos - 0.5 * neg
    return max(0.0, min(1.0, raw / 3.0))


def role_score(cand: dict) -> float:
    """
    Recency-weighted role relevance across current title + career history.
    Current role weighted most; the JD wants people *currently* close to the
    work and explicitly down-weights those who stopped coding years ago.
    """
    profile = cand.get("profile", {})
    history = cand.get("career_history", []) or []
    cur = role_relevance(profile.get("current_title", ""))

    if not history:
        return cur
    # Weight roles by recency (current first). Geometric decay.
    weighted, wsum = 0.0, 0.0
    for i, r in enumerate(history):
        w = 0.6 ** i
        weighted += w * role_relevance(r.get("title", ""))
        wsum += w
    hist = weighted / wsum if wsum else 0.0
    # Blend: current title dominates but trajectory matters.
    return min(1.0, 0.6 * cur + 0.4 * hist)


def experience_score(cand: dict) -> float:
    """
    Soft band around the JD's ideal 6-8 yrs (acceptable 5-9), tapering outside.
    Plus a bump for an applied-ML-at-product-company shape. -> [0,1]
    """
    yoe = float(cand.get("profile", {}).get("years_of_experience", 0) or 0)
    # Smooth bump centered at 7, generous shoulders.
    band = math.exp(-((yoe - 7.0) ** 2) / (2 * 3.0 ** 2))  # 1.0 at 7y, ~0.6 at 3y/11y
    # Product-experience presence is part of the ideal shape.
    prod = 1.0 if has_product_experience(cand.get("career_history", [])) else 0.7
    return max(0.0, min(1.0, band * prod))


def nice_to_have_score(cand: dict) -> float:
    """Small bonus for JD 'would like' items: fine-tuning, LtR, distsys, OSS."""
    names = set(tokenize_skills(cand.get("skills", [])))
    bonus = 0.0
    for k in ("lora", "qlora", "peft", "fine-tuning llms", "xgboost",
              "lightgbm", "learning to rank", "spark", "kafka"):
        if k in names:
            bonus += 0.15
    gh = cand.get("redrob_signals", {}).get("github_activity_score", -1)
    if gh and gh > 30:
        bonus += 0.2  # open-source / external-validation signal
    return min(1.0, bonus)


def behavioral_modifier(cand: dict) -> float:
    """
    Multiplicative availability modifier in ~[0.45, 1.1]. A perfect-on-paper
    candidate who is inactive/unresponsive is, per the JD, "not actually
    available" -- so we down-weight rather than exclude.
    """
    sig = cand.get("redrob_signals", {})
    m = 1.0
    # Open to work.
    m *= 1.0 if sig.get("open_to_work_flag") else 0.8
    # Recency: heavy decay past ~3 months inactive.
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
    # Interview reliability (only when present/meaningful).
    ic = sig.get("interview_completion_rate", 0) or 0
    if ic > 0:
        m *= 0.9 + 0.15 * min(1.0, ic)
    # Profile completeness (light).
    pc = sig.get("profile_completeness_score", 0) or 0
    m *= 0.92 + 0.08 * min(1.0, pc / 100.0)
    return max(0.4, min(1.15, m))


def penalty_multiplier(cand: dict) -> tuple[float, list[str]]:
    """
    Hard penalties for the JD's explicit do-not-want list. Returns a
    multiplier in (0,1] and the list of reasons fired (used in reasoning).
    """
    reasons = []
    mult = 1.0
    profile = cand.get("profile", {})
    history = cand.get("career_history", []) or []
    names = set(tokenize_skills(cand.get("skills", [])))

    # (a) Entire career at consulting/services with NO product experience.
    if history and not has_product_experience(history):
        mult *= 0.45
        reasons.append("services-only career (no product-company experience)")

    # (b) CV/speech/robotics primary WITHOUT NLP/IR exposure.
    cv = names & CV_SPEECH_SKILLS
    nlp = names & NLP_IR_SKILLS
    if len(cv) >= 2 and not nlp:
        mult *= 0.55
        reasons.append("CV/speech-heavy skillset without NLP/IR exposure")

    # (c) AI signal is essentially just LangChain/prompt wrappers.
    ai_core = nlp | (names & {"machine learning", "pytorch", "tensorflow",
                              "scikit-learn", "xgboost", "lightgbm", "mlops"})
    wrapper = names & {"langchain", "llamaindex", "prompt engineering"}
    if wrapper and not ai_core:
        mult *= 0.7
        reasons.append("AI experience appears limited to LLM-wrapper tooling")

    # (d) Title-chasing / job-hopping. The JD's concern is non-substantive
    # title inflation by hopping companies -- NOT a fast-progressing specialist.
    # So we only fire when the role profile is not strongly on-target.
    if (short_tenure_fraction(history) >= 0.6 and len(history) >= 3
            and role_score(cand) < 0.8):
        mult *= 0.85
        reasons.append("frequent short tenures (possible title-chasing)")

    return mult, reasons


def score_candidate(cand: dict) -> dict:
    """Full scored record with component breakdown for transparency/reasoning."""
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
        final *= 0.01  # crush honeypots out of contention

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
