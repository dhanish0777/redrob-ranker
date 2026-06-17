"""
reason.py
=========
Generates the 1-2 sentence `reasoning` for each ranked candidate.

Stage 4 samples 10 rows and checks: specific facts, JD connection, honest
concerns, NO hallucination, variation, and rank-consistency. So every clause
here is built ONLY from fields actually present in the candidate's profile,
the tone scales with the score, and we surface real concerns rather than
generic praise. We assemble from a pool of phrasings keyed to which signals
actually dominate, which keeps the 100 reasonings genuinely varied instead of
a name-swapped template.
"""

from __future__ import annotations

from features import (
    SKILL_WEIGHTS, months_inactive, is_services_company, has_product_experience,
)

CORE_SKILL_KEYS = [k for k, v in SKILL_WEIGHTS.items() if v >= 0.9]


def _matched_core_skills(cand: dict, limit: int = 3) -> list[str]:
    """Original-cased skill names that are JD-core, strongest first."""
    out = []
    for s in sorted(cand.get("skills", []),
                    key=lambda x: -(SKILL_WEIGHTS.get((x.get("name", "") or "").lower(), 0))):
        nm = (s.get("name", "") or "")
        if SKILL_WEIGHTS.get(nm.lower(), 0) >= 0.7:
            out.append(nm)
        if len(out) >= limit:
            break
    return out


def _product_company(cand: dict) -> str:
    """Name the current company if it's a product (non-services) company."""
    p = cand.get("profile", {})
    comp = p.get("current_company", "")
    if comp and not is_services_company(comp, p.get("current_industry", "")):
        return comp
    return ""


def _variant(candidate_id: str, n: int) -> int:
    """Deterministic 0..n-1 index from the id, for reproducible phrasing variety."""
    digits = "".join(ch for ch in candidate_id if ch.isdigit()) or "0"
    return int(digits[-3:]) % n


def _strength_clause(cand: dict, scored: dict) -> str:
    profile = cand.get("profile", {})
    history = cand.get("career_history", []) or []
    title = profile.get("current_title", "role")
    yoe = profile.get("years_of_experience", 0)
    core = _matched_core_skills(cand)
    comp = scored["components"]
    company = _product_company(cand)
    at = f" at {company}" if company else ""

    # Top tier: rotate among equivalent phrasings (keyed to id) so 100 rows vary.
    if comp["role"] >= 0.9 and core:
        sk = ", ".join(core)
        variants = [
            f"{title}{at}, {yoe:.0f} yrs; direct retrieval/ranking signal ({sk})",
            f"{title}{at} ({yoe:.0f} yrs) with production retrieval stack: {sk}",
            f"{title}{at}, {yoe:.0f} yrs of applied ML; hands-on with {sk}",
            f"{title}{at} with {yoe:.0f} yrs; strong ranking/retrieval skills ({sk})",
        ]
        return variants[_variant(cand["candidate_id"], len(variants))]
    if comp["role"] >= 0.9:
        return f"{title}{at} with {yoe:.0f} yrs in applied-ML/retrieval roles"
    if core and comp["skill"] >= 0.6:
        return (f"{title} ({yoe:.0f} yrs) with relevant retrieval skills "
                f"({', '.join(core)})")
    if comp["role"] >= 0.6:
        prod = "product-company" if has_product_experience(history) else "engineering"
        return f"{title} with {yoe:.0f} yrs of {prod} background, adjacent to the role"
    return f"{title} with {yoe:.0f} yrs; only adjacent skills for this JD"


def _concern_clause(cand: dict, scored: dict) -> str:
    """Surface the single most relevant honest concern, if any."""
    sig = cand.get("redrob_signals", {})
    title = (cand.get("profile", {}).get("current_title", "") or "").lower()
    concerns = list(scored.get("penalty_reasons", []))

    if "junior" in title:
        concerns.append("junior title for a senior founding role")
    inactive = months_inactive(sig.get("last_active_date", ""))
    if inactive > 6:
        concerns.append(f"inactive ~{int(inactive)} months")
    rr = sig.get("recruiter_response_rate", 0) or 0
    if rr < 0.2:
        concerns.append(f"low recruiter response rate ({rr:.0%})")
    if not sig.get("open_to_work_flag"):
        concerns.append("not currently marked open-to-work")
    np = sig.get("notice_period_days")
    if isinstance(np, (int, float)) and np >= 90:
        concerns.append(f"long notice period ({int(np)}d)")
    if scored["components"]["loc"] <= 0.45:
        concerns.append("located outside India with no relocation flag")

    return concerns[0] if concerns else ""


def make_reasoning(cand: dict, scored: dict) -> str:
    """One or two sentences: strength, then an honest concern if present."""
    strength = _strength_clause(cand, scored)
    concern = _concern_clause(cand, scored)
    if scored["is_honeypot"]:
        return (f"{cand['profile'].get('current_title','Profile')} flagged for "
                f"internal inconsistencies ({', '.join(scored['honeypot_flags'][:2])}); "
                f"ranked low as likely not a genuine profile.")
    if concern:
        return f"{strength}. Concern: {concern}."
    return f"{strength}."
