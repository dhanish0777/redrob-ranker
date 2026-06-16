"""
JD-derived ranking INVARIANTS -- our offline evaluation with no leaderboard.

Each test encodes a rule the JD states directly, expressed as an ordering the
scorer must respect. We build controlled archetypes (one variable changed at a
time) so a failure points at exactly one cause. This is both our regression
guard and the answer to "how did you validate without feedback?" at Stage 5.

Run:  python tests/test_invariants.py
"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from score import score_candidate  # noqa: E402


def _s(cand):
    return score_candidate(cand)["score"]


# A clean, genuinely strong reference candidate: applied-ML title at a product
# company, real retrieval skills with evidence, active, India, short notice.
BASE = {
    "candidate_id": "CAND_1000000",
    "profile": {
        "anonymized_name": "Test Ref", "headline": "ML Engineer",
        "summary": "Built production retrieval and ranking systems.",
        "location": "Pune, Maharashtra", "country": "India",
        "years_of_experience": 7.0, "current_title": "Machine Learning Engineer",
        "current_company": "Swiggy", "current_company_size": "1001-5000",
        "current_industry": "Food Delivery",
    },
    "career_history": [{
        "company": "Swiggy", "title": "Machine Learning Engineer",
        "start_date": "2022-01-01", "end_date": None, "duration_months": 41,
        "is_current": True, "industry": "Food Delivery", "company_size": "1001-5000",
        "description": "Built and shipped ranking models for the discovery feed.",
    }, {
        "company": "Flipkart", "title": "Applied ML Engineer",
        "start_date": "2019-01-01", "end_date": "2021-12-01", "duration_months": 35,
        "is_current": False, "industry": "E-commerce", "company_size": "10001+",
        "description": "Search relevance and embeddings-based retrieval.",
    }],
    "education": [{"institution": "IIT", "degree": "B.Tech",
                   "field_of_study": "CS", "start_year": 2011, "end_year": 2015,
                   "grade": "8.5", "tier": "tier_1"}],
    "skills": [
        {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 30},
        {"name": "Sentence Transformers", "proficiency": "advanced", "endorsements": 15, "duration_months": 28},
        {"name": "Information Retrieval", "proficiency": "advanced", "endorsements": 18, "duration_months": 36},
        {"name": "Python", "proficiency": "expert", "endorsements": 40, "duration_months": 84},
    ],
    "certifications": [], "languages": [{"language": "English", "proficiency": "professional"}],
    "redrob_signals": {
        "profile_completeness_score": 90, "signup_date": "2024-01-01",
        "last_active_date": "2026-05-20", "open_to_work_flag": True,
        "profile_views_received_30d": 20, "applications_submitted_30d": 2,
        "recruiter_response_rate": 0.8, "avg_response_time_hours": 12,
        "skill_assessment_scores": {"FAISS": 85}, "connection_count": 300,
        "endorsements_received": 90, "notice_period_days": 15,
        "expected_salary_range_inr_lpa": {"min": 30, "max": 45},
        "preferred_work_mode": "hybrid", "willing_to_relocate": True,
        "github_activity_score": 60, "search_appearance_30d": 200,
        "saved_by_recruiters_30d": 10, "interview_completion_rate": 0.9,
        "offer_acceptance_rate": 0.5, "verified_email": True,
        "verified_phone": True, "linkedin_connected": True,
    },
}


def mut(**changes):
    """Deep-copy BASE and apply nested overrides via dotted keys."""
    c = copy.deepcopy(BASE)
    for dotted, val in changes.items():
        node = c
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = val
    return c


def test_stuffer_below_real():
    """JD: 'title is Marketing Manager -> not a fit', even with perfect skills."""
    stuffer = mut(**{"profile.current_title": "Accountant"})
    stuffer["career_history"] = [dict(BASE["career_history"][0],
                                      title="Accountant", company="KPMG",
                                      industry="Consulting")]
    assert _s(stuffer) < _s(BASE), "keyword-stuffer outranked a real ML engineer"
    print("ok: keyword-stuffer (Accountant + AI skills) ranks below real ML engineer")


def test_inactive_below_active_twin():
    """Behavioral availability: identical profile, only last-active differs."""
    inactive = mut(**{"redrob_signals.last_active_date": "2025-09-01"})
    assert _s(inactive) < _s(BASE), "inactive twin not down-weighted"
    print("ok: inactive twin ranks below active twin")


def test_long_notice_below_short_notice_twin():
    """JD: '30+ day notice ... bar gets higher'."""
    longn = mut(**{"redrob_signals.notice_period_days": 120})
    assert _s(longn) < _s(BASE), "long-notice twin not down-weighted"
    print("ok: long-notice twin ranks below short-notice twin")


def test_unresponsive_below_responsive_twin():
    low = mut(**{"redrob_signals.recruiter_response_rate": 0.05})
    assert _s(low) < _s(BASE), "unresponsive twin not down-weighted"
    print("ok: unresponsive twin ranks below responsive twin")


def test_cv_speech_below_retrieval():
    """JD: CV/speech primary WITHOUT NLP/IR is a do-not-want."""
    cv = mut()
    cv["skills"] = [
        {"name": "Image Classification", "proficiency": "expert", "endorsements": 30, "duration_months": 40},
        {"name": "Object Detection", "proficiency": "advanced", "endorsements": 20, "duration_months": 30},
        {"name": "OpenCV", "proficiency": "advanced", "endorsements": 15, "duration_months": 36},
    ]
    cv["profile"]["current_title"] = "Computer Vision Engineer"
    assert _s(cv) < _s(BASE), "CV/speech specialist not down-weighted vs retrieval"
    print("ok: CV/speech-only specialist ranks below retrieval engineer")


def test_services_only_below_product():
    """JD: entire career at consulting/services (no product) is a do-not-want."""
    svc = mut()
    svc["career_history"] = [
        dict(BASE["career_history"][0], company="Infosys", industry="IT Services"),
        dict(BASE["career_history"][1], company="Wipro", industry="IT Services"),
    ]
    svc["profile"]["current_company"] = "Infosys"
    assert _s(svc) < _s(BASE), "services-only career not down-weighted"
    print("ok: services-only career ranks below product-company twin")


def test_outside_india_no_relocate_below():
    """JD: outside India case-by-case, no visa sponsorship."""
    out = mut(**{"profile.country": "USA", "profile.location": "Austin",
                 "redrob_signals.willing_to_relocate": False})
    assert _s(out) < _s(BASE), "outside-India non-relocator not down-weighted"
    print("ok: outside-India (no relocation) ranks below India-based twin")


def test_honeypot_crushed():
    hp = mut()
    hp["skills"] = [{"name": f"S{i}", "proficiency": "expert",
                     "endorsements": 50, "duration_months": 0} for i in range(10)]
    assert _s(hp) < 0.2 * _s(BASE), "honeypot not crushed"
    print("ok: honeypot crushed far below a real candidate")


def test_experience_band():
    """Ideal ~6-8 yrs; a 1-yr and a 20-yr version should score lower."""
    junior = mut(**{"profile.years_of_experience": 1.0})
    veteran = mut(**{"profile.years_of_experience": 20.0})
    assert _s(junior) < _s(BASE) and _s(veteran) < _s(BASE), "experience band not respected"
    print("ok: 7-yr reference outranks 1-yr and 20-yr versions")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} ranking invariants hold.")
