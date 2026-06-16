"""
Validate the honeypot detector two ways:
  1. PRECISION: it must not flag any of the 50 real sample candidates.
  2. RECALL: it must catch synthetic honeypots built from the spec's examples.
Run:  python tests/test_honeypots.py
"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from loader import load_candidates  # noqa: E402
from honeypots import consistency_report  # noqa: E402

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "data", "sample_candidates.json")


def test_precision_no_false_positives():
    cands = load_candidates(SAMPLE)
    flagged = [c["candidate_id"] for c in cands if consistency_report(c)["is_honeypot"]]
    assert not flagged, f"False positives on real candidates: {flagged}"
    print(f"PRECISION ok: 0/{len(cands)} real candidates flagged as honeypot")


def _make_honeypot_zero_skills(base):
    """Spec example: 'expert proficiency in 10 skills with 0 years used'."""
    c = copy.deepcopy(base)
    c["candidate_id"] = "CAND_9000001"
    c["skills"] = [
        {"name": f"Skill{i}", "proficiency": "expert", "endorsements": 99,
         "duration_months": 0}
        for i in range(10)
    ]
    return c


def _make_honeypot_yoe_mismatch(base):
    """Spec example: '8 years of experience at a company founded 3 years ago'."""
    c = copy.deepcopy(base)
    c["candidate_id"] = "CAND_9000002"
    c["profile"]["years_of_experience"] = 8.0
    c["career_history"] = [{
        "company": "NewCo", "title": "Engineer",
        "start_date": "2023-06-01", "end_date": None, "duration_months": 96,
        "is_current": True, "industry": "Software", "company_size": "51-200",
        "description": "Built things.",
    }]
    return c


def _make_honeypot_bad_dates(base):
    """End date before start date + current role carrying an end date."""
    c = copy.deepcopy(base)
    c["candidate_id"] = "CAND_9000003"
    c["career_history"] = [{
        "company": "X", "title": "Engineer",
        "start_date": "2022-01-01", "end_date": "2020-01-01", "duration_months": 12,
        "is_current": True, "industry": "Software", "company_size": "51-200",
        "description": "Time traveller.",
    }]
    return c


def test_recall_catches_synthetic():
    base = load_candidates(SAMPLE)[0]
    cases = {
        "zero-duration experts": _make_honeypot_zero_skills(base),
        "yoe>>career span": _make_honeypot_yoe_mismatch(base),
        "impossible dates": _make_honeypot_bad_dates(base),
    }
    for name, hp in cases.items():
        rep = consistency_report(hp)
        assert rep["is_honeypot"], f"MISSED honeypot ({name}): {rep}"
        print(f"RECALL ok: caught '{name}' -> flags={rep['flags']}")


if __name__ == "__main__":
    test_precision_no_false_positives()
    test_recall_catches_synthetic()
    print("\nAll honeypot detector tests passed.")
