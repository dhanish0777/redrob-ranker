"""
features.py
===========
Extracts features from raw candidate data.
"""

from __future__ import annotations

import re
from datetime import date

REFERENCE_DATE = date(2026, 6, 1)

# Role taxonomy.
ROLE_TIERS = {
    # Target archetype.
    1.00: ["machine learning engineer", "ml engineer", "ai engineer",
           "applied ml", "applied scientist", "applied ai",
           "recommendation systems", "recommender", "search engineer",
           "nlp engineer", "information retrieval", "ranking",
           "ml scientist", "data scientist",
           # ML-qualified variants.
           "machine learning", "(ml)"],
    # Research-flavored titles.
    0.85: ["research engineer", "research scientist"],
    # Strong adjacent roles.
    0.65: ["data engineer", "backend engineer", "software engineer",
           "platform engineer", "full stack", "fullstack", "analytics engineer"],
    # Weak technical roles.
    0.30: ["frontend", "front end", "mobile developer", "devops",
           "site reliability", "sre", "qa engineer", "test engineer",
           "cloud engineer", ".net developer", "java developer"],
    # Non-technical roles.
    0.00: ["marketing manager", "operations manager", "accountant",
           "hr manager", "human resources", "business analyst",
           "project manager", "program manager", "customer support",
           "mechanical engineer", "civil engineer", "graphic designer",
           "sales", "product manager", "consultant", "recruiter"],
}


def _phrase_matches(phrase: str, padded_title: str) -> bool:
    """Word-boundary-aware phrase match."""
    if any(ch in phrase for ch in "()/.-"):
        return phrase in padded_title
    return f" {phrase} " in padded_title


def role_relevance(title: str) -> float:
    """Calculate role relevance [0,1]."""
    padded = f" {(title or '').lower()} "
    best = None
    for score, phrases in ROLE_TIERS.items():
        if any(_phrase_matches(p, padded) for p in phrases):
            best = score if best is None else max(best, score)
    if best is None:
        best = 0.25
    # Junior title demotion.
    if _phrase_matches("junior", padded):
        best *= 0.90
    return best


# Skill taxonomy.
SKILL_WEIGHTS = {
    # Core skills.
    "embeddings": 1.0, "sentence transformers": 1.0, "sentence-transformers": 1.0,
    "bge": 1.0, "e5": 1.0, "faiss": 1.0, "pinecone": 1.0, "qdrant": 1.0,
    "weaviate": 1.0, "milvus": 1.0, "opensearch": 1.0, "elasticsearch": 1.0,
    "information retrieval": 1.0, "semantic search": 1.0, "vector search": 1.0,
    "bm25": 0.9, "hugging face transformers": 0.9, "transformers": 0.7,
    "learning to rank": 1.0, "ranking": 0.9, "retrieval": 0.9,
    "ndcg": 1.0, "mrr": 1.0, "map": 0.6, "a/b testing": 0.8,
    # Strong ML skills.
    "machine learning": 0.7, "nlp": 0.7, "scikit-learn": 0.6, "sklearn": 0.6,
    "xgboost": 0.7, "lightgbm": 0.7, "pytorch": 0.6, "tensorflow": 0.5,
    "mlops": 0.6, "mlflow": 0.6, "feature engineering": 0.6, "kubeflow": 0.5,
    "python": 0.5, "spark": 0.3, "airflow": 0.3, "bentoml": 0.4,
    # Nice-to-have skills.
    "lora": 0.4, "qlora": 0.4, "peft": 0.4, "fine-tuning llms": 0.4,
    "fine-tuning": 0.35,
    # Ambiguous skills.
    "prompt engineering": 0.15, "langchain": 0.1, "llamaindex": 0.1,
    # Negative skills.
    "image classification": -0.3, "speech recognition": -0.3, "tts": -0.3,
    "gans": -0.2, "object detection": -0.3, "opencv": -0.3,
    "computer vision": -0.3, "cnn": -0.15, "image segmentation": -0.3,
}

# CV/speech skills.
CV_SPEECH_SKILLS = {
    "image classification", "speech recognition", "tts", "gans",
    "object detection", "opencv", "computer vision", "image segmentation",
    "cnn", "face recognition", "pose estimation",
}
# NLP/IR skills.
NLP_IR_SKILLS = {
    "nlp", "information retrieval", "embeddings", "sentence transformers",
    "bm25", "retrieval", "semantic search", "hugging face transformers",
    "transformers", "ranking", "learning to rank",
}

# Services companies list.
SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "mindtree", "ltimindtree", "mphasis",
    "hexaware", "dxc", "ibm services", "larsen", "l&t infotech",
}


def is_services_company(company: str, industry: str) -> bool:
    c = (company or "").lower()
    if any(s in c for s in SERVICES_COMPANIES):
        return True
    # Check industry.
    return (industry or "").strip().lower() == "it services"


# Location lists.
PREFERRED_CITIES = ["pune", "noida"]
WELCOME_CITIES = ["hyderabad", "mumbai", "delhi", "ncr", "gurgaon", "gurugram",
                  "bangalore", "bengaluru", "chennai"]


def location_score(profile: dict, willing_to_relocate: bool) -> float:
    loc = (profile.get("location", "") or "").lower()
    country = (profile.get("country", "") or "").lower()
    if any(c in loc for c in PREFERRED_CITIES):
        return 1.0
    if "india" in country:
        if any(c in loc for c in WELCOME_CITIES):
            return 0.85
        return 0.7 if willing_to_relocate else 0.55
    # Check outside India.
    return 0.45 if willing_to_relocate else 0.2


# Helpers
def months_inactive(last_active: str) -> float:
    try:
        y, m, d = (int(x) for x in last_active.split("-"))
        la = date(y, m, d)
    except (ValueError, AttributeError):
        return 999.0
    return max(0.0, (REFERENCE_DATE.year - la.year) * 12 + (REFERENCE_DATE.month - la.month))


def short_tenure_fraction(history: list[dict]) -> float:
    """Calculate short tenure fraction."""
    completed = [r for r in history if not r.get("is_current")]
    if not completed:
        return 0.0
    short = sum(1 for r in completed if int(r.get("duration_months", 0) or 0) < 18)
    return short / len(completed)


def has_product_experience(history: list[dict]) -> bool:
    return any(
        not is_services_company(r.get("company", ""), r.get("industry", ""))
        for r in history
    )


def tokenize_skills(skills: list[dict]) -> list[str]:
    return [(s.get("name", "") or "").strip().lower() for s in skills]
