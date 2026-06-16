"""
features.py
===========
Turns a raw candidate dict into the signals the JD actually cares about.

Everything here is traceable to a specific line in job_description.md. The
taxonomies are constants at the top so they're easy to read, defend, and tune.

The JD's structure gives us a rubric:
  - "Things you absolutely need": embeddings retrieval, vector DB / hybrid
    search, strong Python, ranking-evaluation frameworks (NDCG/MRR/MAP).
  - "Things we'd like": LoRA/QLoRA/PEFT, learning-to-rank, HR-tech, distsys,
    open source.
  - "Things we explicitly do NOT want": consulting-only careers, CV/speech/
    robotics primary without NLP/IR, research-only, recent LangChain wrappers,
    title-chasers.
  - Behavioral: availability matters (open_to_work, recency, response rate).

A KEY DATA OBSERVATION (verified, defensible): career_history *descriptions*
in this dataset are partially scrambled relative to the role's title -- e.g.
a "Marketing Manager" entry can carry a mechanical-engineering paragraph. So
we trust, in order: (1) titles, (2) the self-written summary, (3) skills
(trust-weighted), and we let the Phase-3 semantic layer mine descriptions for
plain-language evidence wherever it genuinely appears. Titles + summary are
internally coherent per candidate; raw description keywords are not.
"""

from __future__ import annotations

import re
from datetime import date

REFERENCE_DATE = date(2026, 6, 1)

# ---------------------------------------------------------------------------
# 1. ROLE / TITLE taxonomy  (the decisive anti-keyword-stuffer signal)
#    The JD wants applied-ML / retrieval / ranking people. A title is matched
#    by checking whether any of these phrases appear in it (lowercased).
# ---------------------------------------------------------------------------
ROLE_TIERS = {
    # Bullseye: exactly the JD's target archetype.
    1.00: ["machine learning engineer", "ml engineer", "ai engineer",
           "applied ml", "applied scientist", "applied ai",
           "recommendation systems", "recommender", "search engineer",
           "nlp engineer", "information retrieval", "ranking",
           "research engineer", "ml scientist", "data scientist"],
    # Strong adjacent: technical builders who plausibly have retrieval/ranking
    # exposure and match the "shipper" attitude. JD's v2 audit started from
    # "BM25 + rule-based", so data/backend folks who built search are in scope.
    0.65: ["data engineer", "backend engineer", "software engineer",
           "platform engineer", "full stack", "fullstack", "analytics engineer"],
    # Weak technical: real engineers but far from NLP/IR; would be re-learning.
    0.30: ["frontend", "front end", "mobile developer", "devops",
           "site reliability", "sre", "qa engineer", "test engineer",
           "cloud engineer", ".net developer", "java developer"],
    # Non-technical / wrong domain: the keyword-stuffer carriers. A perfect
    # skill list cannot lift these (JD: "title is Marketing Manager -> not a fit").
    0.00: ["marketing manager", "operations manager", "accountant",
           "hr manager", "human resources", "business analyst",
           "project manager", "program manager", "customer support",
           "mechanical engineer", "civil engineer", "graphic designer",
           "sales", "product manager", "consultant", "recruiter"],
}


def role_relevance(title: str) -> float:
    """Highest matching tier for a title; default 0.25 for unknown technical-ish."""
    t = (title or "").lower()
    best = None
    for score, phrases in ROLE_TIERS.items():
        if any(p in t for p in phrases):
            best = score if best is None else max(best, score)
    return best if best is not None else 0.25


# ---------------------------------------------------------------------------
# 2. SKILL taxonomy  (matched case-insensitively, substring-aware)
#    Weights reflect the JD's own "absolutely need / like / do not want" split.
# ---------------------------------------------------------------------------
SKILL_WEIGHTS = {
    # CORE: embeddings retrieval + vector/hybrid search + IR/ranking + eval.
    "embeddings": 1.0, "sentence transformers": 1.0, "sentence-transformers": 1.0,
    "bge": 1.0, "e5": 1.0, "faiss": 1.0, "pinecone": 1.0, "qdrant": 1.0,
    "weaviate": 1.0, "milvus": 1.0, "opensearch": 1.0, "elasticsearch": 1.0,
    "information retrieval": 1.0, "semantic search": 1.0, "vector search": 1.0,
    "bm25": 0.9, "hugging face transformers": 0.9, "transformers": 0.7,
    "learning to rank": 1.0, "ranking": 0.9, "retrieval": 0.9,
    "ndcg": 1.0, "mrr": 1.0, "map": 0.6, "a/b testing": 0.8,
    # STRONG ML: the general applied-ML stack the JD assumes underneath.
    "machine learning": 0.7, "nlp": 0.7, "scikit-learn": 0.6, "sklearn": 0.6,
    "xgboost": 0.7, "lightgbm": 0.7, "pytorch": 0.6, "tensorflow": 0.5,
    "mlops": 0.6, "mlflow": 0.6, "feature engineering": 0.6, "kubeflow": 0.5,
    "python": 0.5, "spark": 0.3, "airflow": 0.3, "bentoml": 0.4,
    # NICE-TO-HAVE (JD "would like"): small positive.
    "lora": 0.4, "qlora": 0.4, "peft": 0.4, "fine-tuning llms": 0.4,
    "fine-tuning": 0.35,
    # AMBIGUOUS: prompt/LangChain are positive only in small dose; heavy
    # reliance is a NEGATIVE per the JD ("recent LangChain wrappers"). Handled
    # in score.py as a penalty when they're the *primary* AI signal.
    "prompt engineering": 0.15, "langchain": 0.1, "llamaindex": 0.1,
    # NEGATIVE for THIS role: CV/speech/robotics primary (JD explicit).
    "image classification": -0.3, "speech recognition": -0.3, "tts": -0.3,
    "gans": -0.2, "object detection": -0.3, "opencv": -0.3,
    "computer vision": -0.3, "cnn": -0.15, "image segmentation": -0.3,
}

# Skills that mark CV/speech/robotics specialization (for the explicit penalty).
CV_SPEECH_SKILLS = {
    "image classification", "speech recognition", "tts", "gans",
    "object detection", "opencv", "computer vision", "image segmentation",
    "cnn", "face recognition", "pose estimation",
}
# Skills that count as NLP/IR exposure (cancels the CV/speech penalty per JD:
# "...without significant NLP/IR exposure").
NLP_IR_SKILLS = {
    "nlp", "information retrieval", "embeddings", "sentence transformers",
    "bm25", "retrieval", "semantic search", "hugging face transformers",
    "transformers", "ranking", "learning to rank",
}

# ---------------------------------------------------------------------------
# 3. COMPANY classification: services/consulting vs product.
#    JD: "People who have only worked at consulting firms ... in their entire
#    career" is a do-not-want, BUT current consulting is fine with prior
#    product experience. So we need 'is there ANY product role in history'.
# ---------------------------------------------------------------------------
SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "mindtree", "ltimindtree", "mphasis",
    "hexaware", "dxc", "ibm services", "larsen", "l&t infotech",
}


def is_services_company(company: str, industry: str) -> bool:
    c = (company or "").lower()
    if any(s in c for s in SERVICES_COMPANIES):
        return True
    # Industry "IT Services" is a softer services signal.
    return (industry or "").strip().lower() == "it services"


# ---------------------------------------------------------------------------
# 4. LOCATION: JD prefers Pune/Noida; Hyderabad/Mumbai/Delhi NCR welcome;
#    outside India case-by-case with no visa sponsorship.
# ---------------------------------------------------------------------------
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
    # Outside India: no visa sponsorship; only viable if willing to relocate.
    return 0.45 if willing_to_relocate else 0.2


# ---------------------------------------------------------------------------
# Helpers used by score.py
# ---------------------------------------------------------------------------
def months_inactive(last_active: str) -> float:
    try:
        y, m, d = (int(x) for x in last_active.split("-"))
        la = date(y, m, d)
    except (ValueError, AttributeError):
        return 999.0
    return max(0.0, (REFERENCE_DATE.year - la.year) * 12 + (REFERENCE_DATE.month - la.month))


def short_tenure_fraction(history: list[dict]) -> float:
    """Fraction of *completed* roles shorter than 18 months -> job-hopping tell."""
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
