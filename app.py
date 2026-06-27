"""
app.py — hosted sandbox demo for the candidate ranker.
"""

import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from loader import iter_candidates  # noqa: E402
from score import score_candidate  # noqa: E402
from reason import make_reasoning  # noqa: E402

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")

st.title("Redrob — Intelligent Candidate Ranker")
st.caption(
    "Ranks candidates against the Senior AI Engineer JD with explainable scoring. "
    "This demo runs the deterministic core on a small sample; the full pipeline "
    "adds an offline-precomputed semantic re-ranking pass (see README)."
)

SAMPLE = os.path.join(os.path.dirname(__file__), "data", "sample_candidates.json")


def load_records(upload):
    if upload is not None:
        raw = upload.read().decode("utf-8")
        try:
            data = json.loads(raw)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
    if os.path.exists(SAMPLE):
        with open(SAMPLE, encoding="utf-8") as f:
            return json.load(f)
    return []


with st.sidebar:
    st.header("Input")
    st.write("Upload a small candidates file (JSON array or JSONL, <=100), "
             "or use the bundled 50-candidate sample.")
    upload = st.file_uploader("Candidates file", type=["json", "jsonl"])
    top_n = st.slider("Show top N", 5, 100, 20)

records = load_records(upload)[:100]

if not records:
    st.warning("No candidates loaded. Upload a file or add data/sample_candidates.json.")
    st.stop()

scored = [score_candidate(c) for c in records]
by_id = {c["candidate_id"]: c for c in records}
for s in scored:
    s["score"] = round(float(s["score"]), 6)
scored.sort(key=lambda r: (-r["score"], r["candidate_id"]))

st.subheader(f"Ranked candidates ({len(records)} scored, showing top {min(top_n, len(scored))})")

rows = []
for i, s in enumerate(scored[:top_n], 1):
    c = by_id[s["candidate_id"]]
    comp = s["components"]
    rows.append({
        "rank": i,
        "candidate_id": s["candidate_id"],
        "title": c["profile"].get("current_title", ""),
        "score": s["score"],
        "role": round(comp["role"], 2),
        "skill": round(comp["skill"], 2),
        "exp": round(comp["exp"], 2),
        "behav": round(s["behavioral_modifier"], 2),
        "honeypot": "YES" if s["is_honeypot"] else "",
        "reasoning": make_reasoning(c, s),
    })

st.dataframe(rows, use_container_width=True, hide_index=True)

honey = sum(1 for s in scored if s["is_honeypot"])
st.metric("Honeypots detected (forced to bottom)", honey)

with st.expander("How a candidate is scored"):
    st.markdown(
        "- **role** — title + career-trajectory relevance (the anti-keyword-stuffer gate)\n"
        "- **skill** — JD-skill match, *trust-weighted* by endorsements / duration / assessment\n"
        "- **exp** — fit to the JD's 5–9yr (ideal 6–8) band + product-company experience\n"
        "- **behav** — multiplicative availability modifier (open-to-work, recency, "
        "recruiter response, notice period)\n"
        "- plus hard penalties for the JD's explicit do-not-wants, and a honeypot guard."
    )
