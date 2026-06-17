# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

Ranks the top-100 candidates from `candidates.jsonl` against the
"Senior AI Engineer — Founding Team" job description, with explainable reasoning.

## Why this is not a pure-embedding system

The dataset is built so keyword/embedding-only approaches fail: keyword-stuffers
(non-technical titles with every AI skill listed) and ~80 impossible "honeypot"
profiles would rank high. We use a **hybrid, mostly-deterministic scorer** that
reads profiles structurally — title & career trajectory dominate, skills are
*trust-weighted* by endorsements/duration/assessment, a behavioral modifier
captures availability, hard penalties encode the JD's explicit do-not-wants, and
a high-precision consistency check forces honeypots to the bottom. A small
**local** embedding pass (precomputed offline) sharpens top-band ordering.

## Reproduce the submission

**One-time offline precompute** (builds semantic embeddings; may exceed 5 min —
that's allowed, it is not the ranking step):

```bash
pip install -r requirements.txt
pip install sentence-transformers          # precompute-only dependency
python src/precompute_embeddings.py --candidates ./data/candidates.jsonl
```

**Ranking step** — the single reproduce command (offline, CPU-only, ~25s, <16GB):

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv
```

If the embedding artifacts are absent, `rank.py` automatically runs the pure
deterministic system (still valid, still honeypot-clean) — so the submission is
reproducible even without the precompute.

## Hosted sandbox (demo)

A Streamlit app runs the ranker on a small sample and shows the per-component
explanation. Run locally with `streamlit run app.py`, or deploy free:

- **HuggingFace Spaces:** create a Space (SDK: *Streamlit*), push this repo; it
  auto-installs `requirements.txt` and runs `app.py`.
- **Streamlit Cloud:** point it at this repo, main file `app.py`.

## Project layout

```
rank.py                      single reproduce command -> submission.csv
app.py                       hosted sandbox demo (Streamlit)
src/loader.py                stdlib-only reader for .json/.jsonl/.jsonl.gz
src/features.py              JD-derived taxonomies + feature extraction
src/score.py                 5-component scorer + behavioral modifier + penalties
src/reason.py                fact-grounded, varied, rank-consistent reasoning
src/honeypots.py             high-precision profile-consistency / honeypot detector
src/precompute_embeddings.py OFFLINE: build MiniLM embedding artifacts
src/semantic.py              ranking-time semantic re-rank (numpy only, no model)
src/eda.py                   pool characterization
tests/                       honeypot precision/recall + 10 JD ranking invariants
submission_metadata.yaml     portal metadata mirror
```

## How we validated without a leaderboard

No public leaderboard + a 3-submission cap means we validate by methodology, not
by submitting variants. `tests/test_invariants.py` encodes 10 ranking rules taken
directly from the JD (keyword-stuffer < real ML engineer; inactive < active twin;
long-notice < short-notice; services-only < product; honeypot crushed; etc.) and
runs as a regression suite on every change.

```bash
python tests/test_honeypots.py
python tests/test_invariants.py
```

## Compute profile

- Ranking step: ~25s wall-clock over 100,000 candidates, CPU-only, no network.
- Peak memory well under 16 GB (embeddings artifact ~154 MB + parsed records).
- Honeypots in top 100: 0.
