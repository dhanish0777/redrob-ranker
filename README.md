# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

Ranks the top-100 candidates from `candidates.jsonl(.gz)` against the
"Senior AI Engineer — Founding Team" job description, with explainable reasoning.

## Why this is not a pure-embedding system
The dataset is built so keyword/embedding-only approaches fail: keyword-stuffers
and ~80 impossible "honeypot" profiles would rank high. We use a **hybrid,
mostly-deterministic scorer** that reads profiles structurally (title & career
trajectory dominate), trust-weights skills, applies a behavioral-availability
modifier, hard-penalizes the JD's explicit disqualifiers, and uses a small
local embedding model only as a "plain-language fit" catcher.

## Layout
- `src/loader.py`   — load the pool (.json / .jsonl / .jsonl.gz), stdlib only
- `src/honeypots.py`— high-precision profile-consistency / honeypot detector
- `src/eda.py`      — characterize the full pool; sanity-check honeypot rate
- `tests/`          — precision/recall tests for the detector
- (coming) `src/features.py`, `src/score.py`, `src/reason.py`, `rank.py`

## Reproduce

Optional one-time OFFLINE precompute of semantic embeddings (improves top-band
ordering; ranking still works without it):
```
pip install sentence-transformers
python src/precompute_embeddings.py --candidates ./data/candidates.jsonl
```

Ranking step (the single reproduce command; offline, CPU-only, <5 min):
```
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv
```
If the embedding artifacts are absent, rank.py automatically runs the pure
deterministic system (still valid + honeypot-clean).

## Status
- [x] Phase 0–1: loader, honeypot detector (0 FP on sample, catches synthetic), EDA
- [x] Phase 2: deterministic core scorer (features, score, reason, rank.py) -> valid CSV
- [x] Phase 3: invariant harness, notice-period factor, semantic re-ranking (precompute + optional blend)
- [ ] Phase 4: reasoning polish | Phase 5: sandbox + repro hardening
- [ ] Phase 4: reasoning generation
- [ ] Phase 5: reproducibility + sandbox
