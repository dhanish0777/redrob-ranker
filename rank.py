#!/usr/bin/env python3
"""
rank.py — produce the top-100 submission CSV.

    python rank.py --candidates ./data/candidates.jsonl.gz --out ./submission.csv

This is the SINGLE reproduce command (spec 10.3). It must run offline, CPU-only,
within 5 minutes and 16 GB. The deterministic scorer here is fast (pure Python
over parsed dicts); the optional semantic layer (Phase 3) loads a PRE-COMPUTED
numpy artifact rather than running a model, so the ranking step stays cheap.

Tie-break handling (matches validate_submission.py exactly): scores are rounded
FIRST, then we sort by (-rounded_score, candidate_id). That guarantees equal
rounded scores always appear in ascending candidate_id order, which the
validator requires; sorting before rounding could otherwise violate it.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from loader import load_candidates, default_candidate_path  # noqa: E402
from score import score_candidate  # noqa: E402
from reason import make_reasoning  # noqa: E402

TOP_N = 100
SCORE_DECIMALS = 6


def rank(candidates_path: str, out_path: str, top_n: int = TOP_N) -> None:
    t0 = time.time()
    cands = load_candidates(candidates_path)
    by_id = {c["candidate_id"]: c for c in cands}

    scored = [score_candidate(c) for c in cands]

    # Round scores BEFORE tie-breaking so the validator's "equal score ->
    # ascending candidate_id" rule always holds.
    for s in scored:
        s["score"] = round(float(s["score"]), SCORE_DECIMALS)

    scored.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    top = scored[:top_n]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, s in enumerate(top, start=1):
            cand = by_id[s["candidate_id"]]
            reasoning = make_reasoning(cand, s).replace("\n", " ").strip()
            w.writerow([s["candidate_id"], i, f"{s['score']:.{SCORE_DECIMALS}f}", reasoning])

    elapsed = time.time() - t0
    honeypots_in_top = sum(1 for s in top if s["is_honeypot"])
    print(f"Wrote {len(top)} ranked candidates -> {out_path}")
    print(f"Ranking step took {elapsed:.2f}s over {len(cands):,} candidates")
    print(f"Honeypots in top {top_n}: {honeypots_in_top} "
          f"({100*honeypots_in_top/max(1,top_n):.1f}%) -- must stay <=10% to avoid DQ")
    print(f"Score range: {top[0]['score']:.4f} (rank 1) .. {top[-1]['score']:.4f} (rank {top_n})")


def main():
    ap = argparse.ArgumentParser(description="Rank top-100 candidates for the JD.")
    ap.add_argument("--candidates", default=default_candidate_path(),
                    help="Path to candidates.jsonl(.gz) or sample json")
    ap.add_argument("--out", default="submission.csv", help="Output CSV path")
    ap.add_argument("--top", type=int, default=TOP_N)
    args = ap.parse_args()
    rank(args.candidates, args.out, args.top)


if __name__ == "__main__":
    main()
