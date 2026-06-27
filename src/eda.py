"""
eda.py
======
Exploratory data analysis script.
"""

from __future__ import annotations

import sys
from collections import Counter

from loader import load_candidates, default_candidate_path
from honeypots import consistency_report


def _bucket(x, edges):
    for e in edges:
        if x <= e:
            return f"<= {e}"
    return f"> {edges[-1]}"


def main(path: str):
    cands = load_candidates(path)
    n = len(cands)
    print(f"\n{'='*60}\nPOOL OVERVIEW  ({n:,} candidates from {path})\n{'='*60}")

    titles = Counter()
    industries = Counter()
    countries = Counter()
    skill_freq = Counter()
    yoe_buckets = Counter()
    open_to_work = 0
    has_github = 0
    flagged = 0
    flag_breakdown = Counter()

    for c in cands:
        p = c["profile"]
        titles[p["current_title"]] += 1
        industries[p["current_industry"]] += 1
        countries[p["country"]] += 1
        yoe_buckets[_bucket(p["years_of_experience"], [2, 5, 9, 12, 15])] += 1
        for s in c.get("skills", []):
            skill_freq[s["name"]] += 1

        sig = c.get("redrob_signals", {})
        if sig.get("open_to_work_flag"):
            open_to_work += 1
        if (sig.get("github_activity_score", -1) or -1) >= 0:
            has_github += 1

        rep = consistency_report(c)
        if rep["is_honeypot"]:
            flagged += 1
            for f in rep["flags"]:
                flag_breakdown[f] += 1

    def show(title, counter, k=15):
        print(f"\n{title}:")
        for name, cnt in counter.most_common(k):
            print(f"  {cnt:6,}  {name}")

    show("Top current titles", titles, 25)
    show("Industries", industries)
    show("Countries", countries, 10)
    show("Most common listed skills", skill_freq, 30)

    print("\nYears-of-experience buckets:")
    for b in ["<= 2", "<= 5", "<= 9", "<= 12", "<= 15", "> 15"]:
        if b in yoe_buckets:
            print(f"  {yoe_buckets[b]:6,}  {b}")

    print(f"\nopen_to_work_flag = true : {open_to_work:,} ({100*open_to_work/n:.1f}%)")
    print(f"has linked GitHub        : {has_github:,} ({100*has_github/n:.1f}%)")

    print(f"\n{'='*60}\nHONEYPOT DETECTOR vs REALITY\n{'='*60}")
    print(f"flagged as honeypot : {flagged:,} ({100*flagged/n:.3f}%)")
    print("(spec says ~80 true honeypots = ~0.08%; some extra flags are fine "
          "as long as we stay far under the 10% top-100 DQ bar)")
    print("\nflag breakdown:")
    for f, cnt in flag_breakdown.most_common():
        print(f"  {cnt:6,}  {f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else default_candidate_path())
