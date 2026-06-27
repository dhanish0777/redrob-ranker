"""
loader.py
=========
Loads the candidate pool.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Iterator


def _open_text(path: str):
    """Open a path as a UTF-8 text stream."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path: str) -> Iterator[dict]:
    """Yield candidate dicts one at a time."""
    if path.endswith(".json"):
        with _open_text(path) as f:
            data = json.load(f)
        # Tolerate a single-object file.
        if isinstance(data, dict):
            data = [data]
        for rec in data:
            yield rec
        return

    # Stream line by line
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_candidates(path: str) -> list[dict]:
    """Load all candidates into a list."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Candidate file not found: {path}\n"
            "Place candidates.jsonl.gz under ./data/ or pass --candidates."
        )
    return list(iter_candidates(path))


def default_candidate_path() -> str:
    """Find default candidate file path."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(here, "data", "candidates.jsonl.gz"),
        os.path.join(here, "data", "candidates.jsonl"),
        os.path.join(here, "data", "sample_candidates.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Fallback to sample path.
    return candidates[-1]


if __name__ == "__main__":
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else default_candidate_path()
    cands = load_candidates(p)
    print(f"Loaded {len(cands):,} candidates from {p}")
    print("First candidate id:", cands[0]["candidate_id"])
    print("Top-level keys:", list(cands[0].keys()))
