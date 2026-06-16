"""
loader.py
=========
Loads the Redrob candidate pool from any of the formats we encounter:

  * candidates.jsonl.gz   (the real 100K pool, gzipped JSON-lines)
  * candidates.jsonl      (un-gzipped JSON-lines)
  * sample_candidates.json (the 50-candidate pretty-printed JSON array)

We keep this dead simple and dependency-free (stdlib only) so the *ranking*
step has the smallest possible import surface -- that matters for the 5-minute
CPU budget and for reproducibility inside the Stage-3 sandbox.

Design notes you can defend in the interview:
  - We stream line-by-line for .jsonl(.gz) instead of json.load-ing the whole
    file, so peak memory stays well under the 16 GB cap even on the 465 MB
    uncompressed pool. Each parsed dict is ~a few KB; 100K of them is roughly
    a few hundred MB held in a list, which is fine.
  - We never mutate the raw records here. Feature extraction lives elsewhere
    (features.py) so the "what the data says" layer and the "what we infer"
    layer stay separate and testable.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Iterator


def _open_text(path: str):
    """Open a path as a UTF-8 text stream, transparently handling gzip."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path: str) -> Iterator[dict]:
    """
    Yield candidate dicts one at a time.

    For .json (the sample) we load the whole array. For .jsonl/.jsonl.gz we
    stream line by line -- this is the memory-safe path for the full pool.
    """
    if path.endswith(".json"):
        with _open_text(path) as f:
            data = json.load(f)
        # The sample is a list; tolerate a single-object file too.
        if isinstance(data, dict):
            data = [data]
        for rec in data:
            yield rec
        return

    # .jsonl or .jsonl.gz
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_candidates(path: str) -> list[dict]:
    """Load all candidates into a list. Convenience wrapper over iter_candidates."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Candidate file not found: {path}\n"
            "Place candidates.jsonl.gz under ./data/ or pass --candidates."
        )
    return list(iter_candidates(path))


def default_candidate_path() -> str:
    """
    Best-effort guess of where the candidate file lives, in priority order.
    Lets scripts run with zero args during development.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(here, "data", "candidates.jsonl.gz"),
        os.path.join(here, "data", "candidates.jsonl"),
        os.path.join(here, "data", "sample_candidates.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Fall back to the sample path even if missing, so the error message is clear.
    return candidates[-1]


if __name__ == "__main__":
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else default_candidate_path()
    cands = load_candidates(p)
    print(f"Loaded {len(cands):,} candidates from {p}")
    print("First candidate id:", cands[0]["candidate_id"])
    print("Top-level keys:", list(cands[0].keys()))
