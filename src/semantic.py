"""
semantic.py  (ranking-time -- numpy ONLY, no model, no network)
================================================================
Loads the precomputed embedding artifacts and exposes the JD-similarity for
each candidate as a PERCENTILE in [0,1] (robust to MiniLM's absolute cosine
scale, and directly comparable to our other [0,1] components).

If the artifacts are absent, this degrades gracefully: is_available() returns
False and the ranker runs as the pure deterministic system. That keeps the
baseline reproducible even on a machine where the precompute wasn't run.
"""

from __future__ import annotations

import json
import os

ARTIFACT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"
)


class SemanticIndex:
    def __init__(self, artifact_dir: str = ARTIFACT_DIR):
        self.available = False
        self._pct = {}  # candidate_id -> JD-similarity percentile in [0,1]
        emb_path = os.path.join(artifact_dir, "candidate_embeddings.npy")
        jd_path = os.path.join(artifact_dir, "jd_embedding.npy")
        ids_path = os.path.join(artifact_dir, "candidate_ids.json")
        if not all(os.path.exists(p) for p in (emb_path, jd_path, ids_path)):
            return
        try:
            import numpy as np
            emb = np.load(emb_path)            # (N, D), L2-normalized
            jd = np.load(jd_path)              # (D,), L2-normalized
            with open(ids_path) as f:
                ids = json.load(f)
            sims = emb @ jd                     # cosine, since both normalized
            # Convert to percentile rank in [0,1] (ties share average rank).
            order = sims.argsort()
            ranks = np.empty_like(order, dtype="float64")
            ranks[order] = np.arange(len(sims))
            pct = ranks / max(1, len(sims) - 1)
            self._pct = {cid: float(pct[i]) for i, cid in enumerate(ids)}
            self.available = True
        except Exception as e:  # pragma: no cover - defensive
            print(f"[semantic] disabled (load error: {e})")
            self.available = False

    def is_available(self) -> bool:
        return self.available

    def percentile(self, candidate_id: str) -> float:
        """JD-similarity percentile; neutral 0.5 for unseen ids."""
        return self._pct.get(candidate_id, 0.5)


def blend(det_score: float, sem_pct: float, alpha: float = 0.15) -> float:
    """
    Multiplicative nudge: a candidate at the top semantic percentile gets up to
    +alpha, at the bottom -alpha. det_score stays the backbone, so semantics
    re-orders the tightly-packed top band and lifts plain-language fits WITHOUT
    letting a high-similarity stuffer override the role gate.

        sem_pct=1.0 -> x(1+alpha);  sem_pct=0.5 -> x1.0;  sem_pct=0.0 -> x(1-alpha)
    """
    return det_score * (1.0 + alpha * (2.0 * sem_pct - 1.0))
