"""
semantic.py
===========
Ranking-time semantic re-ranker.
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
        self._pct = {}
        emb_path = os.path.join(artifact_dir, "candidate_embeddings.npy")
        jd_path = os.path.join(artifact_dir, "jd_embedding.npy")
        ids_path = os.path.join(artifact_dir, "candidate_ids.json")
        if not all(os.path.exists(p) for p in (emb_path, jd_path, ids_path)):
            return
        try:
            import numpy as np
            emb = np.load(emb_path)
            jd = np.load(jd_path)
            with open(ids_path) as f:
                ids = json.load(f)
            sims = emb @ jd
            # Convert to percentile rank in [0,1].
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
        return self._pct.get(candidate_id, 0.5)


def blend(det_score: float, sem_pct: float, alpha: float = 0.15) -> float:
    """Apply semantic blend."""
    return det_score * (1.0 + alpha * (2.0 * sem_pct - 1.0))
