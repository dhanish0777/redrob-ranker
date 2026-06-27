#!/usr/bin/env python3
"""
precompute_embeddings.py
========================
Precomputes embeddings for candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import load_candidates, default_candidate_path  # noqa: E402

ARTIFACT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"
)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Curated JD query.
JD_QUERY = (
    "Senior AI engineer at a product company who owns the ranking, retrieval "
    "and matching intelligence layer. Production experience with embeddings-"
    "based retrieval (sentence-transformers, BGE, E5) deployed to real users; "
    "vector databases and hybrid search (Pinecone, Weaviate, Qdrant, Milvus, "
    "FAISS, Elasticsearch, OpenSearch); strong Python. Designs evaluation "
    "frameworks for ranking systems: NDCG, MRR, MAP, A/B testing, offline-to-"
    "online correlation. Has shipped an end-to-end search, ranking or "
    "recommendation system to real users at meaningful scale. Applied ML "
    "engineer who ships, not a pure researcher and not consulting-only."
)


def candidate_text(c: dict) -> str:
    p = c.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", ""),
             p.get("current_title", "")]
    for r in c.get("career_history", []) or []:
        parts.append(r.get("title", ""))
        parts.append(r.get("description", ""))
    parts.extend(s.get("name", "") for s in c.get("skills", []) or [])
    return " ".join(x for x in parts if x).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=default_candidate_path())
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    # Lazy imports.
    import numpy as np
    from sentence_transformers import SentenceTransformer

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print(f"Loading model {MODEL_NAME} (CPU)...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    cands = load_candidates(args.candidates)
    ids = [c["candidate_id"] for c in cands]
    texts = [candidate_text(c) for c in cands]
    print(f"Encoding {len(texts):,} candidate documents (batch={args.batch})...")
    emb = model.encode(texts, batch_size=args.batch, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    emb = emb.astype("float32")

    jd = model.encode([JD_QUERY], convert_to_numpy=True,
                      normalize_embeddings=True)[0].astype("float32")

    np.save(os.path.join(ARTIFACT_DIR, "candidate_embeddings.npy"), emb)
    np.save(os.path.join(ARTIFACT_DIR, "jd_embedding.npy"), jd)
    with open(os.path.join(ARTIFACT_DIR, "candidate_ids.json"), "w") as f:
        json.dump(ids, f)
    with open(os.path.join(ARTIFACT_DIR, "embed_meta.json"), "w") as f:
        json.dump({"model": MODEL_NAME, "jd_query": JD_QUERY,
                   "n": len(ids), "dim": int(emb.shape[1])}, f, indent=2)

    sims = emb @ jd
    print(f"\nSaved artifacts to {ARTIFACT_DIR}")
    print(f"  embeddings: {emb.shape}, ~{emb.nbytes/1e6:.0f} MB")
    print(f"  JD cosine similarity: min={sims.min():.3f} "
          f"mean={sims.mean():.3f} max={sims.max():.3f}")
    print("  top-5 most JD-similar candidates:")
    order = sims.argsort()[::-1][:5]
    for i in order:
        print(f"    {ids[i]}  sim={sims[i]:.3f}")


if __name__ == "__main__":
    main()
