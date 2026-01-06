"""Preference modeling for style personalization.

This module maintains a user preference vector derived from image embeddings
and explicit 1–5 ratings. The resulting vector can be used to pre-filter
garments by style relevance before fit validation (b_spec vs g_spec).

Embeddings:
        Expected shape (D,) flattened. If provided as (32, 1024) it will be
        flattened automatically to length 32768.

Rating -> weight mapping (tunable):
        1 -> -1.00  (strong dislike)
        2 -> -0.25  (mild dislike)
        3 ->  0.00  (neutral)
        4 -> +0.50  (like)
        5 -> +1.00  (strong like)

Update rule (online incremental):
        pref <- norm( pref + lr * w * norm(embedding) )
        If weight is negative the embedding direction is *subtracted*.

Batch construction (optional):
        Weighted average using the mapped weights; falls back to online updates.

The preference vector is always L2 normalized so cosine similarity is a direct
relevance score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import json
import numpy as np


def _flatten_embedding(embedding: np.ndarray) -> np.ndarray:
    """Ensure embedding is a 1D float32 array."""
    if embedding.ndim > 1:
        embedding = embedding.reshape(-1)
    return embedding.astype(np.float32, copy=False)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def rating_to_weight(rating: int) -> float:
    """Map a 1–5 rating to a signed preference weight."""
    mapping = {1: -1.0, 2: -0.25, 3: 0.0, 4: 0.5, 5: 1.0}
    return mapping.get(int(rating), 0.0)


@dataclass
class PreferenceConfig:
    dimension: int
    learning_rate: float = 0.1
    min_positive_weight: float = 0.25  # threshold for counting as positive signal
    weight_clip: float | None = 2.0  # max absolute effective weight
    decay: float = 0.0  # optional multiplicative decay per update


class PreferenceModel:
    """Online preference modeling for user style personalization."""

    def __init__(self, config: PreferenceConfig, seed: int | None = None) -> None:
        self._cfg = config
        rng = np.random.default_rng(seed)
        # Random small init to avoid zero vector issues; normalize.
        self._vector = _l2_normalize(
            rng.normal(0.0, 0.01, size=config.dimension).astype(np.float32)
        )
        self._total_updates = 0
        self._positive_count = 0
        self._negative_count = 0

    @property
    def dimension(self) -> int:  # external read-only
        return self._cfg.dimension

    @property
    def vector(self) -> np.ndarray:
        return self._vector

    @property
    def stats(self) -> dict:
        return {
            "updates": self._total_updates,
            "positive": self._positive_count,
            "negative": self._negative_count,
        }

    def update(self, embedding: np.ndarray, rating: int) -> None:
        """Apply a single (embedding, rating) update to the preference vector."""
        emb = _flatten_embedding(embedding)
        if emb.shape[0] != self.dimension:
            raise ValueError(
                f"Embedding dimension {emb.shape[0]} != model dimension {self.dimension}"
            )
        w = rating_to_weight(rating)
        if w == 0.0:
            return  # neutral, skip
        if self._cfg.weight_clip is not None:
            w = float(np.clip(w, -self._cfg.weight_clip, self._cfg.weight_clip))
        # Track counts
        if w > 0:
            self._positive_count += 1
        else:
            self._negative_count += 1

        emb_norm = _l2_normalize(emb)
        # Decay existing preference if configured
        if self._cfg.decay > 0:
            self._vector *= 1.0 - self._cfg.decay

        self._vector = _l2_normalize(
            self._vector + self._cfg.learning_rate * w * emb_norm
        )
        self._total_updates += 1

    def bulk_update(
        self, embeddings: Sequence[np.ndarray], ratings: Sequence[int]
    ) -> None:
        if len(embeddings) != len(ratings):
            raise ValueError("Embeddings and ratings length mismatch")
        for emb, r in zip(embeddings, ratings):
            self.update(emb, r)

    def build_from_batch(
        self, embeddings: Sequence[np.ndarray], ratings: Sequence[int]
    ) -> None:
        """Rebuild preference vector from scratch using weighted average of a batch.

        This discards the current vector and replaces it.
        """
        if len(embeddings) == 0:
            return
        if len(embeddings) != len(ratings):
            raise ValueError("Embeddings and ratings length mismatch")
        dim = self.dimension
        accum = np.zeros(dim, dtype=np.float32)
        total_weight = 0.0
        pos = neg = 0
        for emb, r in zip(embeddings, ratings):
            embf = _flatten_embedding(emb)
            if embf.shape[0] != dim:
                raise ValueError(
                    f"Embedding dimension {embf.shape[0]} != model dimension {dim}"
                )
            w = rating_to_weight(r)
            if w == 0.0:
                continue
            if w > 0:
                pos += 1
            else:
                neg += 1
            emb_norm = _l2_normalize(embf)
            accum += w * emb_norm
            total_weight += abs(w)
        if total_weight == 0.0:
            return
        self._vector = _l2_normalize(accum)
        self._total_updates = len(ratings)
        self._positive_count = pos
        self._negative_count = neg

    def score(self, embedding: np.ndarray) -> float:
        """Return cosine similarity of an embedding to the preference vector."""
        emb = _flatten_embedding(embedding)
        if emb.shape[0] != self.dimension:
            raise ValueError(
                f"Embedding dimension {emb.shape[0]} != model dimension {self.dimension}"
            )
        emb_norm = _l2_normalize(emb)
        return float(np.dot(self._vector, emb_norm))  # already normalized

    def rank(self, embeddings: Sequence[np.ndarray]) -> List[float]:
        """Return list of scores for given embeddings (parallelizable externally)."""
        return [self.score(e) for e in embeddings]

    def serialize(self) -> str:
        """Serialize model state to JSON string."""
        payload = {
            "config": {
                "dimension": self.dimension,
                "learning_rate": self._cfg.learning_rate,
                "min_positive_weight": self._cfg.min_positive_weight,
                "weight_clip": self._cfg.weight_clip,
                "decay": self._cfg.decay,
            },
            "vector": self._vector.astype(np.float32).tolist(),
            "stats": self.stats,
        }
        return json.dumps(payload)

    @classmethod
    def deserialize(cls, data: str) -> PreferenceModel:
        obj = json.loads(data)
        cfg = PreferenceConfig(
            dimension=obj["config"]["dimension"],
            learning_rate=obj["config"]["learning_rate"],
            min_positive_weight=obj["config"].get("min_positive_weight", 0.25),
            weight_clip=obj["config"].get("weight_clip"),
            decay=obj["config"].get("decay", 0.0),
        )
        model = cls(cfg)
        vec_list = obj.get("vector")
        if vec_list:
            v = np.array(vec_list, dtype=np.float32)
            if v.shape[0] != cfg.dimension:
                raise ValueError("Serialized vector dimension mismatch")
            model._vector = _l2_normalize(v)
        # Restore counters heuristically (optional)
        stats = obj.get("stats", {})
        model._total_updates = stats.get("updates", 0)
        model._positive_count = stats.get("positive", 0)
        model._negative_count = stats.get("negative", 0)
        return model


# ----------------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    D = 32768  # example dimension (32 * 1024)
    cfg = PreferenceConfig(dimension=D, learning_rate=0.2)
    model = PreferenceModel(cfg, seed=42)

    # Simulate 5 embeddings + ratings
    rng = np.random.default_rng(0)
    embeddings = [rng.normal(0, 1, size=D).astype(np.float32) for _ in range(5)]
    ratings = [5, 4, 3, 2, 1]  # like -> neutral -> dislike

    model.bulk_update(embeddings, ratings)
    print("Preference stats:", model.stats)
    print("Vector norm (should be 1):", np.linalg.norm(model.vector))
    print("Scores for the same embeddings:")
    for i, emb in enumerate(embeddings):
        print(f"  emb[{i}] rating={ratings[i]} score={model.score(emb):.4f}")

    # Serialize & restore
    serialized = model.serialize()
    restored = PreferenceModel.deserialize(serialized)
    print("Restored equal norm:", np.linalg.norm(restored.vector))
    print("Restored first score:", restored.score(embeddings[0]))
