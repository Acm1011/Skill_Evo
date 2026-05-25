from __future__ import annotations

import os

from baselines.ReasoningBankMath.embedding_backend import (
    cosine_similarity,
    embed_texts,
    hash_embed_text,
)


def embedding_env() -> dict[str, str]:
    return {
        "base_url": (
            os.environ.get("MMM_EMBED_BASE_URL", "").strip().rstrip("/")
            or os.environ.get("RBM_EMBED_BASE_URL", "").strip().rstrip("/")
        ),
        "api_key": (
            os.environ.get("MMM_EMBED_API_KEY", "").strip()
            or os.environ.get("RBM_EMBED_API_KEY", "").strip()
        ),
        "model": (
            os.environ.get("MMM_EMBED_MODEL", "").strip()
            or os.environ.get("RBM_EMBED_MODEL", "").strip()
        ),
    }

__all__ = ["cosine_similarity", "embed_texts", "embedding_env", "hash_embed_text"]
