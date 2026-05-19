from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Dict, Iterable, List, Sequence

import httpx


def embedding_env() -> Dict[str, str]:
    return {
        "base_url": os.environ.get("RBM_EMBED_BASE_URL", "").strip().rstrip("/"),
        "api_key": os.environ.get("RBM_EMBED_API_KEY", "").strip(),
        "model": os.environ.get("RBM_EMBED_MODEL", "").strip(),
    }


def _l2_normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm <= 0:
        return [0.0 for _ in vec]
    return [float(x) / norm for x in vec]


def hash_embed_text(text: str, dim: int = 256) -> List[float]:
    vec = [0.0] * dim
    tokens = str(text or "").split()
    if not tokens:
        return vec
    for tok in tokens:
        digest = hashlib.sha1(tok.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    return _l2_normalize(vec)


def openai_embed_texts(
    texts: Iterable[str],
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 600.0,
) -> List[List[float]]:
    if not base_url:
        raise ValueError("RBM_EMBED_BASE_URL is empty")
    if not model:
        raise ValueError("RBM_EMBED_MODEL is empty")
    url = f"{base_url}/embeddings"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body: Dict[str, Any] = {
        "model": model,
        "input": list(texts),
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
    rows = data.get("data") or []
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"bad embedding response: {data.keys()}")
    out: List[List[float]] = []
    for row in sorted(rows, key=lambda x: x.get("index", 0)):
        emb = row.get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError("embedding row missing embedding list")
        out.append(_l2_normalize([float(x) for x in emb]))
    return out


def embed_texts(
    texts: Iterable[str],
    *,
    backend: str,
    base_url: str = "",
    api_key: str = "",
    model: str = "",
    timeout: float = 600.0,
    dim: int = 256,
) -> List[List[float]]:
    items = list(texts)
    if backend == "hash":
        return [hash_embed_text(x, dim=dim) for x in items]
    if backend == "openai":
        return openai_embed_texts(
            items,
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
        )
    raise ValueError(f"unsupported embedding backend: {backend}")


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))

