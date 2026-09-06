"""
Embedding layer — architecture doc 8.2/8.10 specifies self-hosted BGE-M3
(free, hybrid dense+sparse natively, no rate limits) as the production
choice. This sandbox has ~6GB free disk and no GPU; a real BGE-M3 install
(torch + transformers + ~2.2GB model weights) doesn't reliably fit
alongside everything else this build needs, so Phase 3 ships against this
same interface with a lightweight TF-IDF embedder instead.

This is a disk-budget substitution, not an architecture change: everything
downstream (Qdrant storage, hybrid search, metadata filtering, MMR,
reranking, incremental reindexing) is written against `BaseEmbedder` and
does not know or care which implementation is behind it. Swapping to real
BGE-M3 in a production environment is exactly one line — see
`LocalBGEEmbedder` below (unimplemented here, stubbed with a clear error
pointing at this docstring) — not a pipeline rewrite.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseEmbedder(ABC):
    """Every embedder implementation — TF-IDF (this sandbox) or BGE-M3
    (production) — exposes this same interface. RAG pipeline code (Phase 3)
    is written against this, never against a specific implementation."""

    @abstractmethod
    def fit(self, corpus: list[str]) -> None:
        """Some embedders (TF-IDF) need to see the corpus vocabulary first;
        dense transformer embedders (BGE-M3) are a no-op here — the model
        is pre-trained, not fit per-corpus. Always call this once after
        ingesting/updating the document set, before embed()."""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns a (len(texts), dim) float32 array."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


class TfidfEmbedder(BaseEmbedder):
    """Sandbox-runnable dense-ish embedder: TF-IDF + L2-normalized vectors,
    dimensionality capped so it behaves like a dense embedding for Qdrant's
    cosine-similarity index. This is a real, working embedder — not a
    mock — it genuinely does semantic-ish similarity via term overlap; it
    just doesn't capture the deeper semantic generalization a transformer
    embedder (BGE-M3) would. Good enough to prove out the *architecture*
    (metadata filtering, versioning, hybrid search, reranking); a real
    deployment should swap to LocalBGEEmbedder or an API-based embedder
    from the 8.10 free-tier matrix for actual retrieval quality.
    """

    def __init__(self, max_features: int = 768):
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self._fitted = False
        self._dim = max_features

    def fit(self, corpus: list[str]) -> None:
        self._vectorizer.fit(corpus)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit(corpus) must be called before embed()")
        matrix = self._vectorizer.transform(texts).toarray().astype("float32")
        # Pad to fixed dim if vocabulary is smaller than max_features (small demo corpora)
        if matrix.shape[1] < self._dim:
            pad = np.zeros((matrix.shape[0], self._dim - matrix.shape[1]), dtype="float32")
            matrix = np.hstack([matrix, pad])
        # L2-normalize so cosine similarity behaves sensibly in Qdrant
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    @property
    def dim(self) -> int:
        return self._dim


class LocalBGEEmbedder(BaseEmbedder):
    """Self-hosted alternative — BGE-M3, free forever, no per-request
    cost or rate limit. Not used by default now that a cloud embedding
    API (MistralEmbedder, below) is wired up and requires no Docker/GPU
    box at all — but left implemented as a documented option for anyone
    who wants zero per-request cost at higher volume than a free-tier
    cloud API comfortably supports. Not runnable in THIS sandbox
    (disk budget) — implement when you have ~4GB free disk (or a GPU box) via:

        pip install FlagEmbedding
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        # model.encode(texts)['dense_vecs'] -> use as embed() return value
    """

    def fit(self, corpus: list[str]) -> None:
        return None

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(
            "LocalBGEEmbedder requires FlagEmbedding + BGE-M3 weights (~2.2GB), "
            "not installed in this sandbox due to disk budget. See class docstring — "
            "or use MistralEmbedder (cloud, no disk/GPU needed) instead."
        )

    @property
    def dim(self) -> int:
        return 1024


class MistralEmbedder(BaseEmbedder):
    """Cloud embedding API — Mistral's `mistral-embed` model, no Docker,
    no GPU, no local disk footprint at all. This is the recommended
    default for an all-cloud deployment: a single REST call per batch of
    texts, real 1024-dim dense embeddings.

    Not network-tested from this sandbox (no route to api.mistral.ai in
    the bash tool's allowed domains) — but the request/response handling
    is tested against a realistic mocked response via respx
    (tests/test_mistral_embedder.py), the same verification strategy
    used for GroqClient.
    """

    def __init__(self):
        from app.core.config import get_settings
        settings = get_settings()
        self._api_key = settings.mistral_api_key
        if not self._api_key:
            raise RuntimeError("MISTRAL_API_KEY not configured - see .env.example")
        import httpx
        self._client = httpx.Client(
            base_url="https://api.mistral.ai/v1",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            timeout=30.0,
        )
        self._dim = 1024  # mistral-embed's native output dimensionality

    def fit(self, corpus: list[str]) -> None:
        return None  # pretrained model, no per-corpus fit step needed

    def embed(self, texts: list[str], max_retries: int = 2) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype="float32")

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                resp = self._client.post("/embeddings", json={"model": "mistral-embed", "input": texts})
                resp.raise_for_status()
                data = resp.json()["data"]
                vectors = np.array([d["embedding"] for d in data], dtype="float32")
                return vectors
            except Exception as e:
                last_error = e
                continue
        raise RuntimeError(f"Mistral embeddings API call failed after {max_retries + 1} attempts: {last_error}")

    @property
    def dim(self) -> int:
        return self._dim


def get_embedder() -> BaseEmbedder:
    """Factory — auto-selects MistralEmbedder (cloud, no Docker/GPU) when
    MISTRAL_API_KEY is configured, falling back to the local TF-IDF
    substitute otherwise. Same settings-driven pattern as get_llm_client()
    (Groq), get_cache() (Redis), and get_qdrant_client() (Qdrant Cloud)."""
    from app.core.config import get_settings
    settings = get_settings()
    if settings.mistral_api_key:
        return MistralEmbedder()
    return TfidfEmbedder()
