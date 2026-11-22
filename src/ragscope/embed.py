from typing import Protocol

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from ragscope.config import active_models


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    # similarity = 1 - cosine distance only holds for unit-length vectors
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class LocalEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.model.get_embedding_dimension()))
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True)


class OpenAIEmbedder:
    def __init__(self, model_name: str, batch_size: int = 128):
        self.model_name = model_name
        self.client = OpenAI(max_retries=3, timeout=30)  # reads OPENAI_API_KEY
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0))
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            res = self.client.embeddings.create(model=self.model_name, input=texts[i:i + self.batch_size])
            vectors.extend(d.embedding for d in res.data)
        return _normalize(np.array(vectors))


PROVIDERS = {"sentence_transformers": LocalEmbedder, "openai": OpenAIEmbedder}


def make_embedder() -> Embedder:
    provider, model_name = active_models()["embedder"]
    return PROVIDERS[provider](model_name)
