from __future__ import annotations
import asyncio
from sentence_transformers import SentenceTransformer
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformerEmbeddingProvider:

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        try:
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(model_name)

    @property
    def dimensions(self) -> int:
        return self._model.get_embedding_dimension()

    async def embed(self, text: str) -> list[float]:
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, texts)

    def embed_sync(self, text: str) -> list[float]:
        """Synchronous variant for use in non-async ingestion scripts."""
        return self._encode([text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts,
            normalize_embeddings=True,
        ).tolist()
