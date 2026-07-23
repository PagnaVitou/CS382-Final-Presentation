"""
Vector store: turn chunks into vectors and support similarity search over them.

This version uses semantic embeddings when the optional sentence-transformers
package is available and falls back to TF-IDF so the starter still runs
immediately. It also supports an optional FAISS index once the chunk count grows
large while keeping the VectorStore interface (`build`, `query`) unchanged.
"""

from typing import List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ingest import Chunk

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None


class VectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = None
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[Chunk] = []
        self.model = None
        self.index = None
        self.backend = "tfidf"

    def build(self, chunks: List[Chunk]) -> None:
        """Build the retrieval index for all chunks."""
        self.chunks = chunks
        texts = [c.text for c in chunks]

        if SentenceTransformer is not None:
            if self.model is None:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            self.matrix = None
            self.backend = "sentence-transformers"
            self._build_faiss_index_if_possible()
            return

        self.embeddings = None
        self.matrix = self.vectorizer.fit_transform(texts)
        self.backend = "tfidf"
        self.index = None

    def _build_faiss_index_if_possible(self) -> None:
        """Create a FAISS index when the optional package is available and the corpus is large."""
        if faiss is None or self.embeddings is None or len(self.chunks) < 2000:
            self.index = None
            return

        normalized = self.embeddings / np.maximum(
            np.linalg.norm(self.embeddings, axis=1, keepdims=True),
            1e-12,
        )
        index = faiss.IndexFlatIP(normalized.shape[1])
        index.add(normalized.astype("float32"))
        self.index = index

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the top_k (chunk, similarity_score) pairs for a query string."""
        if not self.chunks:
            raise RuntimeError("VectorStore.build() must be called before query().")

        if self.embeddings is not None:
            if self.model is None:
                if SentenceTransformer is None:
                    raise RuntimeError("sentence-transformers is not installed.")
                self.model = SentenceTransformer("all-MiniLM-L6-v2")

            query_embedding = self.model.encode(
                [query_text],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]

            if self.index is not None:
                scores, indices = self.index.search(
                    query_embedding.reshape(1, -1).astype("float32"),
                    top_k,
                )
                ranked_idx = [int(i) for i in indices[0] if i >= 0]
                return [(self.chunks[i], float(scores[0][j])) for j, i in enumerate(ranked_idx)]

            similarity_scores = cosine_similarity(
                query_embedding.reshape(1, -1),
                self.embeddings,
            ).flatten()
            ranked_idx = np.argsort(similarity_scores)[::-1][:top_k]
            return [(self.chunks[i], float(similarity_scores[i])) for i in ranked_idx]

        if self.matrix is None:
            raise RuntimeError("VectorStore.build() must be called before query().")

        query_vec = self.vectorizer.transform([query_text])
        similarity_scores = cosine_similarity(query_vec, self.matrix).flatten()
        ranked_idx = np.argsort(similarity_scores)[::-1][:top_k]
        return [(self.chunks[i], float(similarity_scores[i])) for i in ranked_idx]
