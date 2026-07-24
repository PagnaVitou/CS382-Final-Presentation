"""
Vector store: turn chunks into vectors and support similarity search over them.

Upgraded from TF-IDF to real sentence embeddings using sentence-transformers.
This provides better semantic understanding of document chunks while maintaining
the same VectorStore interface for backward compatibility with app.py.

Upgrade path (if corpus grows past ~10k chunks):
- Swap the in-memory cosine_similarity search for FAISS or Chroma for faster retrieval.
- Keep the VectorStore interface (`build`, `query`) the same so app.py doesn't change.
"""

from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .ingest import Chunk

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class VectorStore:
    def __init__(self, use_embeddings: bool = True):
        """
        Initialize the vector store.
        
        Args:
            use_embeddings: If True, use sentence-transformers for real embeddings.
                           If False, fall back to TF-IDF.
        """
        self.use_embeddings = use_embeddings and HAS_SENTENCE_TRANSFORMERS
        self.model = None
        self.matrix = None
        self.chunks: List[Chunk] = []
        
        if self.use_embeddings:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words="english")

    def build(self, chunks: List[Chunk]) -> None:
        """Encode all chunk texts into vectors using embeddings or TF-IDF."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        
        if self.use_embeddings:
            self.matrix = self.model.encode(texts, convert_to_numpy=True)
        else:
            self.matrix = self.vectorizer.fit_transform(texts)

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the top_k (chunk, similarity_score) pairs for a query string."""
        if self.matrix is None:
            raise RuntimeError("VectorStore.build() must be called before query().")
        
        if self.use_embeddings:
            query_vec = self.model.encode([query_text], convert_to_numpy=True)
        else:
            query_vec = self.vectorizer.transform([query_text]).toarray()
        
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        
        # Filter out results with very low similarity scores
        results = [(self.chunks[i], float(scores[i])) for i in ranked_idx if scores[i] > 0.0]
        return results
