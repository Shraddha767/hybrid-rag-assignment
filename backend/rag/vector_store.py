from typing import List, Tuple
import numpy as np
import faiss


class FaissIndex:
    """
    Simple FAISS index using inner product (with L2-normalized vectors)
    to approximate cosine similarity.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product
        self.chunk_ids: List[int] = []
        self._built = False

    def build(self, embeddings: np.ndarray, chunk_ids: List[int]) -> None:
        """
        Build the FAISS index from embeddings and corresponding chunk IDs.
        Embeddings expected shape: (N, dim)
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected embeddings of shape (N, {self.dim}), got {embeddings.shape}")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.chunk_ids = list(chunk_ids)
        self._built = True

    def search(self, query_emb: np.ndarray, top_k: int = 10) -> Tuple[List[float], List[int]]:
        """
        Search top_k most similar chunks.
        query_emb expected shape: (1, dim)
        Returns:
            - scores: List[float] (similarities)
            - ids:    List[int]   (chunk IDs corresponding to those scores)
        """
        if not self._built:
            raise RuntimeError("FAISS index not built yet. Call build() first.")

        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        # Normalize query embedding
        q = query_emb.astype("float32").copy()
        faiss.normalize_L2(q)

        scores, idxs = self.index.search(q, top_k)
        scores = scores[0]
        idxs = idxs[0]

        # Map FAISS indices back to our chunk_ids
        result_ids = [self.chunk_ids[i] for i in idxs]
        return list(scores), result_ids
