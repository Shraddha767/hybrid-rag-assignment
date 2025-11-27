from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi


class BM25Store:
    """
    Simple BM25 index over chunk texts.
    """

    def __init__(self, chunks: List[Dict]):
        """
        chunks: list of {"id": int, "text": str}
        """
        self.chunks = chunks
        self.chunk_ids = [c["id"] for c in chunks]

        # very simple tokenization: lowercase + split on whitespace
        self.tokenized_docs: List[List[str]] = [
            self._tokenize(c["text"]) for c in chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def search(self, query: str, top_k: int = 10) -> Tuple[List[float], List[int]]:
        """
        Return top_k chunks ranked by BM25 score.
        """
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)  # numpy array length N

        # get indices of top_k scores
        import numpy as np
        scores = np.array(scores)
        idxs = np.argsort(-scores)[:top_k]

        top_scores = scores[idxs].tolist()
        top_ids = [self.chunk_ids[i] for i in idxs]

        return top_scores, top_ids
