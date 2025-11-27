from typing import List, Dict
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Rerank candidate chunks using a cross-encoder model.
    Each input is (query, chunk_text) -> relevance score.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        id_to_text: Dict[int, str],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Args:
            query: user query string
            candidates: list of dicts from hybrid_fusion, each with:
                        { "chunk_id", "dense_score", "sparse_score", "final_score" }
            id_to_text: mapping from chunk_id -> chunk text
            top_k: how many reranked results to return

        Returns:
            New list of dicts, sorted by cross-encoder score desc, with added "ce_score".
        """
        if not candidates:
            return []

        # Build pairs (query, chunk_text)
        pairs = []
        for item in candidates:
            cid = item["chunk_id"]
            text = id_to_text.get(cid, "")
            pairs.append((query, text))

        # Get cross-encoder scores
        ce_scores = self.model.predict(pairs).tolist()

        # Attach ce_score to candidates
        for item, ce in zip(candidates, ce_scores):
            item["ce_score"] = ce

        # Sort by ce_score desc and keep top_k
        reranked = sorted(candidates, key=lambda x: x["ce_score"], reverse=True)
        return reranked[:top_k]
