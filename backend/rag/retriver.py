from typing import List, Dict


def _normalize(scores: List[float]) -> List[float]:
    """
    Min-max normalize a list of scores to [0, 1].
    If all scores are equal, return 0.5 for all to avoid divide-by-zero.
    """
    if not scores:
        return []

    s_min = min(scores)
    s_max = max(scores)
    if s_max == s_min:
        return [0.5 for _ in scores]

    return [(s - s_min) / (s_max - s_min) for s in scores]


def hybrid_fusion(
    dense_scores: List[float],
    dense_ids: List[int],
    sparse_scores: List[float],
    sparse_ids: List[int],
    alpha: float = 0.6,
    overlap_bonus: float = 0.05,
    top_k: int = 20,
    overlap_k: int = 5,
) -> List[Dict]:
    """
    Custom hybrid scoring function that combines dense (embedding) scores
    and sparse (BM25) scores into a single ranked list.

    Args:
        dense_scores: similarity scores from dense retriever
        dense_ids:    chunk_ids corresponding to dense_scores
        sparse_scores: BM25 scores from sparse retriever
        sparse_ids:    chunk_ids corresponding to sparse_scores
        alpha: weight for dense scores in [0,1]
        overlap_bonus: extra boost if a chunk is high in both lists
        top_k: how many final items to return
        overlap_k: how deep to consider for "high in both" bonus

    Returns:
        List of dicts sorted by final_score desc:
        [
          {
            "chunk_id": int,
            "dense_score": float,
            "sparse_score": float,
            "final_score": float,
          },
          ...
        ]
    """

    # 1. Normalize scores separately to [0,1]
    dense_norm = _normalize(dense_scores)
    sparse_norm = _normalize(sparse_scores)

    # 2. Build maps from chunk_id -> normalized score
    dense_map = {cid: s for cid, s in zip(dense_ids, dense_norm)}
    sparse_map = {cid: s for cid, s in zip(sparse_ids, sparse_norm)}

    # All unique chunk_ids that appeared in either list
    all_ids = set(dense_ids) | set(sparse_ids)

    # 3. Precompute sets for overlap bonus (top overlap_k from each list)
    dense_top_set = set(dense_ids[:overlap_k])
    sparse_top_set = set(sparse_ids[:overlap_k])

    fused: List[Dict] = []

    # 4. Combine scores for each chunk_id
    for cid in all_ids:
        d = dense_map.get(cid, 0.0)
        sp = sparse_map.get(cid, 0.0)

        # Weighted sum of normalized scores
        final = alpha * d + (1.0 - alpha) * sp

        # Optional: bonus if chunk is high in both rankings
        if cid in dense_top_set and cid in sparse_top_set:
            final += overlap_bonus

        fused.append(
            {
                "chunk_id": cid,
                "dense_score": d,
                "sparse_score": sp,
                "final_score": final,
            }
        )

    # 5. Sort by final_score descending and return top_k
    fused_sorted = sorted(fused, key=lambda x: x["final_score"], reverse=True)
    return fused_sorted[:top_k]
