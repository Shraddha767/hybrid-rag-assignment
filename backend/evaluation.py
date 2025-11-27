import time
from typing import List, Dict

from rag.pipeline import HybridRAGPipeline
from typing import List, Dict

EVAL_QUERIES: List[Dict] = [
    # 1) High-level definition of RAG
    {
        "query": "What is retrieval-augmented generation and why is it needed for large language models?",
        # Intro + overview + typical RAG explanation
        "relevant_chunks": [3, 13, 16],
    },

    # 2) Typical RAG pipeline steps
    {
        "query": "What are the three main steps in a typical RAG process?",
        # Fig.2 description: indexing, retrieval, generation
        "relevant_chunks": [16, 17, 18],
    },

    # 3) Naive vs Advanced vs Modular RAG
    {
        "query": "What are Naive RAG, Advanced RAG, and Modular RAG and how do they differ?",
        # Paradigm comparison text
        "relevant_chunks": [22, 23, 24, 27, 31],
    },

    # 4) Why hybrid retrieval (keyword + semantic) is useful
    {
        "query": "Why do we need hybrid retrieval strategies in RAG systems?",
        # Mentions hybrid retrieval integrating keyword, semantic, vector
        "relevant_chunks": [32],
    },

    # 5) RAG vs fine-tuning vs prompt engineering
    {
        "query": "How does RAG compare to fine-tuning and prompt engineering in terms of external knowledge and model adaptation?",
        # RAG vs FT discussion + Fig.4
        "relevant_chunks": [34, 35, 36, 37, 47],
    },

    # 6) Retrieval sources and data types
    {
        "query": "What retrieval sources and data types can RAG use beyond plain text documents?",
        # Retrieval source: semi-structured (PDF, tables), structured (KG), etc.
        "relevant_chunks": [39, 48, 49, 50, 51],
    },

    # 7) Retrieval granularity
    {
        "query": "What is retrieval granularity in RAG and what levels of granularity are discussed?",
        # Token → phrase → sentence → chunk → document; entity/triplet/sub-graph
        "relevant_chunks": [53, 54, 55],
    },

    # 8) Structural / hierarchical indexes and knowledge graphs
    {
        "query": "How can structural or hierarchical indexes, including knowledge graphs, improve retrieval in RAG?",
        # Hierarchical index, KG-based index, nodes/edges, etc.
        "relevant_chunks": [59, 60, 61],
    },

    # 9) Embedding models for RAG & how to choose them
    {
        "query": "How are embedding models used and evaluated in RAG, and what guidance does the survey give on choosing an embedding model?",
        # Embedding section + MTEB leaderboard mention
        "relevant_chunks": [68, 69, 70, 71, 72],
    },

    # 10) Retrieval metrics like Hit Rate, MRR, NDCG
    {
        "query": "Which metrics are commonly used to evaluate the retrieval module in RAG, such as Hit Rate, MRR, and NDCG?",
        # Retrieval metrics description + table summary
        "relevant_chunks": [99, 109],
    },

    # 11) Evaluation aspects (context relevance, faithfulness, robustness, etc.)
    {
        "query": "What evaluation aspects are used to assess RAG models, including context relevance, faithfulness, answer relevance, and robustness?",
        # Evaluation aspects and abilities
        "relevant_chunks": [100, 101, 102, 103, 104],
    },

    # 12) Long-context LLMs vs RAG
    {
        "query": "How do very long-context LLMs affect the role of RAG in long-document question answering?",
        # Discussion of 200k token context, how RAG still matters
        "relevant_chunks": [112, 113, 114],
    },
]


def hit_at_k(ranked_ids: List[int], relevant_ids: List[int], k: int = 5) -> float:
    top_k = ranked_ids[:k]
    return 1.0 if any(rid in top_k for rid in relevant_ids) else 0.0


def mrr_at_k(ranked_ids: List[int], relevant_ids: List[int], k: int = 5) -> float:
    top_k = ranked_ids[:k]
    for rank, cid in enumerate(top_k, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def main():
    pipeline = HybridRAGPipeline()

    hits = []
    mrrs = []
    latencies = []

    for item in EVAL_QUERIES:
        q = item["query"]
        rel = item["relevant_chunks"]

        start = time.time()
        result = pipeline.answer_question(q, final_k=5)
        elapsed = (time.time() - start) * 1000  # ms

        # ranked chunk_ids from scores
        ranked_ids = [s["chunk_id"] for s in result["scores"]]

        h5 = hit_at_k(ranked_ids, rel, k=5)
        m5 = mrr_at_k(ranked_ids, rel, k=5)

        hits.append(h5)
        mrrs.append(m5)
        latencies.append(elapsed)

        print(f"Query: {q}")
        print(f"  Relevant: {rel}")
        print(f"  Ranked:   {ranked_ids}")
        print(f"  Hit@5: {h5:.3f}, MRR@5: {m5:.3f}, latency: {elapsed:.1f} ms\n")

    avg_hit5 = sum(hits) / len(hits)
    avg_mrr5 = sum(mrrs) / len(mrrs)
    avg_lat = sum(latencies) / len(latencies)

    print("==== Overall ====")
    print(f"Avg Hit@5: {avg_hit5:.3f}")
    print(f"Avg MRR@5: {avg_mrr5:.3f}")
    print(f"Avg latency (ms): {avg_lat:.1f}")


if __name__ == "__main__":
    main()
