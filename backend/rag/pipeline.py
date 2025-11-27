import os
import time
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

from .config import PDF_PATH
from .pdf_loader import load_pdf
from .chunker import simple_chunk
from .embeddings import EmbeddingModel
from .vector_store import FaissIndex
from .bm25_store import BM25Store
from .retriver import hybrid_fusion
from .reranker import CrossEncoderReranker

# Load environment variables from .env (including OPENAI_API_KEY)
load_dotenv()


class HybridRAGPipeline:
    """
    End-to-end Hybrid RAG pipeline:
    - Load + chunk PDF
    - Build dense (FAISS) and sparse (BM25) indexes
    - Hybrid fusion + cross-encoder rerank
    - LLM answer generation via OpenAI
    """

    def __init__(self):
        # 1) Load and chunk the PDF
        full_text = load_pdf(PDF_PATH)
        self.chunks = simple_chunk(full_text)
        self.chunk_texts = [c["text"] for c in self.chunks]
        self.chunk_ids = [c["id"] for c in self.chunks]
        self.id_to_text: Dict[int, str] = {
            c["id"]: c["text"] for c in self.chunks
        }

        # 2) Embeddings + FAISS dense index
        self.embed_model = EmbeddingModel()
        embeddings = self.embed_model.encode(self.chunk_texts)
        dim = embeddings.shape[1]

        self.dense_index = FaissIndex(dim)
        self.dense_index.build(embeddings, self.chunk_ids)

        # 3) BM25 sparse index
        self.bm25 = BM25Store(self.chunks)

        # 4) Cross-encoder reranker
        self.reranker = CrossEncoderReranker()

        # 5) LLM client (OpenAI)
        self._client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            # OpenAI client will pick up OPENAI_API_KEY from env
            self._client = OpenAI()

    # ----------------- Internal helper ----------------- #

    def _generate_answer(self, query: str, contexts: List[str]) -> str:
        """
        Generate final answer from top-ranked contexts using an OpenAI LLM.
        Falls back to a simple answer if no API key is set.
        """
        if self._client is not None:
            system_prompt = (
                "You are a helpful assistant answering questions strictly "
                "based on the provided document context. "
                "If the context is insufficient, say you are not sure. "
                "Do not fabricate information outside the context."
            )

            context_block = "\n\n---\n\n".join(contexts)

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context_block}\n\n"
                        f"Question: {query}\n\n"
                        f"Answer clearly and concisely based only on the context."
                    ),
                },
            ]

            resp = self._client.chat.completions.create(
                model="gpt-4.1-mini",  # adjust if you want a different model
                messages=messages,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()

        # Fallback if no LLM configured – still valid for testing/evaluation
        joined = "\n\n---\n\n".join(contexts)
        return (
            "LLM not configured (OPENAI_API_KEY not set); this is a placeholder answer.\n\n"
            "Here are the most relevant context snippets from the document:\n\n"
            + joined[:1500]
        )

    # ----------------- Public API ----------------- #

    def answer_question(
        self,
        query: str,
        top_k_dense: int = 20,
        top_k_sparse: int = 20,
        final_k: int = 5,
    ) -> Dict:
        """
        Full query pipeline:
        - Dense retrieval
        - BM25 retrieval
        - Hybrid fusion
        - Cross-encoder rerank
        - LLM answer generation

        Returns a dict matching the assignment's expected JSON shape:
        {
          "answer": "...",
          "sources": ["chunk_12", "chunk_7"],
          "scores": [
            {"chunk_id": 12, "dense_score": ..., "sparse_score": ..., "final_score": ...},
            ...
          ],
          "latency_ms": 143
        }
        """
        start = time.time()

        # 1) Dense retrieval
        q_emb = self.embed_model.encode(query)
        dense_scores, dense_ids = self.dense_index.search(q_emb, top_k=top_k_dense)

        # 2) Sparse retrieval (BM25)
        sparse_scores, sparse_ids = self.bm25.search(query, top_k=top_k_sparse)

        # 3) Hybrid fusion of dense + sparse scores
        fused = hybrid_fusion(
            dense_scores,
            dense_ids,
            sparse_scores,
            sparse_ids,
            alpha=0.6,          # dense weight
            overlap_bonus=0.05, # bonus if high in both
            top_k=max(top_k_dense, top_k_sparse),
            overlap_k=5,
        )

        # 4) Cross-encoder rerank over fused candidates
        reranked = self.reranker.rerank(
            query=query,
            candidates=fused,
            id_to_text=self.id_to_text,
            top_k=final_k,
        )

        # 5) Extract final contexts and generate answer
        contexts = [self.id_to_text[item["chunk_id"]] for item in reranked]
        answer = self._generate_answer(query, contexts)

        latency_ms = int((time.time() - start) * 1000)

        # Prepare scores list in the expected format (we don’t expose ce_score)
        scores_output = [
            {
                "chunk_id": item["chunk_id"],
                "dense_score": float(item["dense_score"]),
                "sparse_score": float(item["sparse_score"]),
                "final_score": float(item["final_score"]),
            }
            for item in reranked
        ]

        sources = [f"chunk_{item['chunk_id']}" for item in reranked]

        return {
            "answer": answer,
            "sources": sources,
            "scores": scores_output,
            "latency_ms": latency_ms,
        }
