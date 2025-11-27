# scripts/inspect_chunks.py

from rag.pipeline import HybridRAGPipeline

def main():
    pipeline = HybridRAGPipeline()

    # I’m assuming pipeline stores chunks as a list of dicts like:
    # [{"id": 0, "text": "..."} , ...]
    # If your attribute name is different, adjust `pipeline.chunks`.
    chunks = pipeline.chunks  

    print(f"Total chunks: {len(chunks)}\n")

    for ch in chunks:
        cid = ch["id"]
        text = ch["text"].replace("\n", " ")
        preview = text[:300]  # first 300 chars

        print(f"chunk_id = {cid}")
        print(preview)
        print("-" * 80)

if __name__ == "__main__":
    main()
