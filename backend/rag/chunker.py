from typing import List, Dict


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs using double newlines as separator.
    Empty/whitespace-only paragraphs are removed.
    """
    raw_paragraphs = text.split("\n\n")
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]
    return paragraphs


def _split_long_paragraph(
    para: str,
    max_chars: int,
    overlap_chars: int,
) -> List[str]:
    """
    Fallback for a single paragraph that is longer than max_chars.
    Split it into multiple overlapping character-based segments.
    """
    segments: List[str] = []
    para = para.strip()
    n = len(para)
    if not para:
        return segments

    start = 0
    while start < n:
        end = start + max_chars
        segment = para[start:end]
        segments.append(segment.strip())

        if overlap_chars > 0:
            start = end - overlap_chars
        else:
            start = end

    return segments


def simple_chunk(
    text: str,
    max_chars: int = 800,
    overlap_chars: int = 200,
) -> List[Dict]:
    """
    Paragraph-aware chunking with fallback for very long paragraphs.

    1. Split document into paragraphs (by blank lines).
    2. Group consecutive paragraphs into chunks up to max_chars.
    3. Apply character overlap between chunks.
    4. If a single paragraph is longer than max_chars, split that
       paragraph itself into overlapping character-based segments.

    Returns:
        [
          {"id": 0, "text": "..."},
          {"id": 1, "text": "..."},
          ...
        ]
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = _split_into_paragraphs(text)
    chunks: List[Dict] = []

    chunk_id = 0
    current_chunk = ""

    for para in paragraphs:
        # Case 1: normal paragraph (short/medium)
        if len(para) <= max_chars:
            if current_chunk:
                # If adding this para would overflow current chunk -> finalize and start new
                if len(current_chunk) + len(para) + 2 > max_chars:
                    chunks.append({"id": chunk_id, "text": current_chunk.strip()})
                    chunk_id += 1

                    # Build overlap seed for next chunk
                    if overlap_chars > 0:
                        overlap_seed = current_chunk[-overlap_chars:]
                        current_chunk = overlap_seed.strip() + "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    # Still room in this chunk
                    current_chunk += "\n\n" + para
            else:
                # First paragraph in a new chunk
                current_chunk = para

        # Case 2: very long paragraph – split within the paragraph itself
        else:
            # First, flush any existing current_chunk
            if current_chunk.strip():
                chunks.append({"id": chunk_id, "text": current_chunk.strip()})
                chunk_id += 1
                current_chunk = ""

            long_segments = _split_long_paragraph(para, max_chars, overlap_chars)
            for seg in long_segments:
                if not seg:
                    continue
                chunks.append({"id": chunk_id, "text": seg})
                chunk_id += 1

    # Add last chunk if any text remains
    if current_chunk.strip():
        chunks.append({"id": chunk_id, "text": current_chunk.strip()})

    return chunks
